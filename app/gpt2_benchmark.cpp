#include "gpt2/kernels.hpp"
#include "gpt2/linear_executor.hpp"
#include "gpt2/model.hpp"
#include "gpt2/session.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/utsname.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path checkpoint;
    std::vector<std::int32_t> tokens{15496, 11, 616};
    std::size_t generation_tokens{2};
    std::size_t warmup{1};
    std::size_t repetitions{5};
    std::size_t capacity{};
    std::size_t threads{1};
    bool skip_full_prefix{};
};

struct Samples {
    std::string name;
    std::vector<double> milliseconds;

    [[nodiscard]] double median() const {
        auto sorted = milliseconds;
        std::sort(sorted.begin(), sorted.end());
        const auto middle = sorted.size() / 2;
        if (sorted.size() % 2 != 0) {
            return sorted[middle];
        }
        return (sorted[middle - 1] + sorted[middle]) / 2.0;
    }
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " --checkpoint PATH [--tokens ID[,ID...]]"
                 " [--generation-tokens N] [--capacity N]"
                 " [--warmup N] [--repetitions N] [--threads N]"
                 " [--skip-full-prefix]\n";
}

[[nodiscard]] std::size_t parse_size(std::string_view text, const char* option) {
    std::size_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid value for ") + option);
    }
    return value;
}

[[nodiscard]] std::vector<std::int32_t> parse_tokens(std::string_view text) {
    std::vector<std::int32_t> tokens;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto item = text.substr(0, comma);
        if (item.empty()) {
            throw std::invalid_argument("--tokens contains an empty token ID");
        }
        std::int32_t token = 0;
        const auto result = std::from_chars(item.data(), item.data() + item.size(), token);
        if (result.ec != std::errc{} || result.ptr != item.data() + item.size()) {
            throw std::invalid_argument("--tokens contains an invalid token ID");
        }
        tokens.push_back(token);
        if (comma == std::string_view::npos) {
            break;
        }
        text.remove_prefix(comma + 1);
    }
    if (tokens.empty()) {
        throw std::invalid_argument("--tokens must contain at least one token ID");
    }
    return tokens;
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--checkpoint" && index + 1 < argc) {
            options.checkpoint = argv[++index];
        } else if (argument == "--tokens" && index + 1 < argc) {
            options.tokens = parse_tokens(argv[++index]);
        } else if (argument == "--generation-tokens" && index + 1 < argc) {
            options.generation_tokens = parse_size(argv[++index], "--generation-tokens");
        } else if (argument == "--capacity" && index + 1 < argc) {
            options.capacity = parse_size(argv[++index], "--capacity");
        } else if (argument == "--warmup" && index + 1 < argc) {
            options.warmup = parse_size(argv[++index], "--warmup");
        } else if (argument == "--repetitions" && index + 1 < argc) {
            options.repetitions = parse_size(argv[++index], "--repetitions");
        } else if (argument == "--threads" && index + 1 < argc) {
            options.threads = parse_size(argv[++index], "--threads");
        } else if (argument == "--skip-full-prefix") {
            options.skip_full_prefix = true;
        } else if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown or incomplete argument: " +
                                        std::string(argument));
        }
    }
    if (options.checkpoint.empty()) {
        throw std::invalid_argument("--checkpoint is required");
    }
    if (options.generation_tokens == 0) {
        throw std::invalid_argument("--generation-tokens must be positive");
    }
    if (options.repetitions == 0) {
        throw std::invalid_argument("--repetitions must be positive");
    }
    if (options.threads == 0) {
        throw std::invalid_argument("--threads must be positive");
    }
    return options;
}

template <typename Setup, typename Operation>
[[nodiscard]] Samples measure(std::string name,
                              std::size_t warmup,
                              std::size_t repetitions,
                              Setup&& setup,
                              Operation&& operation) {
    for (std::size_t index = 0; index < warmup; ++index) {
        setup();
        operation();
    }

    Samples result{.name = std::move(name), .milliseconds = {}};
    result.milliseconds.reserve(repetitions);
    for (std::size_t index = 0; index < repetitions; ++index) {
        setup();
        const auto begin = Clock::now();
        operation();
        const auto end = Clock::now();
        result.milliseconds.push_back(
            std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return result;
}

void print_samples(const Samples& samples) {
    std::cout << "metric=" << samples.name << " samples_ms=";
    for (std::size_t index = 0; index < samples.milliseconds.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << samples.milliseconds[index];
    }
    std::cout << " median_ms=" << samples.median() << '\n';
}

[[nodiscard]] std::int32_t greedy_token(std::span<const float> logits) {
    return static_cast<std::int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
}

[[nodiscard]] std::string cpu_model() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    constexpr std::string_view field = "model name";
    while (std::getline(cpuinfo, line)) {
        if (line.starts_with(field)) {
            const auto colon = line.find(':');
            return colon == std::string::npos ? line : line.substr(colon + 2);
        }
    }
    return "unavailable";
}

[[nodiscard]] std::string kernel_release() {
#if defined(__linux__)
    utsname information{};
    if (uname(&information) == 0) {
        return information.release;
    }
#endif
    return "unavailable";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        std::cout << std::fixed << std::setprecision(3);

        const auto load_begin = Clock::now();
        gpt2::GPT2Model model(gpt2::ModelWeights::load_checkpoint(options.checkpoint));
        const auto load_end = Clock::now();
        const double load_milliseconds =
            std::chrono::duration<double, std::milli>(load_end - load_begin).count();

        const auto& config = model.weights().config();
        std::unique_ptr<gpt2::LinearExecutor> executor;
        if (options.threads != 1) {
            executor = std::make_unique<gpt2::LinearExecutor>(options.threads);
        }
        const auto capacity = options.capacity == 0 ? config.max_sequence_length
                                                    : options.capacity;
        if (options.tokens.size() > config.max_sequence_length ||
            options.generation_tokens >
                config.max_sequence_length - options.tokens.size()) {
            throw std::invalid_argument("benchmark generation exceeds model context length");
        }
        const auto consumed_for_generation =
            options.tokens.size() + options.generation_tokens - 1;
        if (capacity < consumed_for_generation || capacity > config.max_sequence_length) {
            throw std::invalid_argument("benchmark capacity cannot hold generation state");
        }
        for (const auto token : options.tokens) {
            if (token < 0 || static_cast<std::size_t>(token) >= config.vocabulary_size) {
                throw std::invalid_argument("benchmark token is outside the real vocabulary");
            }
        }

        std::cout << "benchmark=gpt2_milestone3\n"
                  << "checkpoint=" << options.checkpoint.string() << '\n'
                  << "checkpoint_version=" << model.weights().checkpoint_version() << '\n'
                  << "model=L" << config.num_layers << "_H" << config.num_heads
                  << "_C" << config.channels << "_V" << config.vocabulary_size << '\n'
                  << "prompt_tokens=";
        for (std::size_t index = 0; index < options.tokens.size(); ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            std::cout << options.tokens[index];
        }
        std::cout << '\n'
                  << "prompt_length=" << options.tokens.size() << '\n'
                  << "generation_tokens=" << options.generation_tokens << '\n'
                  << "capacity=" << capacity << '\n'
                  << "warmup=" << options.warmup << '\n'
                  << "repetitions=" << options.repetitions << '\n'
                  << "threads=" << options.threads << '\n'
                  << "cpu=" << cpu_model() << '\n'
                  << "kernel=" << kernel_release() << '\n'
#if defined(__clang__)
                  << "compiler=clang " << __clang_version__ << '\n'
#elif defined(__GNUC__)
                  << "compiler=gcc " << __VERSION__ << '\n'
#else
                  << "compiler=unknown\n"
#endif
#if defined(NDEBUG)
                  << "build_type=Release\n"
#else
                  << "build_type=non-Release\n"
#endif
                  << "fp_contract=off\n"
                  << "metric=model_load samples_ms=" << load_milliseconds
                  << " median_ms=" << load_milliseconds << '\n';

        std::unique_ptr<gpt2::InferenceSession> constructed_session;
        const auto session_creation = measure(
            "session_creation", options.warmup, options.repetitions,
            [&] { constructed_session.reset(); },
            [&] { constructed_session = std::make_unique<gpt2::InferenceSession>(model, capacity); });
        print_samples(session_creation);
        constructed_session.reset();

        gpt2::InferenceSession session(model, capacity);
        std::span<const float> latest_logits;
        const auto prefill = measure(
            "prefill", options.warmup, options.repetitions,
            [&] { session.reset(); },
            [&] { latest_logits = session.prefill(options.tokens, executor.get()); });
        print_samples(prefill);

        session.reset();
        const auto initial_logits = session.prefill(options.tokens, executor.get());
        const auto decode_input = greedy_token(initial_logits);
        const auto decode = measure(
            "single_decode", options.warmup, options.repetitions,
            [&] {
                session.reset();
                latest_logits = session.prefill(options.tokens, executor.get());
            },
            [&] { latest_logits = session.decode(decode_input, executor.get()); });
        print_samples(decode);
        std::cout << "single_decode_context_length=" << options.tokens.size() << '\n';

        gpt2::InferenceWorkspace full_workspace;
        std::vector<std::int32_t> full_tokens;
        std::vector<std::int32_t> last_full_tokens;
        std::optional<Samples> full_generation;
        if (!options.skip_full_prefix) {
            full_generation = measure(
                "full_prefix_generation", options.warmup, options.repetitions,
                [&] { full_tokens = options.tokens; },
                [&] {
                    for (std::size_t step = 0; step < options.generation_tokens; ++step) {
                        const auto all_logits = model.forward(full_tokens, full_workspace);
                        const auto real = all_logits.last(config.padded_vocabulary_size)
                                              .first(config.vocabulary_size);
                        full_tokens.push_back(greedy_token(real));
                    }
                    last_full_tokens = full_tokens;
                });
            print_samples(*full_generation);
        } else {
            std::cout << "full_prefix_generation=skipped\n";
        }

        std::vector<std::int32_t> cached_tokens;
        std::vector<std::int32_t> last_cached_tokens;
        const auto cached_generation = measure(
            "cached_generation", options.warmup, options.repetitions,
            [&] {
                session.reset();
                cached_tokens = options.tokens;
            },
            [&] {
                auto logits = session.prefill(cached_tokens, executor.get());
                for (std::size_t step = 0; step < options.generation_tokens; ++step) {
                    const auto next = greedy_token(logits);
                    cached_tokens.push_back(next);
                    if (step + 1 < options.generation_tokens) {
                        logits = session.decode(next, executor.get());
                    }
                }
                last_cached_tokens = cached_tokens;
            });
        print_samples(cached_generation);
        if (full_generation && last_cached_tokens != last_full_tokens) {
            throw std::runtime_error("full and cached benchmark generation diverged");
        }

        std::vector<float> projection_input(options.tokens.size() * config.channels);
        for (std::size_t index = 0; index < projection_input.size(); ++index) {
            projection_input[index] = static_cast<float>(index % 17) * 0.01F;
        }
        const auto projection_weight = model.weights().tensor("wte").values().first(
            config.vocabulary_size * config.channels);
        std::vector<float> projection_all(options.tokens.size() * config.vocabulary_size);
        std::vector<float> projection_last(config.vocabulary_size);
        const auto all_row_projection = measure(
            "projection_real_vocab_all_prompt_rows", options.warmup, options.repetitions,
            [] {},
            [&] {
                gpt2::kernels::linear(projection_all, projection_input, projection_weight,
                                      {}, options.tokens.size(), config.channels,
                                      config.vocabulary_size);
            });
        print_samples(all_row_projection);
        const auto last_row_projection = measure(
            "projection_real_vocab_last_row", options.warmup, options.repetitions,
            [] {},
            [&] {
                gpt2::kernels::linear(
                    projection_last,
                    std::span<const float>(projection_input).last(config.channels),
                    projection_weight, {}, 1, config.channels, config.vocabulary_size);
            });
        print_samples(last_row_projection);

        if (full_generation) {
            std::cout << "full_over_cached_generation_ratio="
                      << full_generation->median() / cached_generation.median() << '\n';
        }
        std::cout << "all_over_last_projection_ratio="
                  << all_row_projection.median() / last_row_projection.median() << '\n'
                  << "kv_cache_bytes=" << session.cache_bytes() << '\n'
                  << "session_logits_bytes="
                  << config.vocabulary_size * sizeof(float) << '\n'
                  << "m1_final_logits_bytes="
                  << (options.tokens.size() + options.generation_tokens - 1) *
                         config.padded_vocabulary_size * sizeof(float)
                  << '\n'
                  << "generated_tokens=";
        for (std::size_t index = 0; index < last_cached_tokens.size(); ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            std::cout << last_cached_tokens[index];
        }
        std::cout << '\n'
                  << "timing_excludes=model_load,session_creation,logging,logits_dump\n";
        return 0;
    } catch (const std::exception& error) {
        print_usage(argv[0]);
        std::cerr << "gpt2_benchmark: " << error.what() << '\n';
        return 1;
    }
}
