#include "gpt2/model.hpp"
#include "gpt2/linear_executor.hpp"
#include "gpt2/session.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

enum class ExecutionMode {
    full,
    cached,
};

struct Options {
    std::filesystem::path checkpoint;
    std::vector<std::int32_t> tokens;
    std::size_t max_new_tokens{};
    std::filesystem::path dump_last_logits;
    std::optional<std::size_t> capacity;
    std::size_t threads{1};
    ExecutionMode mode{ExecutionMode::full};
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " --checkpoint PATH --tokens ID[,ID...]"
                 " [--max-new-tokens N] [--greedy]"
                 " [--mode full|cached] [--capacity N] [--threads N]"
                 " [--dump-last-logits PATH]\n";
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

[[nodiscard]] ExecutionMode parse_mode(std::string_view text) {
    if (text == "full") {
        return ExecutionMode::full;
    }
    if (text == "cached") {
        return ExecutionMode::cached;
    }
    throw std::invalid_argument("--mode must be full or cached");
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options options;
    bool saw_greedy = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--checkpoint" && index + 1 < argc) {
            options.checkpoint = argv[++index];
        } else if (argument == "--tokens" && index + 1 < argc) {
            options.tokens = parse_tokens(argv[++index]);
        } else if (argument == "--max-new-tokens" && index + 1 < argc) {
            options.max_new_tokens = parse_size(argv[++index], "--max-new-tokens");
        } else if (argument == "--dump-last-logits" && index + 1 < argc) {
            options.dump_last_logits = argv[++index];
        } else if (argument == "--mode" && index + 1 < argc) {
            options.mode = parse_mode(argv[++index]);
        } else if (argument == "--capacity" && index + 1 < argc) {
            options.capacity = parse_size(argv[++index], "--capacity");
        } else if (argument == "--threads" && index + 1 < argc) {
            options.threads = parse_size(argv[++index], "--threads");
        } else if (argument == "--greedy") {
            saw_greedy = true;
        } else if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown or incomplete argument: " +
                                        std::string(argument));
        }
    }
    if (options.checkpoint.empty() || options.tokens.empty()) {
        throw std::invalid_argument("--checkpoint and --tokens are required");
    }
    if (!saw_greedy && options.max_new_tokens != 0) {
        throw std::invalid_argument("generation requires --greedy");
    }
    if (options.mode == ExecutionMode::full && options.capacity.has_value()) {
        throw std::invalid_argument("--capacity is only valid with --mode cached");
    }
    if (options.threads == 0 ||
        (options.mode == ExecutionMode::full && options.threads != 1)) {
        throw std::invalid_argument("--threads must be positive and only cached mode may use more than one");
    }
    return options;
}

void dump_binary(const std::filesystem::path& path, std::span<const float> values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open logits output: " + path.string());
    }
    const auto bytes = values.size_bytes();
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::overflow_error("logits output is too large for stream I/O");
    }
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(bytes));
    if (!output) {
        throw std::runtime_error("failed to write logits output: " + path.string());
    }
}

[[nodiscard]] std::int32_t greedy_token(std::span<const float> logits) {
    const auto maximum = std::max_element(logits.begin(), logits.end());
    return static_cast<std::int32_t>(maximum - logits.begin());
}

void print_step(std::size_t step, std::size_t prefix_length, std::int32_t token) {
    std::cout << "step " << step
              << " prefix_length " << prefix_length
              << " next_token " << token << '\n';
}

void generate_full(Options& options, const gpt2::GPT2Model& model) {
    gpt2::InferenceWorkspace workspace;
    const auto& config = model.weights().config();
    std::span<const float> logits;
    for (std::size_t step = 0; step < options.max_new_tokens; ++step) {
        // This remains the independent Milestone 1 full-prefix control flow.
        logits = model.forward(options.tokens, workspace);
        const auto real_vocabulary =
            logits.last(config.padded_vocabulary_size).first(config.vocabulary_size);
        const auto next = greedy_token(real_vocabulary);
        options.tokens.push_back(next);
        print_step(step + 1, options.tokens.size() - 1, next);
    }

    if (options.max_new_tokens == 0 || !options.dump_last_logits.empty()) {
        logits = model.forward(options.tokens, workspace);
    }
    if (!options.dump_last_logits.empty()) {
        const auto last = logits.last(config.padded_vocabulary_size);
        dump_binary(options.dump_last_logits, last);
        std::cout << "dumped_last_logits: " << options.dump_last_logits.string()
                  << " (" << last.size() << " float32 values)\n";
    }
}

void generate_cached(Options& options, const gpt2::GPT2Model& model) {
    const auto& config = model.weights().config();
    const auto capacity = options.capacity.value_or(config.max_sequence_length);
    const std::size_t generated_decode_count =
        options.max_new_tokens == 0 ? 0 : options.max_new_tokens - 1;
    const std::size_t dump_decode_count =
        !options.dump_last_logits.empty() && options.max_new_tokens != 0 ? 1 : 0;
    const auto required = options.tokens.size() + generated_decode_count + dump_decode_count;
    if (required > capacity) {
        throw std::invalid_argument("session capacity is too small for requested computation");
    }

    gpt2::InferenceSession session(model, capacity);
    std::unique_ptr<gpt2::LinearExecutor> executor;
    if (options.threads != 1) {
        executor = std::make_unique<gpt2::LinearExecutor>(options.threads);
    }
    std::span<const float> logits = session.prefill(options.tokens, executor.get());
    std::optional<std::int32_t> final_generated;
    for (std::size_t step = 0; step < options.max_new_tokens; ++step) {
        const auto next = greedy_token(logits);
        options.tokens.push_back(next);
        final_generated = next;
        print_step(step + 1, options.tokens.size() - 1, next);
        if (step + 1 < options.max_new_tokens) {
            logits = session.decode(next, executor.get());
        }
    }

    if (!options.dump_last_logits.empty()) {
        if (final_generated.has_value()) {
            // The dump describes the complete output sequence, so consume the
            // otherwise-unconsumed final generated token.
            logits = session.decode(*final_generated, executor.get());
        }
        dump_binary(options.dump_last_logits, logits);
        std::cout << "dumped_last_logits: " << options.dump_last_logits.string()
                  << " (" << logits.size() << " float32 values)\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto options = parse_options(argc, argv);
        gpt2::GPT2Model model(gpt2::ModelWeights::load_checkpoint(options.checkpoint));
        const auto& config = model.weights().config();

        if (options.tokens.size() > config.max_sequence_length ||
            options.max_new_tokens > config.max_sequence_length - options.tokens.size()) {
            throw std::invalid_argument("requested output exceeds the checkpoint context length");
        }

        if (options.mode == ExecutionMode::full) {
            generate_full(options, model);
        } else {
            generate_cached(options, model);
        }

        std::cout << "output_tokens:";
        for (const auto token : options.tokens) {
            std::cout << ' ' << token;
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        print_usage(argv[0]);
        std::cerr << "gpt2_generate: " << error.what() << '\n';
        return 1;
    }
}
