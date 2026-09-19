#include "gpt2/kernels.hpp"
#include "gpt2/linear_executor.hpp"
#include "gpt2/model.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Case {
    std::string_view name;
    std::size_t input_channels;
    std::size_t output_channels;
    std::span<const float> weight;
    std::span<const float> bias;
    std::vector<float> input;
    std::vector<float> output;
};

[[nodiscard]] std::size_t parse_size(std::string_view text) {
    std::size_t value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value == 0) {
        throw std::invalid_argument("warmup, repetitions and inner calls must be positive");
    }
    return value;
}

[[nodiscard]] double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto center = values.size() / 2;
    return values.size() % 2 == 0 ? (values[center - 1] + values[center]) / 2.0
                                  : values[center];
}

[[nodiscard]] Case make_case(std::string_view name,
                             std::size_t input_channels,
                             std::size_t output_channels,
                             std::span<const float> weight,
                             std::span<const float> bias) {
    Case result{name, input_channels, output_channels, weight, bias,
                std::vector<float>(input_channels), std::vector<float>(output_channels)};
    for (std::size_t index = 0; index < input_channels; ++index) {
        result.input[index] = static_cast<float>(index % 31) * 0.01F;
    }
    return result;
}

void run(Case& item, gpt2::LinearExecutor* executor) {
    if (executor == nullptr) {
        gpt2::kernels::linear(item.output, item.input, item.weight, item.bias,
                              1, item.input_channels, item.output_channels);
    } else {
        executor->linear(item.output, item.input, item.weight, item.bias,
                         item.input_channels, item.output_channels);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 5 && argc != 6) {
            std::cerr << "usage: gpt2_linear_benchmark CHECKPOINT [WARMUP REPETITIONS INNER [THREADS]]\n";
            return 2;
        }
        const std::size_t warmup = argc >= 5 ? parse_size(argv[2]) : 1;
        const std::size_t repetitions = argc >= 5 ? parse_size(argv[3]) : 5;
        const std::size_t inner = argc >= 5 ? parse_size(argv[4]) : 8;
        const std::size_t threads = argc == 6 ? parse_size(argv[5]) : 1;
        gpt2::GPT2Model model(gpt2::ModelWeights::load_checkpoint(argv[1]));
        const auto& weights = model.weights();
        const auto& config = weights.config();
        const auto c = config.channels;
        const auto v = config.vocabulary_size;
        std::unique_ptr<gpt2::LinearExecutor> executor;
        if (threads != 1) {
            executor = std::make_unique<gpt2::LinearExecutor>(threads);
        }
        std::vector<Case> cases;
        cases.reserve(5);
        cases.push_back(make_case("qkv", c, 3 * c,
                                  weights.tensor("qkvw").values().first(3 * c * c),
                                  weights.tensor("qkvb").values().first(3 * c)));
        cases.push_back(make_case("attention_projection", c, c,
                                  weights.tensor("attprojw").values().first(c * c),
                                  weights.tensor("attprojb").values().first(c)));
        cases.push_back(make_case("fc", c, 4 * c,
                                  weights.tensor("fcw").values().first(4 * c * c),
                                  weights.tensor("fcb").values().first(4 * c)));
        cases.push_back(make_case("fc_projection", 4 * c, c,
                                  weights.tensor("fcprojw").values().first(4 * c * c),
                                  weights.tensor("fcprojb").values().first(c)));
        cases.push_back(make_case("vocabulary", c, v,
                                  weights.tensor("wte").values().first(v * c), {}));

        std::cout << std::fixed << std::setprecision(3)
                  << "benchmark=single_row_linear\n"
                  << "checkpoint=" << argv[1] << '\n'
                  << "warmup=" << warmup << " repetitions=" << repetitions
                  << " inner_calls=" << inner << " threads=" << threads << '\n';
        double layer_total{};
        for (auto& item : cases) {
            for (std::size_t sample = 0; sample < warmup; ++sample) {
                for (std::size_t call = 0; call < inner; ++call) {
                    run(item, executor.get());
                }
            }
            std::vector<double> samples;
            samples.reserve(repetitions);
            for (std::size_t sample = 0; sample < repetitions; ++sample) {
                const auto begin = Clock::now();
                for (std::size_t call = 0; call < inner; ++call) {
                    run(item, executor.get());
                }
                const auto end = Clock::now();
                samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count()
                                  / static_cast<double>(inner));
            }
            const auto middle = median(samples);
            std::cout << "shape=" << item.name
                      << " input=" << item.input_channels
                      << " output=" << item.output_channels
                      << " weights=" << item.weight.size()
                      << " samples_ms=";
            for (std::size_t index = 0; index < samples.size(); ++index) {
                if (index != 0) {
                    std::cout << ',';
                }
                std::cout << samples[index];
            }
            std::cout << " median_ms=" << middle << '\n';
            if (item.name != "vocabulary") {
                layer_total += middle;
            } else {
                std::cout << "estimated_linear_per_token_ms="
                          << config.num_layers * layer_total + middle << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gpt2_linear_benchmark: " << error.what() << '\n';
        return 1;
    }
}
