// Portions of the scalar GPT-2 kernels in this file are adapted from
// karpathy/llm.c (MIT). C++20 refactoring and cached-attention work are part
// of this project. See LICENSE and THIRD_PARTY_NOTICES.md.

#include "gpt2/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace gpt2::kernels {
namespace {

void require_size(std::span<const float> values,
                  std::size_t expected,
                  const char* name) {
    if (values.size() != expected) {
        throw std::invalid_argument(std::string(name) + " has the wrong size");
    }
}

void require_size(std::span<float> values,
                  std::size_t expected,
                  const char* name) {
    require_size(std::span<const float>(values), expected, name);
}

}  // namespace

void layer_norm(std::span<float> output,
                std::span<const float> input,
                std::span<const float> weight,
                std::span<const float> bias,
                std::size_t rows,
                std::size_t channels) {
    require_size(output, rows * channels, "layer_norm output");
    require_size(input, rows * channels, "layer_norm input");
    require_size(weight, channels, "layer_norm weight");
    require_size(bias, channels, "layer_norm bias");
    if (channels == 0) {
        throw std::invalid_argument("layer_norm channels must be positive");
    }

    constexpr float epsilon = 1.0e-5F;
    for (std::size_t row = 0; row < rows; ++row) {
        const auto base = row * channels;
        float mean = 0.0F;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            mean += input[base + channel];
        }
        mean /= static_cast<float>(channels);

        float variance = 0.0F;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const float shifted = input[base + channel] - mean;
            variance += shifted * shifted;
        }
        variance /= static_cast<float>(channels);
        const float reciprocal_stddev = 1.0F / std::sqrt(variance + epsilon);

        for (std::size_t channel = 0; channel < channels; ++channel) {
            const float normalized = reciprocal_stddev * (input[base + channel] - mean);
            output[base + channel] = normalized * weight[channel] + bias[channel];
        }
    }
}

void linear(std::span<float> output,
            std::span<const float> input,
            std::span<const float> weight,
            std::span<const float> bias,
            std::size_t rows,
            std::size_t input_channels,
            std::size_t output_channels) {
    require_size(output, rows * output_channels, "linear output");
    require_size(input, rows * input_channels, "linear input");
    require_size(weight, output_channels * input_channels, "linear weight");
    if (!bias.empty()) {
        require_size(bias, output_channels, "linear bias");
    }

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out_channel = 0; out_channel < output_channels; ++out_channel) {
            float value = bias.empty() ? 0.0F : bias[out_channel];
            const auto input_base = row * input_channels;
            const auto weight_base = out_channel * input_channels;
            for (std::size_t in_channel = 0; in_channel < input_channels; ++in_channel) {
                value += input[input_base + in_channel] * weight[weight_base + in_channel];
            }
            output[row * output_channels + out_channel] = value;
        }
    }
}

void causal_self_attention(std::span<float> output,
                           std::span<const float> qkv,
                           std::span<float> score_scratch,
                           std::size_t sequence_length,
                           std::size_t channels,
                           std::size_t num_heads) {
    require_size(output, sequence_length * channels, "attention output");
    require_size(qkv, sequence_length * 3 * channels, "attention qkv");
    require_size(score_scratch, sequence_length, "attention scratch");
    if (num_heads == 0 || channels == 0 || channels % num_heads != 0) {
        throw std::invalid_argument("invalid attention dimensions");
    }

    const std::size_t head_size = channels / num_heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));
    std::fill(output.begin(), output.end(), 0.0F);

    for (std::size_t position = 0; position < sequence_length; ++position) {
        for (std::size_t head = 0; head < num_heads; ++head) {
            const auto query_base = position * 3 * channels + head * head_size;
            float maximum = std::numeric_limits<float>::lowest();

            for (std::size_t source = 0; source <= position; ++source) {
                const auto key_base = source * 3 * channels + channels + head * head_size;
                float score = 0.0F;
                for (std::size_t index = 0; index < head_size; ++index) {
                    score += qkv[query_base + index] * qkv[key_base + index];
                }
                score *= scale;
                score_scratch[source] = score;
                maximum = std::max(maximum, score);
            }

            float exponential_sum = 0.0F;
            for (std::size_t source = 0; source <= position; ++source) {
                const float value = std::exp(score_scratch[source] - maximum);
                score_scratch[source] = value;
                exponential_sum += value;
            }
            const float inverse_sum = exponential_sum == 0.0F ? 0.0F : 1.0F / exponential_sum;

            const auto output_base = position * channels + head * head_size;
            for (std::size_t source = 0; source <= position; ++source) {
                const auto value_base = source * 3 * channels + 2 * channels + head * head_size;
                const float probability = score_scratch[source] * inverse_sum;
                for (std::size_t index = 0; index < head_size; ++index) {
                    output[output_base + index] += probability * qkv[value_base + index];
                }
            }
        }
    }
}

void cached_self_attention(std::span<float> output,
                           std::span<const float> query,
                           std::span<const float> key_cache,
                           std::span<const float> value_cache,
                           std::span<float> score_scratch,
                           std::size_t visible_length,
                           std::size_t capacity,
                           std::size_t channels,
                           std::size_t num_heads) {
    require_size(output, channels, "cached attention output");
    require_size(query, channels, "cached attention query");
    require_size(key_cache, capacity * channels, "cached attention keys");
    require_size(value_cache, capacity * channels, "cached attention values");
    require_size(score_scratch, capacity, "cached attention scratch");
    if (visible_length == 0 || visible_length > capacity) {
        throw std::invalid_argument("cached attention visible length is out of range");
    }
    if (num_heads == 0 || channels == 0 || channels % num_heads != 0) {
        throw std::invalid_argument("invalid cached attention dimensions");
    }

    const std::size_t head_size = channels / num_heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));
    std::fill(output.begin(), output.end(), 0.0F);

    for (std::size_t head = 0; head < num_heads; ++head) {
        const auto head_offset = head * head_size;
        float maximum = std::numeric_limits<float>::lowest();

        for (std::size_t source = 0; source < visible_length; ++source) {
            const auto cache_base = source * channels + head_offset;
            float score = 0.0F;
            for (std::size_t index = 0; index < head_size; ++index) {
                score += query[head_offset + index] * key_cache[cache_base + index];
            }
            score *= scale;
            score_scratch[source] = score;
            maximum = std::max(maximum, score);
        }

        float exponential_sum = 0.0F;
        for (std::size_t source = 0; source < visible_length; ++source) {
            const float value = std::exp(score_scratch[source] - maximum);
            score_scratch[source] = value;
            exponential_sum += value;
        }
        const float inverse_sum = exponential_sum == 0.0F ? 0.0F : 1.0F / exponential_sum;

        for (std::size_t source = 0; source < visible_length; ++source) {
            const auto cache_base = source * channels + head_offset;
            const float probability = score_scratch[source] * inverse_sum;
            for (std::size_t index = 0; index < head_size; ++index) {
                output[head_offset + index] +=
                    probability * value_cache[cache_base + index];
            }
        }
    }
}

void gelu_in_place(std::span<float> values) {
    const float scaling = std::sqrt(static_cast<float>(2.0 / std::numbers::pi));
    for (float& value : values) {
        const float cube_term = 0.044715F * value * value * value;
        value = 0.5F * value * (1.0F + std::tanh(scaling * (value + cube_term)));
    }
}

void add_in_place(std::span<float> destination, std::span<const float> source) {
    require_size(source, destination.size(), "add source");
    for (std::size_t index = 0; index < destination.size(); ++index) {
        destination[index] += source[index];
    }
}

}  // namespace gpt2::kernels
