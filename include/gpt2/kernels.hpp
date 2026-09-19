#pragma once

#include <cstddef>
#include <span>

namespace gpt2::kernels {

void layer_norm(std::span<float> output,
                std::span<const float> input,
                std::span<const float> weight,
                std::span<const float> bias,
                std::size_t rows,
                std::size_t channels);

void linear(std::span<float> output,
            std::span<const float> input,
            std::span<const float> weight,
            std::span<const float> bias,
            std::size_t rows,
            std::size_t input_channels,
            std::size_t output_channels);

void causal_self_attention(std::span<float> output,
                           std::span<const float> qkv,
                           std::span<float> score_scratch,
                           std::size_t sequence_length,
                           std::size_t channels,
                           std::size_t num_heads);

void cached_self_attention(std::span<float> output,
                           std::span<const float> query,
                           std::span<const float> key_cache,
                           std::span<const float> value_cache,
                           std::span<float> score_scratch,
                           std::size_t visible_length,
                           std::size_t capacity,
                           std::size_t channels,
                           std::size_t num_heads);

void gelu_in_place(std::span<float> values);
void add_in_place(std::span<float> destination, std::span<const float> source);

}  // namespace gpt2::kernels
