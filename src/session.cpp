#include "gpt2/session.hpp"

#include "gpt2/kernels.hpp"
#include "gpt2/linear_executor.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace gpt2 {
namespace {

[[nodiscard]] std::size_t checked_multiply(std::size_t left,
                                           std::size_t right,
                                           std::string_view context) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error("size overflow while computing " + std::string(context));
    }
    return left * right;
}

[[nodiscard]] std::span<const float> layer_slice(TensorView tensor,
                                                 std::size_t layer,
                                                 std::size_t per_layer) {
    const auto offset = checked_multiply(layer, per_layer, "session layer offset");
    return tensor.values().subspan(offset, per_layer);
}

[[nodiscard]] std::size_t validate_session_capacity(const GPT2Model& model,
                                                    std::size_t capacity) {
    if (capacity == 0 || capacity > model.weights().config().max_sequence_length) {
        throw std::invalid_argument("session capacity is outside the model context length");
    }
    return capacity;
}

}  // namespace

KVCache::KVCache(std::size_t num_layers,
                 std::size_t capacity,
                 std::size_t channels)
    : num_layers_(num_layers), capacity_(capacity), channels_(channels) {
    if (num_layers_ == 0 || capacity_ == 0 || channels_ == 0) {
        throw std::invalid_argument("KV cache dimensions must be positive");
    }
    layer_float_count_ = checked_multiply(capacity_, channels_, "KV cache layer size");
    const auto total = checked_multiply(num_layers_, layer_float_count_,
                                        "KV cache storage size");
    storage_float_count_ = checked_multiply(2, total, "combined K/V float count");
    (void)checked_multiply(storage_float_count_, sizeof(float),
                           "combined K/V storage bytes");
    keys_.resize(total);
    values_.resize(total);
}

std::size_t KVCache::num_layers() const noexcept {
    return num_layers_;
}

std::size_t KVCache::capacity() const noexcept {
    return capacity_;
}

std::size_t KVCache::channels() const noexcept {
    return channels_;
}

std::size_t KVCache::storage_float_count() const noexcept {
    return storage_float_count_;
}

std::size_t KVCache::layer_offset(std::size_t layer) const {
    if (layer >= num_layers_) {
        throw std::out_of_range("KV cache layer is out of range");
    }
    return layer * layer_float_count_;
}

std::size_t KVCache::position_offset(std::size_t layer,
                                     std::size_t position) const {
    if (position >= capacity_) {
        throw std::out_of_range("KV cache position is out of range");
    }
    return layer_offset(layer) + position * channels_;
}

void KVCache::store(std::size_t layer,
                    std::size_t position,
                    std::span<const float> key_values,
                    std::span<const float> value_values) {
    if (key_values.size() != channels_ || value_values.size() != channels_) {
        throw std::invalid_argument("KV cache write has the wrong channel count");
    }
    const auto offset = position_offset(layer, position);
    std::copy(key_values.begin(), key_values.end(), keys_.begin() + offset);
    std::copy(value_values.begin(), value_values.end(), values_.begin() + offset);
}

std::span<const float> KVCache::key(std::size_t layer,
                                    std::size_t position) const {
    return std::span<const float>(keys_).subspan(position_offset(layer, position), channels_);
}

std::span<const float> KVCache::value(std::size_t layer,
                                      std::size_t position) const {
    return std::span<const float>(values_).subspan(position_offset(layer, position), channels_);
}

std::span<const float> KVCache::layer_keys(std::size_t layer) const {
    return std::span<const float>(keys_).subspan(layer_offset(layer), layer_float_count_);
}

std::span<const float> KVCache::layer_values(std::size_t layer) const {
    return std::span<const float>(values_).subspan(layer_offset(layer), layer_float_count_);
}

InferenceSession::InferenceSession(const GPT2Model& model, std::size_t capacity)
    : model_(&model),
      cache_(model.weights().config().num_layers,
             validate_session_capacity(model, capacity),
             model.weights().config().channels) {
    const auto& config = model_->weights().config();

    const auto channels = config.channels;
    const auto c3 = checked_multiply(3, channels, "session qkv size");
    const auto c4 = checked_multiply(4, channels, "session MLP size");
    residual_.resize(channels);
    normalized_.resize(channels);
    qkv_.resize(c3);
    attention_output_.resize(channels);
    projected_.resize(channels);
    mlp_hidden_.resize(c4);
    attention_scores_.resize(capacity);
    logits_.resize(config.vocabulary_size);
}

void InferenceSession::validate_token(std::int32_t token) const {
    const auto vocabulary_size = model_->weights().config().vocabulary_size;
    if (token < 0 || static_cast<std::size_t>(token) >= vocabulary_size) {
        throw std::invalid_argument("input token ID is outside the real vocabulary");
    }
}

std::span<const float> InferenceSession::prefill(
    std::span<const std::int32_t> prompt,
    LinearExecutor* executor) {
    if (size_ != 0) {
        throw std::logic_error("prefill requires an empty session");
    }
    if (prompt.empty()) {
        throw std::invalid_argument("prefill prompt must not be empty");
    }
    if (prompt.size() > capacity()) {
        throw std::length_error("prefill prompt exceeds session capacity");
    }
    for (const auto token : prompt) {
        validate_token(token);
    }

    for (std::size_t position = 0; position < prompt.size(); ++position) {
        consume_token(prompt[position], position, executor);
    }
    size_ = prompt.size();
    return logits_;
}

std::span<const float> InferenceSession::decode(std::int32_t token,
                                                LinearExecutor* executor) {
    if (size_ == 0) {
        throw std::logic_error("decode requires a nonempty session");
    }
    validate_token(token);
    if (size_ == capacity()) {
        throw std::length_error("session KV cache is full");
    }

    const auto position = size_;
    consume_token(token, position, executor);
    size_ = position + 1;
    return logits_;
}

void InferenceSession::reset() noexcept {
    size_ = 0;
}

std::size_t InferenceSession::size() const noexcept {
    return size_;
}

std::size_t InferenceSession::capacity() const noexcept {
    return cache_.capacity();
}

std::size_t InferenceSession::cache_bytes() const noexcept {
    return cache_.storage_float_count() * sizeof(float);
}

void InferenceSession::consume_token(std::int32_t token,
                                     std::size_t position,
                                     LinearExecutor* executor) {
    const auto& weights = model_->weights();
    const auto& config = weights.config();
    const auto channels = config.channels;
    const auto c3 = checked_multiply(3, channels, "session 3 * channels");
    const auto c4 = checked_multiply(4, channels, "session 4 * channels");
    const auto linear = [&](std::span<float> output,
                            std::span<const float> input,
                            std::span<const float> weight,
                            std::span<const float> bias,
                            std::size_t input_count,
                            std::size_t output_count) {
        if (executor == nullptr) {
            kernels::linear(output, input, weight, bias, 1, input_count, output_count);
        } else {
            executor->linear(output, input, weight, bias, input_count, output_count);
        }
    };

    const auto wte = weights.tensor("wte").values();
    const auto wpe = weights.tensor("wpe").values();
    const auto token_base = static_cast<std::size_t>(token) * channels;
    const auto position_base = position * channels;
    for (std::size_t channel = 0; channel < channels; ++channel) {
        residual_[channel] = wte[token_base + channel] + wpe[position_base + channel];
    }

    for (std::size_t layer = 0; layer < config.num_layers; ++layer) {
        const auto ln1w = layer_slice(weights.tensor("ln1w"), layer, channels);
        const auto ln1b = layer_slice(weights.tensor("ln1b"), layer, channels);
        const auto qkvw = layer_slice(weights.tensor("qkvw"), layer, c3 * channels);
        const auto qkvb = layer_slice(weights.tensor("qkvb"), layer, c3);
        const auto attprojw = layer_slice(weights.tensor("attprojw"), layer,
                                         channels * channels);
        const auto attprojb = layer_slice(weights.tensor("attprojb"), layer, channels);
        const auto ln2w = layer_slice(weights.tensor("ln2w"), layer, channels);
        const auto ln2b = layer_slice(weights.tensor("ln2b"), layer, channels);
        const auto fcw = layer_slice(weights.tensor("fcw"), layer, c4 * channels);
        const auto fcb = layer_slice(weights.tensor("fcb"), layer, c4);
        const auto fcprojw = layer_slice(weights.tensor("fcprojw"), layer,
                                        channels * c4);
        const auto fcprojb = layer_slice(weights.tensor("fcprojb"), layer, channels);

        kernels::layer_norm(normalized_, residual_, ln1w, ln1b, 1, channels);
        linear(qkv_, normalized_, qkvw, qkvb, channels, c3);
        cache_.store(layer, position,
                     std::span<const float>(qkv_).subspan(channels, channels),
                     std::span<const float>(qkv_).subspan(2 * channels, channels));
        kernels::cached_self_attention(attention_output_,
                                       std::span<const float>(qkv_).first(channels),
                                       cache_.layer_keys(layer),
                                       cache_.layer_values(layer),
                                       attention_scores_, position + 1, capacity(),
                                       channels, config.num_heads);
        linear(projected_, attention_output_, attprojw, attprojb,
               channels, channels);
        kernels::add_in_place(residual_, projected_);

        kernels::layer_norm(normalized_, residual_, ln2w, ln2b, 1, channels);
        linear(mlp_hidden_, normalized_, fcw, fcb, channels, c4);
        kernels::gelu_in_place(mlp_hidden_);
        linear(projected_, mlp_hidden_, fcprojw, fcprojb,
               c4, channels);
        kernels::add_in_place(residual_, projected_);
    }

    kernels::layer_norm(normalized_, residual_,
                        weights.tensor("lnfw").values(),
                        weights.tensor("lnfb").values(),
                        1, channels);
    linear(logits_, normalized_,
           wte.first(config.vocabulary_size * channels), {},
           channels, config.vocabulary_size);
}

}  // namespace gpt2
