#pragma once

#include "gpt2/model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gpt2 {

class LinearExecutor;

class KVCache {
public:
    KVCache(std::size_t num_layers, std::size_t capacity, std::size_t channels);

    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;
    KVCache(KVCache&&) noexcept = default;
    KVCache& operator=(KVCache&&) noexcept = default;

    [[nodiscard]] std::size_t num_layers() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t channels() const noexcept;
    [[nodiscard]] std::size_t storage_float_count() const noexcept;

    void store(std::size_t layer,
               std::size_t position,
               std::span<const float> key,
               std::span<const float> value);
    [[nodiscard]] std::span<const float> key(std::size_t layer,
                                             std::size_t position) const;
    [[nodiscard]] std::span<const float> value(std::size_t layer,
                                               std::size_t position) const;
    [[nodiscard]] std::span<const float> layer_keys(std::size_t layer) const;
    [[nodiscard]] std::span<const float> layer_values(std::size_t layer) const;

private:
    [[nodiscard]] std::size_t position_offset(std::size_t layer,
                                               std::size_t position) const;
    [[nodiscard]] std::size_t layer_offset(std::size_t layer) const;

    std::size_t num_layers_{};
    std::size_t capacity_{};
    std::size_t channels_{};
    std::size_t layer_float_count_{};
    std::size_t storage_float_count_{};
    std::vector<float> keys_;
    std::vector<float> values_;
};

class InferenceSession {
public:
    // The borrowed model must outlive this nonmovable session and remain unmoved.
    InferenceSession(const GPT2Model& model, std::size_t capacity);
    InferenceSession(GPT2Model&& model, std::size_t capacity) = delete;
    InferenceSession(const GPT2Model&& model, std::size_t capacity) = delete;

    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;
    InferenceSession(InferenceSession&&) = delete;
    InferenceSession& operator=(InferenceSession&&) = delete;

    // Returned views alias reusable session storage; the next mutating call
    // invalidates their contents even when the address remains unchanged.
    [[nodiscard]] std::span<const float> prefill(
        std::span<const std::int32_t> prompt,
        LinearExecutor* executor = nullptr);
    [[nodiscard]] std::span<const float> decode(std::int32_t token,
                                                LinearExecutor* executor = nullptr);
    void reset() noexcept;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t cache_bytes() const noexcept;

private:
    void validate_token(std::int32_t token) const;
    void consume_token(std::int32_t token,
                       std::size_t position,
                       LinearExecutor* executor);

    const GPT2Model* model_;
    KVCache cache_;
    std::size_t size_{};
    std::vector<float> residual_;
    std::vector<float> normalized_;
    std::vector<float> qkv_;
    std::vector<float> attention_output_;
    std::vector<float> projected_;
    std::vector<float> mlp_hidden_;
    std::vector<float> attention_scores_;
    std::vector<float> logits_;
};

}  // namespace gpt2
