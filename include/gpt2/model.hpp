#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace gpt2 {

inline constexpr std::int32_t checkpoint_magic = 20240326;
inline constexpr std::size_t checkpoint_header_ints = 256;

struct ModelConfig {
    std::size_t max_sequence_length{};
    std::size_t vocabulary_size{};
    std::size_t padded_vocabulary_size{};
    std::size_t num_layers{};
    std::size_t num_heads{};
    std::size_t channels{};

    void validate() const;
};

struct TensorSpec {
    std::string_view name;
    std::size_t offset{};
    std::size_t count{};
    std::array<std::size_t, 4> shape{};
    std::size_t rank{};
};

class ParameterLayout {
public:
    static constexpr std::size_t tensor_count = 16;

    explicit ParameterLayout(const ModelConfig& config);

    [[nodiscard]] const std::array<TensorSpec, tensor_count>& tensors() const noexcept;
    [[nodiscard]] const TensorSpec& at(std::string_view name) const;
    [[nodiscard]] std::size_t total_count() const noexcept;

private:
    std::array<TensorSpec, tensor_count> tensors_{};
    std::size_t total_count_{};
};

class TensorView {
public:
    TensorView(std::span<const float> values,
               std::array<std::size_t, 4> shape,
               std::size_t rank);

    [[nodiscard]] std::span<const float> values() const noexcept;
    [[nodiscard]] std::span<const std::size_t> shape() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const float& operator[](std::size_t index) const;

private:
    std::span<const float> values_;
    std::array<std::size_t, 4> shape_{};
    std::size_t rank_{};
};

class ModelWeights {
public:
    ModelWeights(const ModelWeights&) = delete;
    ModelWeights& operator=(const ModelWeights&) = delete;
    ModelWeights(ModelWeights&&) noexcept = default;
    ModelWeights& operator=(ModelWeights&&) noexcept = default;

    [[nodiscard]] static ModelWeights load_checkpoint(const std::filesystem::path& path);

    [[nodiscard]] const ModelConfig& config() const noexcept;
    [[nodiscard]] int checkpoint_version() const noexcept;
    [[nodiscard]] const ParameterLayout& layout() const noexcept;
    [[nodiscard]] TensorView tensor(std::string_view name) const;

private:
    ModelWeights(ModelConfig config,
                 int checkpoint_version,
                 ParameterLayout layout,
                 std::vector<float> storage);

    ModelConfig config_;
    int checkpoint_version_{};
    ParameterLayout layout_;
    std::vector<float> storage_;
};

class InferenceWorkspace {
public:
    void resize(const ModelConfig& config, std::size_t sequence_length);
    [[nodiscard]] std::size_t sequence_length() const noexcept;

private:
    friend class GPT2Model;

    std::size_t sequence_length_{};
    std::vector<float> residual_;
    std::vector<float> normalized_;
    std::vector<float> qkv_;
    std::vector<float> attention_output_;
    std::vector<float> projected_;
    std::vector<float> mlp_hidden_;
    std::vector<float> attention_scores_;
    std::vector<float> logits_;
};

class GPT2Model {
public:
    explicit GPT2Model(ModelWeights weights);

    GPT2Model(const GPT2Model&) = delete;
    GPT2Model& operator=(const GPT2Model&) = delete;
    GPT2Model(GPT2Model&&) noexcept = default;
    GPT2Model& operator=(GPT2Model&&) noexcept = default;

    [[nodiscard]] const ModelWeights& weights() const noexcept;

    // The returned view aliases workspace and remains valid until its next resize.
    [[nodiscard]] std::span<const float> forward(std::span<const std::int32_t> tokens,
                                                 InferenceWorkspace& workspace) const;

private:
    ModelWeights weights_;
};

}  // namespace gpt2
