#include "gpt2/model.hpp"

#include "gpt2/kernels.hpp"

#include <algorithm>
#include <bit>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace gpt2 {
namespace {

[[nodiscard]] std::size_t checked_add(std::size_t left,
                                      std::size_t right,
                                      std::string_view context) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error("size overflow while computing " + std::string(context));
    }
    return left + right;
}

[[nodiscard]] std::size_t checked_multiply(std::size_t left,
                                           std::size_t right,
                                           std::string_view context) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error("size overflow while computing " + std::string(context));
    }
    return left * right;
}

[[nodiscard]] std::size_t product(std::initializer_list<std::size_t> dimensions,
                                  std::string_view context) {
    std::size_t result = 1;
    for (const auto dimension : dimensions) {
        result = checked_multiply(result, dimension, context);
    }
    return result;
}

[[nodiscard]] std::size_t checked_bytes(std::size_t float_count) {
    return checked_multiply(float_count, sizeof(float), "checkpoint payload bytes");
}

[[nodiscard]] std::span<const float> layer_slice(TensorView tensor,
                                                 std::size_t layer,
                                                 std::size_t per_layer) {
    const auto offset = checked_multiply(layer, per_layer, "layer tensor offset");
    return tensor.values().subspan(offset, per_layer);
}

}  // namespace

void ModelConfig::validate() const {
    if (max_sequence_length == 0 || vocabulary_size == 0 ||
        padded_vocabulary_size == 0 || num_layers == 0 ||
        num_heads == 0 || channels == 0) {
        throw std::invalid_argument("all GPT-2 configuration dimensions must be positive");
    }
    if (channels % num_heads != 0) {
        throw std::invalid_argument("channels must be divisible by num_heads");
    }
    if (padded_vocabulary_size < vocabulary_size) {
        throw std::invalid_argument("padded_vocabulary_size must be at least vocabulary_size");
    }
}

ParameterLayout::ParameterLayout(const ModelConfig& config) {
    config.validate();
    const auto c3 = checked_multiply(3, config.channels, "3 * channels");
    const auto c4 = checked_multiply(4, config.channels, "4 * channels");
    std::size_t index = 0;

    const auto add = [&](std::string_view name,
                         std::initializer_list<std::size_t> dimensions) {
        if (dimensions.size() > tensors_[index].shape.size()) {
            throw std::logic_error("parameter tensor rank exceeds fixed metadata capacity");
        }
        TensorSpec spec;
        spec.name = name;
        spec.offset = total_count_;
        spec.rank = dimensions.size();
        std::copy(dimensions.begin(), dimensions.end(), spec.shape.begin());
        spec.count = product(dimensions, name);
        total_count_ = checked_add(total_count_, spec.count, "total parameter count");
        tensors_[index++] = spec;
    };

    add("wte", {config.padded_vocabulary_size, config.channels});
    add("wpe", {config.max_sequence_length, config.channels});
    add("ln1w", {config.num_layers, config.channels});
    add("ln1b", {config.num_layers, config.channels});
    add("qkvw", {config.num_layers, c3, config.channels});
    add("qkvb", {config.num_layers, c3});
    add("attprojw", {config.num_layers, config.channels, config.channels});
    add("attprojb", {config.num_layers, config.channels});
    add("ln2w", {config.num_layers, config.channels});
    add("ln2b", {config.num_layers, config.channels});
    add("fcw", {config.num_layers, c4, config.channels});
    add("fcb", {config.num_layers, c4});
    add("fcprojw", {config.num_layers, config.channels, c4});
    add("fcprojb", {config.num_layers, config.channels});
    add("lnfw", {config.channels});
    add("lnfb", {config.channels});

    if (index != tensor_count) {
        throw std::logic_error("parameter layout tensor count mismatch");
    }
}

const std::array<TensorSpec, ParameterLayout::tensor_count>&
ParameterLayout::tensors() const noexcept {
    return tensors_;
}

const TensorSpec& ParameterLayout::at(std::string_view name) const {
    const auto found = std::find_if(tensors_.begin(), tensors_.end(),
                                    [name](const TensorSpec& spec) {
                                        return spec.name == name;
                                    });
    if (found == tensors_.end()) {
        throw std::out_of_range("unknown parameter tensor: " + std::string(name));
    }
    return *found;
}

std::size_t ParameterLayout::total_count() const noexcept {
    return total_count_;
}

TensorView::TensorView(std::span<const float> values,
                       std::array<std::size_t, 4> shape,
                       std::size_t rank)
    : values_(values), shape_(shape), rank_(rank) {
    if (rank_ == 0 || rank_ > shape_.size()) {
        throw std::invalid_argument("tensor view has invalid rank");
    }
    std::size_t expected = 1;
    for (std::size_t index = 0; index < rank_; ++index) {
        expected = checked_multiply(expected, shape_[index], "tensor view shape");
    }
    if (expected != values_.size()) {
        throw std::invalid_argument("tensor view shape does not match its storage range");
    }
}

std::span<const float> TensorView::values() const noexcept {
    return values_;
}

std::span<const std::size_t> TensorView::shape() const noexcept {
    return {shape_.data(), rank_};
}

std::size_t TensorView::size() const noexcept {
    return values_.size();
}

const float& TensorView::operator[](std::size_t index) const {
    if (index >= values_.size()) {
        throw std::out_of_range("tensor view index is out of range");
    }
    return values_[index];
}

ModelWeights::ModelWeights(ModelConfig config,
                           int checkpoint_version,
                           ParameterLayout layout,
                           std::vector<float> storage)
    : config_(config),
      checkpoint_version_(checkpoint_version),
      layout_(std::move(layout)),
      storage_(std::move(storage)) {
    if (storage_.size() != layout_.total_count()) {
        throw std::invalid_argument("weight storage does not match parameter layout");
    }
}

ModelWeights ModelWeights::load_checkpoint(const std::filesystem::path& path) {
    static_assert(std::endian::native == std::endian::little,
                  "the llm.c checkpoint loader currently requires a little-endian host");

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open checkpoint: " + path.string());
    }
    const auto end_position = input.tellg();
    if (end_position < 0) {
        throw std::runtime_error("cannot determine checkpoint size: " + path.string());
    }
    const auto file_size = static_cast<std::uintmax_t>(end_position);
    constexpr auto header_bytes = checkpoint_header_ints * sizeof(std::int32_t);
    if (file_size < header_bytes) {
        throw std::runtime_error("checkpoint has a truncated header");
    }

    input.seekg(0, std::ios::beg);
    std::array<std::int32_t, checkpoint_header_ints> header{};
    input.read(reinterpret_cast<char*>(header.data()),
               static_cast<std::streamsize>(header_bytes));
    if (!input) {
        throw std::runtime_error("failed to read checkpoint header");
    }
    if (header[0] != checkpoint_magic) {
        throw std::runtime_error("checkpoint has bad magic");
    }
    if (header[1] != 1 && header[1] != 3) {
        throw std::runtime_error("unsupported checkpoint version: " +
                                 std::to_string(header[1]));
    }

    const auto positive_dimension = [&](std::size_t index, const char* name) {
        if (header[index] <= 0) {
            throw std::runtime_error(std::string("invalid checkpoint dimension: ") + name);
        }
        return static_cast<std::size_t>(header[index]);
    };

    ModelConfig config;
    config.max_sequence_length = positive_dimension(2, "max_sequence_length");
    config.vocabulary_size = positive_dimension(3, "vocabulary_size");
    config.num_layers = positive_dimension(4, "num_layers");
    config.num_heads = positive_dimension(5, "num_heads");
    config.channels = positive_dimension(6, "channels");
    config.padded_vocabulary_size =
        header[1] == 1 ? config.vocabulary_size
                       : positive_dimension(7, "padded_vocabulary_size");
    config.validate();

    ParameterLayout layout(config);
    const auto payload_bytes = checked_bytes(layout.total_count());
    const auto expected_size = checked_add(header_bytes, payload_bytes, "checkpoint file size");
    if (file_size < expected_size) {
        throw std::runtime_error("checkpoint has a truncated parameter payload");
    }
    if (file_size > expected_size) {
        throw std::runtime_error("checkpoint contains unexpected trailing data");
    }
    if (payload_bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::overflow_error("checkpoint payload is too large for stream I/O");
    }

    std::vector<float> storage(layout.total_count());
    input.read(reinterpret_cast<char*>(storage.data()),
               static_cast<std::streamsize>(payload_bytes));
    if (!input) {
        throw std::runtime_error("failed to read checkpoint parameter payload");
    }

    return ModelWeights(config, header[1], std::move(layout), std::move(storage));
}

const ModelConfig& ModelWeights::config() const noexcept {
    return config_;
}

int ModelWeights::checkpoint_version() const noexcept {
    return checkpoint_version_;
}

const ParameterLayout& ModelWeights::layout() const noexcept {
    return layout_;
}

TensorView ModelWeights::tensor(std::string_view name) const {
    const auto& spec = layout_.at(name);
    return TensorView(std::span<const float>(storage_).subspan(spec.offset, spec.count),
                      spec.shape,
                      spec.rank);
}

void InferenceWorkspace::resize(const ModelConfig& config, std::size_t sequence_length) {
    config.validate();
    if (sequence_length == 0 || sequence_length > config.max_sequence_length) {
        throw std::invalid_argument("workspace sequence length is out of range");
    }
    const auto token_channels = checked_multiply(sequence_length, config.channels,
                                                 "workspace token channels");
    const auto qkv_count = checked_multiply(token_channels, 3, "workspace qkv");
    const auto hidden_count = checked_multiply(token_channels, 4, "workspace MLP hidden");
    const auto logits_count = checked_multiply(sequence_length,
                                               config.padded_vocabulary_size,
                                               "workspace logits");

    residual_.resize(token_channels);
    normalized_.resize(token_channels);
    qkv_.resize(qkv_count);
    attention_output_.resize(token_channels);
    projected_.resize(token_channels);
    mlp_hidden_.resize(hidden_count);
    attention_scores_.resize(sequence_length);
    logits_.resize(logits_count);
    sequence_length_ = sequence_length;
}

std::size_t InferenceWorkspace::sequence_length() const noexcept {
    return sequence_length_;
}

GPT2Model::GPT2Model(ModelWeights weights) : weights_(std::move(weights)) {}

const ModelWeights& GPT2Model::weights() const noexcept {
    return weights_;
}

std::span<const float> GPT2Model::forward(std::span<const std::int32_t> tokens,
                                         InferenceWorkspace& workspace) const {
    const auto& config = weights_.config();
    if (tokens.empty() || tokens.size() > config.max_sequence_length) {
        throw std::invalid_argument("token prefix length is out of range");
    }
    for (const auto token : tokens) {
        if (token < 0 || static_cast<std::size_t>(token) >= config.vocabulary_size) {
            throw std::invalid_argument("input token ID is outside the real vocabulary");
        }
    }

    workspace.resize(config, tokens.size());
    const auto rows = tokens.size();
    const auto channels = config.channels;
    const auto token_channels = rows * channels;

    const auto wte = weights_.tensor("wte").values();
    const auto wpe = weights_.tensor("wpe").values();
    for (std::size_t position = 0; position < rows; ++position) {
        const auto token_base = static_cast<std::size_t>(tokens[position]) * channels;
        const auto position_base = position * channels;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            workspace.residual_[position_base + channel] =
                wte[token_base + channel] + wpe[position_base + channel];
        }
    }

    const auto c3 = checked_multiply(3, channels, "forward 3 * channels");
    const auto c4 = checked_multiply(4, channels, "forward 4 * channels");
    for (std::size_t layer = 0; layer < config.num_layers; ++layer) {
        const auto ln1w = layer_slice(weights_.tensor("ln1w"), layer, channels);
        const auto ln1b = layer_slice(weights_.tensor("ln1b"), layer, channels);
        const auto qkvw = layer_slice(weights_.tensor("qkvw"), layer, c3 * channels);
        const auto qkvb = layer_slice(weights_.tensor("qkvb"), layer, c3);
        const auto attprojw = layer_slice(weights_.tensor("attprojw"), layer,
                                         channels * channels);
        const auto attprojb = layer_slice(weights_.tensor("attprojb"), layer, channels);
        const auto ln2w = layer_slice(weights_.tensor("ln2w"), layer, channels);
        const auto ln2b = layer_slice(weights_.tensor("ln2b"), layer, channels);
        const auto fcw = layer_slice(weights_.tensor("fcw"), layer, c4 * channels);
        const auto fcb = layer_slice(weights_.tensor("fcb"), layer, c4);
        const auto fcprojw = layer_slice(weights_.tensor("fcprojw"), layer,
                                        channels * c4);
        const auto fcprojb = layer_slice(weights_.tensor("fcprojb"), layer, channels);

        kernels::layer_norm(workspace.normalized_, workspace.residual_, ln1w, ln1b,
                            rows, channels);
        kernels::linear(workspace.qkv_, workspace.normalized_, qkvw, qkvb,
                        rows, channels, c3);
        kernels::causal_self_attention(workspace.attention_output_, workspace.qkv_,
                                       workspace.attention_scores_, rows, channels,
                                       config.num_heads);
        kernels::linear(workspace.projected_, workspace.attention_output_,
                        attprojw, attprojb, rows, channels, channels);
        kernels::add_in_place(workspace.residual_, workspace.projected_);

        kernels::layer_norm(workspace.normalized_, workspace.residual_, ln2w, ln2b,
                            rows, channels);
        kernels::linear(workspace.mlp_hidden_, workspace.normalized_, fcw, fcb,
                        rows, channels, c4);
        kernels::gelu_in_place(workspace.mlp_hidden_);
        kernels::linear(workspace.projected_, workspace.mlp_hidden_, fcprojw, fcprojb,
                        rows, c4, channels);
        kernels::add_in_place(workspace.residual_, workspace.projected_);
    }

    kernels::layer_norm(workspace.normalized_, workspace.residual_,
                        weights_.tensor("lnfw").values(),
                        weights_.tensor("lnfb").values(),
                        rows, channels);
    kernels::linear(workspace.logits_, workspace.normalized_, wte, {}, rows, channels,
                    config.padded_vocabulary_size);

    if (workspace.residual_.size() != token_channels) {
        throw std::logic_error("workspace invariant failed after forward pass");
    }
    return workspace.logits_;
}

}  // namespace gpt2
