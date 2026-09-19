#include "gpt2/kernels.hpp"
#include "gpt2/model.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

static_assert(!std::is_copy_constructible_v<gpt2::ModelWeights>);
static_assert(!std::is_copy_assignable_v<gpt2::ModelWeights>);
static_assert(std::is_move_constructible_v<gpt2::ModelWeights>);
static_assert(std::is_move_assignable_v<gpt2::ModelWeights>);

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

template <typename Exception, typename Function>
void check_throws(Function&& function, std::string_view message) {
    try {
        function();
        check(false, message);
    } catch (const Exception&) {
    } catch (...) {
        check(false, std::string(message) + " (wrong exception type)");
    }
}

[[nodiscard]] gpt2::ModelConfig tiny_config() {
    return {
        .max_sequence_length = 4,
        .vocabulary_size = 5,
        .padded_vocabulary_size = 8,
        .num_layers = 2,
        .num_heads = 2,
        .channels = 4,
    };
}

class TemporaryFile {
public:
    explicit TemporaryFile(std::string_view suffix) {
        static std::size_t serial = 0;
        path_ = std::filesystem::temp_directory_path() /
                ("gpt2_runtime_test_" + std::to_string(++serial) + std::string(suffix));
    }

    ~TemporaryFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void write_checkpoint(const std::filesystem::path& path,
                      std::int32_t magic,
                      std::int32_t version,
                      const gpt2::ModelConfig& config,
                      bool write_payload,
                      std::size_t payload_shortfall = 0) {
    std::array<std::int32_t, gpt2::checkpoint_header_ints> header{};
    header[0] = magic;
    header[1] = version;
    header[2] = static_cast<std::int32_t>(config.max_sequence_length);
    header[3] = static_cast<std::int32_t>(config.vocabulary_size);
    header[4] = static_cast<std::int32_t>(config.num_layers);
    header[5] = static_cast<std::int32_t>(config.num_heads);
    header[6] = static_cast<std::int32_t>(config.channels);
    header[7] = static_cast<std::int32_t>(config.padded_vocabulary_size);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(header.data()),
                 static_cast<std::streamsize>(sizeof(header)));
    if (write_payload) {
        const gpt2::ParameterLayout layout(config);
        std::vector<float> payload(layout.total_count() - payload_shortfall, 0.0F);
        output.write(reinterpret_cast<const char*>(payload.data()),
                     static_cast<std::streamsize>(payload.size() * sizeof(float)));
    }
}

void test_parameter_layout() {
    const auto config = tiny_config();
    const gpt2::ParameterLayout layout(config);
    std::size_t expected_offset = 0;
    for (const auto& tensor : layout.tensors()) {
        std::size_t shape_product = 1;
        for (std::size_t dimension = 0; dimension < tensor.rank; ++dimension) {
            shape_product *= tensor.shape[dimension];
        }
        check(tensor.count == shape_product, "tensor count equals shape product");
        check(tensor.offset == expected_offset, "parameter tensors are contiguous");
        expected_offset += tensor.count;
    }
    check(expected_offset == layout.total_count(), "final offset equals total count");
    check(layout.at("wte").count == 32, "wte count");
    check(layout.at("qkvw").count == 96, "qkvw count");
    check(layout.at("fcprojw").count == 128, "fcprojw count");
}

void test_invalid_configuration_and_overflow() {
    auto config = tiny_config();
    config.num_heads = 3;
    check_throws<std::invalid_argument>([&] { gpt2::ParameterLayout unused(config); },
                                        "reject channels not divisible by heads");
    config = tiny_config();
    config.padded_vocabulary_size = config.vocabulary_size - 1;
    check_throws<std::invalid_argument>([&] { gpt2::ParameterLayout unused(config); },
                                        "reject padded vocabulary smaller than vocabulary");
    config = tiny_config();
    config.channels = std::numeric_limits<std::size_t>::max() / 2 + 1;
    config.num_heads = 1;
    check_throws<std::overflow_error>([&] { gpt2::ParameterLayout unused(config); },
                                      "reject layout arithmetic overflow");
}

void test_checkpoint_errors_and_versions() {
    const auto config = tiny_config();
    TemporaryFile truncated_header("_short.bin");
    {
        std::ofstream output(truncated_header.path(), std::ios::binary);
        const std::int32_t value = gpt2::checkpoint_magic;
        output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    check_throws<std::runtime_error>(
        [&] { (void)gpt2::ModelWeights::load_checkpoint(truncated_header.path()); },
        "reject truncated checkpoint header");

    TemporaryFile bad_magic("_magic.bin");
    write_checkpoint(bad_magic.path(), 123, 3, config, false);
    check_throws<std::runtime_error>(
        [&] { (void)gpt2::ModelWeights::load_checkpoint(bad_magic.path()); },
        "reject bad checkpoint magic");

    TemporaryFile bad_version("_version.bin");
    write_checkpoint(bad_version.path(), gpt2::checkpoint_magic, 2, config, false);
    check_throws<std::runtime_error>(
        [&] { (void)gpt2::ModelWeights::load_checkpoint(bad_version.path()); },
        "reject unsupported checkpoint version");

    TemporaryFile truncated_payload("_payload.bin");
    write_checkpoint(truncated_payload.path(), gpt2::checkpoint_magic, 3, config, true, 1);
    check_throws<std::runtime_error>(
        [&] { (void)gpt2::ModelWeights::load_checkpoint(truncated_payload.path()); },
        "reject truncated parameter payload");

    TemporaryFile version3("_v3.bin");
    write_checkpoint(version3.path(), gpt2::checkpoint_magic, 3, config, true);
    auto weights3 = gpt2::ModelWeights::load_checkpoint(version3.path());
    check(weights3.checkpoint_version() == 3, "load version 3 checkpoint");
    check(weights3.config().padded_vocabulary_size == 8, "version 3 padded vocabulary");

    auto version1_config = config;
    version1_config.padded_vocabulary_size = version1_config.vocabulary_size;
    TemporaryFile version1("_v1.bin");
    write_checkpoint(version1.path(), gpt2::checkpoint_magic, 1, version1_config, true);
    auto weights1 = gpt2::ModelWeights::load_checkpoint(version1.path());
    check(weights1.checkpoint_version() == 1, "load version 1 checkpoint");
    check(weights1.config().padded_vocabulary_size == 5, "version 1 old vocabulary semantics");
}

void test_kernels() {
    {
        const std::array<float, 2> input{1.0F, 2.0F};
        const std::array<float, 2> weight{1.0F, 1.0F};
        const std::array<float, 2> bias{0.0F, 0.0F};
        std::array<float, 2> output{};
        gpt2::kernels::layer_norm(output, input, weight, bias, 1, 2);
        check(std::abs(output[0] + 0.99998F) < 1.0e-4F, "layer norm first value");
        check(std::abs(output[1] - 0.99998F) < 1.0e-4F, "layer norm second value");
    }
    {
        const std::array<float, 2> input{1.0F, 2.0F};
        const std::array<float, 6> weight{1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
        const std::array<float, 3> bias{0.0F, 0.0F, 1.0F};
        std::array<float, 3> output{};
        gpt2::kernels::linear(output, input, weight, bias, 1, 2, 3);
        check(output == std::array<float, 3>{1.0F, 2.0F, 4.0F}, "linear known answer");
    }
    {
        std::array<float, 3> values{-1.0F, 0.0F, 1.0F};
        gpt2::kernels::gelu_in_place(values);
        check(std::abs(values[0] + 0.158808F) < 1.0e-5F, "GELU negative value");
        check(values[1] == 0.0F, "GELU zero");
        check(std::abs(values[2] - 0.841192F) < 1.0e-5F, "GELU positive value");
    }
    {
        // Per token: q[2], k[2], v[2]. Position zero must only attend to itself.
        const std::array<float, 12> qkv{
            1.0F, 0.0F, 1.0F, 0.0F, 2.0F, 4.0F,
            0.0F, 1.0F, 0.0F, 1.0F, 6.0F, 8.0F,
        };
        std::array<float, 4> output{};
        std::array<float, 2> scratch{};
        gpt2::kernels::causal_self_attention(output, qkv, scratch, 2, 2, 1);
        check(output[0] == 2.0F && output[1] == 4.0F,
              "causal attention first position");
        const float second_weight = std::exp(1.0F / std::sqrt(2.0F));
        const float expected0 = (2.0F + second_weight * 6.0F) / (1.0F + second_weight);
        const float expected1 = (4.0F + second_weight * 8.0F) / (1.0F + second_weight);
        check(std::abs(output[2] - expected0) < 1.0e-6F,
              "causal attention second position channel zero");
        check(std::abs(output[3] - expected1) < 1.0e-6F,
              "causal attention second position channel one");
    }
    {
        // A one-element softmax is exactly one even for a very negative score.
        const std::array<float, 3> qkv{1.0F, -20000.0F, 7.0F};
        std::array<float, 1> output{};
        std::array<float, 1> scratch{};
        gpt2::kernels::causal_self_attention(output, qkv, scratch, 1, 1, 1);
        check(output[0] == 7.0F,
              "causal attention normalizes a single score below negative ten thousand");
    }
}

}  // namespace

int main() {
    test_parameter_layout();
    test_invalid_configuration_and_overflow();
    test_checkpoint_errors_and_versions();
    test_kernels();
    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "all unit tests passed\n";
    return 0;
}
