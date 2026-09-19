#include "gpt2/kernels.hpp"
#include "gpt2/model.hpp"
#include "gpt2/session.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

static_assert(!std::is_copy_constructible_v<gpt2::InferenceSession>);
static_assert(!std::is_copy_assignable_v<gpt2::InferenceSession>);
static_assert(!std::is_move_constructible_v<gpt2::InferenceSession>);
static_assert(!std::is_move_assignable_v<gpt2::InferenceSession>);
static_assert(std::is_constructible_v<gpt2::InferenceSession,
                                      const gpt2::GPT2Model&,
                                      std::size_t>);
static_assert(!std::is_constructible_v<gpt2::InferenceSession,
                                       gpt2::GPT2Model&&,
                                       std::size_t>);
static_assert(!std::is_constructible_v<gpt2::InferenceSession,
                                       const gpt2::GPT2Model&&,
                                       std::size_t>);

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
    } catch (const std::exception& error) {
        check(false, std::string(message) + " (wrong exception: " + error.what() + ")");
    } catch (...) {
        check(false, std::string(message) + " (wrong non-standard exception)");
    }
}

[[nodiscard]] gpt2::ModelConfig version3_config() {
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
                ("gpt2_session_test_" + std::to_string(++serial) + std::string(suffix));
    }

    ~TemporaryFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

[[nodiscard]] float patterned_value(std::size_t index, float scale) {
    const auto centered = static_cast<int>((index * 37 + 11) % 29) - 14;
    return static_cast<float>(centered) * scale;
}

void write_deterministic_checkpoint(const std::filesystem::path& path,
                                    std::int32_t version,
                                    const gpt2::ModelConfig& config) {
    std::array<std::int32_t, gpt2::checkpoint_header_ints> header{};
    header[0] = gpt2::checkpoint_magic;
    header[1] = version;
    header[2] = static_cast<std::int32_t>(config.max_sequence_length);
    header[3] = static_cast<std::int32_t>(config.vocabulary_size);
    header[4] = static_cast<std::int32_t>(config.num_layers);
    header[5] = static_cast<std::int32_t>(config.num_heads);
    header[6] = static_cast<std::int32_t>(config.channels);
    header[7] = static_cast<std::int32_t>(config.padded_vocabulary_size);

    const gpt2::ParameterLayout layout(config);
    std::vector<float> payload(layout.total_count());
    for (const auto& tensor : layout.tensors()) {
        const bool layer_norm_weight =
            tensor.name == "ln1w" || tensor.name == "ln2w" || tensor.name == "lnfw";
        const bool bias = tensor.name.ends_with('b');
        const float scale = bias ? 0.003F : 0.0125F;
        for (std::size_t index = 0; index < tensor.count; ++index) {
            const float pattern = patterned_value(tensor.offset + index, scale);
            payload[tensor.offset + index] = layer_norm_weight ? 1.0F + pattern : pattern;
        }
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(header.data()),
                 static_cast<std::streamsize>(sizeof(header)));
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write deterministic test checkpoint");
    }
}

[[nodiscard]] bool bit_equal(std::span<const float> left,
                             std::span<const float> right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(), left.size_bytes()) == 0;
}

void compare_with_oracle(const gpt2::GPT2Model& model,
                         gpt2::InferenceWorkspace& workspace,
                         std::span<const std::int32_t> prefix,
                         std::span<const float> cached,
                         std::string_view context) {
    const std::vector<float> saved(cached.begin(), cached.end());
    const auto full = model.forward(prefix, workspace);
    const auto& config = model.weights().config();
    const auto expected = full.last(config.padded_vocabulary_size)
                              .first(config.vocabulary_size);
    if (saved.size() != expected.size()) {
        std::cerr << "cached/oracle length mismatch in " << context
                  << ": saved=" << saved.size()
                  << " expected=" << expected.size() << '\n';
        check(false, context);
        return;
    }
    if (!bit_equal(saved, expected)) {
        float maximum = 0.0F;
        std::size_t maximum_index = 0;
        for (std::size_t index = 0; index < saved.size(); ++index) {
            const float difference = std::abs(saved[index] - expected[index]);
            if (difference > maximum) {
                maximum = difference;
                maximum_index = index;
            }
        }
        std::cerr << "cached/oracle mismatch in " << context
                  << ": max_abs=" << maximum
                  << " index=" << maximum_index << '\n';
        check(false, context);
    }
}

void test_kv_cache_layout_and_bounds() {
    gpt2::KVCache cache(2, 2, 3);
    check(cache.num_layers() == 2, "KV cache layer count");
    check(cache.capacity() == 2, "KV cache capacity");
    check(cache.channels() == 3, "KV cache channels");
    check(cache.storage_float_count() == 24, "KV cache K plus V storage count");

    const std::array<float, 3> key00{1.0F, 2.0F, 3.0F};
    const std::array<float, 3> value00{4.0F, 5.0F, 6.0F};
    const std::array<float, 3> key11{11.0F, 12.0F, 13.0F};
    const std::array<float, 3> value11{14.0F, 15.0F, 16.0F};
    cache.store(0, 0, key00, value00);
    cache.store(1, 1, key11, value11);
    check(std::ranges::equal(cache.key(0, 0), key00),
          "KV cache key channels remain contiguous");
    check(std::ranges::equal(cache.value(0, 0), value00),
          "KV cache value channels remain contiguous");
    check(std::ranges::equal(cache.key(1, 1), key11),
          "KV cache layer and position are isolated");
    check(std::ranges::equal(cache.value(1, 1), value11),
          "KV cache value layer and position are isolated");
    check(cache.layer_keys(1).subspan(3, 3)[2] == 13.0F,
          "KV cache layer layout uses position-major channels");

    gpt2::KVCache capacity_one(1, 1, 3);
    capacity_one.store(0, 0, key00, value00);
    check(std::ranges::equal(capacity_one.key(0, 0), key00),
          "KV cache capacity one stores its only position");

    check_throws<std::out_of_range>([&] { (void)cache.key(2, 0); },
                                    "KV cache rejects an invalid layer");
    check_throws<std::out_of_range>([&] { (void)cache.value(0, 2); },
                                    "KV cache rejects an invalid position");
    check_throws<std::invalid_argument>(
        [&] { cache.store(0, 0, std::span<const float>(key00).first(2), value00); },
        "KV cache rejects the wrong channel count");
    check_throws<std::invalid_argument>([] { gpt2::KVCache unused(0, 1, 1); },
                                        "KV cache rejects zero dimensions");
    check_throws<std::overflow_error>(
        [] { gpt2::KVCache unused(1, std::numeric_limits<std::size_t>::max(), 2); },
        "KV cache checks capacity times channels");
    check_throws<std::overflow_error>(
        [] { gpt2::KVCache unused(std::numeric_limits<std::size_t>::max(), 2, 2); },
        "KV cache checks layer times layer storage");
    check_throws<std::overflow_error>(
        [] {
            gpt2::KVCache unused(
                1, std::numeric_limits<std::size_t>::max() / 4 + 1, 1);
        },
        "KV cache checks combined K/V allocation bytes");
}

void test_cached_attention_ignores_future_storage() {
    {
        // A one-element softmax is exactly one even for a very negative score.
        const std::array<float, 1> query{1.0F};
        const std::array<float, 1> keys{-20000.0F};
        const std::array<float, 1> values{7.0F};
        std::array<float, 1> output{};
        std::array<float, 1> scratch{};
        gpt2::kernels::cached_self_attention(output, query, keys, values, scratch,
                                              1, 1, 1, 1);
        check(output[0] == 7.0F,
              "cached attention normalizes a single score below negative ten thousand");
    }

    constexpr std::size_t capacity = 3;
    constexpr std::size_t channels = 4;
    const std::array<float, channels> query{0.5F, -0.25F, 0.75F, 0.125F};
    std::array<float, capacity * channels> keys{
        0.1F, 0.2F, 0.3F, 0.4F,
        0.4F, 0.3F, 0.2F, 0.1F,
        1000000.0F, -1000000.0F, 1000000.0F, -1000000.0F,
    };
    std::array<float, capacity * channels> values{
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        1000000.0F, 1000000.0F, 1000000.0F, 1000000.0F,
    };
    std::array<float, channels> first{};
    std::array<float, channels> second{};
    std::array<float, capacity> scratch{};
    gpt2::kernels::cached_self_attention(first, query, keys, values, scratch,
                                          2, capacity, channels, 2);
    std::fill(keys.begin() + 2 * channels, keys.end(), -2000000.0F);
    std::fill(values.begin() + 2 * channels, values.end(), -3000000.0F);
    gpt2::kernels::cached_self_attention(second, query, keys, values, scratch,
                                          2, capacity, channels, 2);
    check(bit_equal(first, second), "cached attention never reads future cache positions");
    check(std::ranges::all_of(second, [](float value) { return std::isfinite(value); }),
          "cached attention output is finite");
    check_throws<std::invalid_argument>(
        [&] {
            gpt2::kernels::cached_self_attention(second, query, keys, values, scratch,
                                                  0, capacity, channels, 2);
        },
        "cached attention rejects an empty visible prefix");
}

void test_session_errors_reset_and_isolation(const std::filesystem::path& checkpoint) {
    gpt2::GPT2Model model(gpt2::ModelWeights::load_checkpoint(checkpoint));
    check_throws<std::invalid_argument>([&] { gpt2::InferenceSession unused(model, 0); },
                                        "session rejects zero capacity before allocation");
    check_throws<std::invalid_argument>([&] { gpt2::InferenceSession unused(model, 5); },
                                        "session rejects capacity beyond context");

    gpt2::InferenceSession session(model, 4);
    check(session.size() == 0 && session.capacity() == 4,
          "new session is empty with fixed capacity");
    check(session.cache_bytes() == 2 * 2 * 4 * 4 * sizeof(float),
          "session reports exact KV cache bytes");
    check_throws<std::invalid_argument>([&] { (void)session.prefill({}); },
                                        "prefill rejects an empty prompt");
    check_throws<std::logic_error>([&] { (void)session.decode(1); },
                                   "decode rejects an empty session");

    const std::array<std::int32_t, 2> invalid_prompt{1, 5};
    check_throws<std::invalid_argument>([&] { (void)session.prefill(invalid_prompt); },
                                        "prefill validates every token before committing");
    check(session.size() == 0, "invalid prefill leaves session empty");

    const std::array<std::int32_t, 1> first_prompt{1};
    const auto first_logits = session.prefill(first_prompt);
    const float* logits_address = first_logits.data();
    check(first_logits.size() == 5, "session returns only real vocabulary logits");
    check(session.size() == 1, "prefill publishes prompt length");
    check_throws<std::logic_error>([&] { (void)session.prefill(first_prompt); },
                                   "prefill rejects a ready session");
    check_throws<std::invalid_argument>([&] { (void)session.decode(-1); },
                                        "decode rejects a negative token");
    check(session.size() == 1, "invalid decode preserves logical length");
    (void)session.decode(3);
    check(session.size() == 2, "decode appends at the current size");
    check(session.decode(2).data() == logits_address,
          "decode reuses the session logits allocation");
    (void)session.decode(4);
    check(session.size() == 4, "session can become exactly full");
    check_throws<std::length_error>([&] { (void)session.decode(0); },
                                    "decode rejects a full cache");
    check(session.size() == 4, "full-cache rejection preserves size");

    session.reset();
    check(session.size() == 0, "reset returns session to empty");
    const std::array<std::int32_t, 1> second_prompt{2};
    const auto reset_logits = session.prefill(second_prompt);
    gpt2::InferenceWorkspace workspace;
    compare_with_oracle(model, workspace, second_prompt, reset_logits,
                        "reset hides old cache positions");

    session.reset();
    const std::array<std::int32_t, 5> oversized_prompt{1, 2, 3, 4, 0};
    check_throws<std::length_error>([&] { (void)session.prefill(oversized_prompt); },
                                    "prefill rejects a prompt beyond capacity");
    check(session.size() == 0, "oversized prefill preserves empty state");
    const std::array<std::int32_t, 4> full_prompt{1, 3, 2, 4};
    compare_with_oracle(model, workspace, full_prompt, session.prefill(full_prompt),
                        "prefill can make the cache exactly full");
    check(session.size() == session.capacity(),
          "exact-capacity prefill publishes full size");
    check_throws<std::length_error>([&] { (void)session.decode(0); },
                                    "exact-capacity prefill blocks decode");

    gpt2::InferenceSession capacity_one(model, 1);
    (void)capacity_one.prefill(first_prompt);
    check(capacity_one.size() == 1, "capacity-one session becomes exactly full");
    check_throws<std::length_error>([&] { (void)capacity_one.decode(2); },
                                    "capacity-one session rejects decode");

    gpt2::InferenceSession left(model, 4);
    gpt2::InferenceSession right(model, 4);
    std::vector<std::int32_t> left_prefix{1};
    std::vector<std::int32_t> right_prefix{4};
    compare_with_oracle(model, workspace, left_prefix, left.prefill(left_prefix),
                        "left session initial prefix");
    compare_with_oracle(model, workspace, right_prefix, right.prefill(right_prefix),
                        "right session initial prefix");
    left_prefix.push_back(3);
    compare_with_oracle(model, workspace, left_prefix, left.decode(3),
                        "left session after interleaved decode");
    right_prefix.push_back(0);
    compare_with_oracle(model, workspace, right_prefix, right.decode(0),
                        "right session after interleaved decode");
}

void test_stepwise_numerical_regression(std::int32_t version,
                                        gpt2::ModelConfig config) {
    TemporaryFile checkpoint(version == 1 ? "_v1.bin" : "_v3.bin");
    write_deterministic_checkpoint(checkpoint.path(), version, config);
    gpt2::GPT2Model model(gpt2::ModelWeights::load_checkpoint(checkpoint.path()));
    gpt2::InferenceSession session(model, config.max_sequence_length);
    gpt2::InferenceWorkspace workspace;
    const std::array<std::int32_t, 4> tokens{1, 3, 2, 4};
    std::vector<std::int32_t> prefix{tokens[0]};

    compare_with_oracle(model, workspace, prefix,
                        session.prefill(std::span<const std::int32_t>(tokens).first(1)),
                        version == 1 ? "v1 step 1" : "v3 step 1");
    for (std::size_t index = 1; index < tokens.size(); ++index) {
        prefix.push_back(tokens[index]);
        compare_with_oracle(model, workspace, prefix, session.decode(tokens[index]),
                            version == 1 ? "v1 decode step" : "v3 decode step");
    }

    session.reset();
    const auto prompt = std::span<const std::int32_t>(tokens).first(3);
    compare_with_oracle(model, workspace, prompt, session.prefill(prompt),
                        version == 1 ? "v1 multi-token prefill" : "v3 multi-token prefill");
}

}  // namespace

int main() {
    test_kv_cache_layout_and_bounds();
    test_cached_attention_ignores_future_storage();

    TemporaryFile checkpoint("_state_v3.bin");
    const auto config = version3_config();
    write_deterministic_checkpoint(checkpoint.path(), 3, config);
    test_session_errors_reset_and_isolation(checkpoint.path());

    auto version1 = config;
    version1.padded_vocabulary_size = version1.vocabulary_size;
    test_stepwise_numerical_regression(1, version1);
    test_stepwise_numerical_regression(3, config);

    if (failures != 0) {
        std::cerr << failures << " session test assertion(s) failed\n";
        return 1;
    }
    std::cout << "all session and cached-inference tests passed\n";
    return 0;
}
