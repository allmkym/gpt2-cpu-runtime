#include "gpt2/kernels.hpp"
#include "gpt2/linear_executor.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace {

std::atomic<int> failures{};

void check(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

[[nodiscard]] bool same_bits(std::span<const float> left,
                             std::span<const float> right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(), left.size_bytes()) == 0;
}

void check_shape(gpt2::LinearExecutor& executor,
                 std::size_t input_channels,
                 std::size_t output_channels,
                 bool with_bias) {
    std::vector<float> input(input_channels);
    std::vector<float> weights(input_channels * output_channels);
    std::vector<float> bias(with_bias ? output_channels : 0);
    std::vector<float> reference(output_channels);
    std::vector<float> actual(output_channels);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 19) - 9) * 0.025F;
    }
    for (std::size_t index = 0; index < weights.size(); ++index) {
        weights[index] = static_cast<float>(static_cast<int>(index % 37) - 18) * 0.004F;
    }
    for (std::size_t index = 0; index < bias.size(); ++index) {
        bias[index] = static_cast<float>(index % 11) * 0.003F;
    }

    gpt2::kernels::linear(reference, input, weights, bias,
                          1, input_channels, output_channels);
    for (int iteration = 0; iteration < 12; ++iteration) {
        std::fill(actual.begin(), actual.end(), 123.0F);
        executor.linear(actual, input, weights, bias,
                        input_channels, output_channels);
        check(same_bits(actual, reference), "parallel and scalar dot products match bitwise");
    }

    try {
        executor.linear(std::span<float>(actual).first(actual.size() - 1),
                        input, weights, bias, input_channels, output_channels);
        check(false, "invalid output size throws");
    } catch (const std::invalid_argument&) {
    }
    executor.linear(actual, input, weights, bias, input_channels, output_channels);
    check(same_bits(actual, reference), "executor recovers after rejected input");
}

void test_concurrent_submissions() {
    gpt2::LinearExecutor executor(4);
    std::exception_ptr first_error;
    std::exception_ptr second_error;
    auto task = [&](std::exception_ptr& error) {
        try {
            check_shape(executor, 137, 4097, true);
        } catch (...) {
            error = std::current_exception();
        }
    };
    std::thread first([&] { task(first_error); });
    std::thread second([&] { task(second_error); });
    first.join();
    second.join();
    check(!first_error && !second_error, "concurrent submissions complete safely");
}

void test_concurrent_scalar_fallback() {
    gpt2::LinearExecutor executor(4);
    const std::vector<float> first_input{1.0F, 2.0F, 3.0F};
    const std::vector<float> second_input{4.0F, 5.0F, 6.0F};
    const std::vector<float> weights{
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
        9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F};
    std::vector<float> first_expected(5);
    std::vector<float> second_expected(5);
    std::vector<float> shared_output(5);
    gpt2::kernels::linear(first_expected, first_input, weights, {}, 1, 3, 5);
    gpt2::kernels::linear(second_expected, second_input, weights, {}, 1, 3, 5);

    std::atomic<bool> start{};
    auto submit = [&](std::span<const float> input) {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (int iteration = 0; iteration < 32; ++iteration) {
            executor.linear(shared_output, input, weights, {}, 3, 5);
        }
    };
    std::thread first([&] { submit(first_input); });
    std::thread second([&] { submit(second_input); });
    start.store(true, std::memory_order_release);
    first.join();
    second.join();
    check(same_bits(shared_output, first_expected) ||
              same_bits(shared_output, second_expected),
          "concurrent scalar fallbacks serialize the complete output");
}

}  // namespace

int main() {
    try {
        gpt2::LinearExecutor invalid(0);
        check(false, "zero total threads rejected");
    } catch (const std::invalid_argument&) {
    }

    for (const std::size_t threads : {1U, 2U, 4U, 8U}) {
        gpt2::LinearExecutor executor(threads);
        check(executor.total_threads() == threads, "total thread count includes caller");
        check_shape(executor, 3, 5, true);          // small serial fallback, ragged output
        check_shape(executor, 137, 4097, true);    // parallel, odd input and output
        check_shape(executor, 137, 4097, false);   // no-bias path
    }
    test_concurrent_submissions();
    test_concurrent_scalar_fallback();
    for (int iteration = 0; iteration < 5; ++iteration) {
        gpt2::LinearExecutor temporary(4);
        check_shape(temporary, 3, 5, false);
    }
    if (failures != 0) {
        std::cerr << failures << " linear executor assertion(s) failed\n";
        return 1;
    }
    std::cout << "linear executor tests passed\n";
    return 0;
}
