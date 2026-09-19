#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace gpt2 {

// A synchronous, single-row FP32 linear executor. The submitting thread is
// included in total_threads. Input, weights, bias, and output must stay alive
// until linear() returns; simultaneous submissions are serialized internally.
class LinearExecutor {
public:
    explicit LinearExecutor(std::size_t total_threads);
    ~LinearExecutor();

    LinearExecutor(const LinearExecutor&) = delete;
    LinearExecutor& operator=(const LinearExecutor&) = delete;
    LinearExecutor(LinearExecutor&&) = delete;
    LinearExecutor& operator=(LinearExecutor&&) = delete;

    [[nodiscard]] std::size_t total_threads() const noexcept;

    void linear(std::span<float> output,
                std::span<const float> input,
                std::span<const float> weight,
                std::span<const float> bias,
                std::size_t input_channels,
                std::size_t output_channels);

private:
    struct Job {
        std::span<float> output;
        std::span<const float> input;
        std::span<const float> weight;
        std::span<const float> bias;
        std::size_t input_channels{};
        std::size_t output_channels{};
    };

    void worker_loop(std::size_t shard) noexcept;
    void run_shard(const Job& job, std::size_t shard) const;
    void stop_workers() noexcept;

    std::size_t total_threads_{};
    std::mutex submit_mutex_;
    std::mutex state_mutex_;
    std::condition_variable work_ready_;
    std::condition_variable work_done_;
    std::vector<std::thread> workers_;
    Job job_{};
    std::exception_ptr worker_error_;
    std::size_t generation_{};
    std::size_t completed_{};
    bool stopping_{};
};

}  // namespace gpt2
