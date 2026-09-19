#include "gpt2/linear_executor.hpp"

#include "gpt2/kernels.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace gpt2 {
namespace {

constexpr std::size_t min_weight_floats_per_thread = 65536;

}  // namespace

LinearExecutor::LinearExecutor(std::size_t total_threads)
    : total_threads_(total_threads) {
    if (total_threads_ == 0) {
        throw std::invalid_argument("linear executor requires at least one total thread");
    }
    workers_.reserve(total_threads_ - 1);
    try {
        for (std::size_t shard = 1; shard < total_threads_; ++shard) {
            workers_.emplace_back([this, shard] { worker_loop(shard); });
        }
    } catch (...) {
        stop_workers();
        throw;
    }
}

LinearExecutor::~LinearExecutor() {
    stop_workers();
}

std::size_t LinearExecutor::total_threads() const noexcept {
    return total_threads_;
}

void LinearExecutor::stop_workers() noexcept {
    {
        std::lock_guard lock(state_mutex_);
        stopping_ = true;
    }
    work_ready_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void LinearExecutor::run_shard(const Job& job, std::size_t shard) const {
    // Quotient/remainder partition avoids output_channels * shard overflow.
    const auto block = job.output_channels / total_threads_;
    const auto remainder = job.output_channels % total_threads_;
    const auto first = shard * block + std::min(shard, remainder);
    const auto count = block + (shard < remainder ? 1 : 0);
    if (count == 0) {
        return;
    }
    const auto weight_first = first * job.input_channels;
    const auto weight_count = count * job.input_channels;
    const auto bias = job.bias.empty() ? std::span<const float>{}
                                        : job.bias.subspan(first, count);
    kernels::linear(job.output.subspan(first, count), job.input,
                    job.weight.subspan(weight_first, weight_count), bias,
                    1, job.input_channels, count);
}

void LinearExecutor::worker_loop(std::size_t shard) noexcept {
    std::size_t observed_generation = 0;
    for (;;) {
        Job current;
        {
            std::unique_lock lock(state_mutex_);
            work_ready_.wait(lock, [&] {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_) {
                return;
            }
            observed_generation = generation_;
            current = job_;
        }

        std::exception_ptr error;
        try {
            run_shard(current, shard);
        } catch (...) {
            error = std::current_exception();
        }
        {
            std::lock_guard lock(state_mutex_);
            if (error && !worker_error_) {
                worker_error_ = error;
            }
            ++completed_;
        }
        work_done_.notify_one();
    }
}

void LinearExecutor::linear(std::span<float> output,
                            std::span<const float> input,
                            std::span<const float> weight,
                            std::span<const float> bias,
                            std::size_t input_channels,
                            std::size_t output_channels) {
    if (input_channels != 0 &&
        output_channels > std::numeric_limits<std::size_t>::max() / input_channels) {
        throw std::overflow_error("linear executor weight size overflow");
    }
    if (output.size() != output_channels || input.size() != input_channels ||
        weight.size() != input_channels * output_channels ||
        (!bias.empty() && bias.size() != output_channels)) {
        throw std::invalid_argument("linear executor tensor size mismatch");
    }

    // Serialize every submission, including the scalar fallback.
    std::lock_guard submission(submit_mutex_);
    if (total_threads_ == 1 || output_channels < total_threads_ ||
        weight.size() / total_threads_ < min_weight_floats_per_thread) {
        kernels::linear(output, input, weight, bias, 1,
                        input_channels, output_channels);
        return;
    }

    // No caller can replace job_ until every worker and the submitter finish.
    const Job current{output, input, weight, bias, input_channels, output_channels};
    {
        std::lock_guard lock(state_mutex_);
        job_ = current;
        completed_ = 0;
        worker_error_ = nullptr;
        ++generation_;
    }
    work_ready_.notify_all();

    std::exception_ptr caller_error;
    try {
        run_shard(current, 0);
    } catch (...) {
        caller_error = std::current_exception();
    }

    std::exception_ptr worker_error;
    {
        std::unique_lock lock(state_mutex_);
        work_done_.wait(lock, [&] { return completed_ == workers_.size(); });
        worker_error = worker_error_;
        job_ = {};
    }
    if (caller_error) {
        std::rethrow_exception(caller_error);
    }
    if (worker_error) {
        std::rethrow_exception(worker_error);
    }
}

}  // namespace gpt2
