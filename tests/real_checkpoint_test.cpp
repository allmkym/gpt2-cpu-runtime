#include "gpt2/model.hpp"
#include "gpt2/linear_executor.hpp"
#include "gpt2/session.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

[[nodiscard]] std::size_t argmax(std::span<const float> logits) {
    return static_cast<std::size_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
}

[[nodiscard]] bool compare_step(const gpt2::GPT2Model& model,
                                gpt2::InferenceWorkspace& workspace,
                                std::span<const std::int32_t> prefix,
                                std::span<const float> cached,
                                std::string_view label) {
    const std::vector<float> saved(cached.begin(), cached.end());
    const auto full = model.forward(prefix, workspace);
    const auto& config = model.weights().config();
    const auto expected = full.last(config.padded_vocabulary_size)
                              .first(config.vocabulary_size);
    if (saved.size() != expected.size()) {
        std::cerr << label << " length mismatch: saved=" << saved.size()
                  << " expected=" << expected.size() << '\n';
        return false;
    }
    if (std::memcmp(saved.data(), expected.data(), expected.size_bytes()) != 0) {
        float maximum = 0.0F;
        std::size_t maximum_index = 0;
        for (std::size_t index = 0; index < saved.size(); ++index) {
            const float difference = std::abs(saved[index] - expected[index]);
            if (difference > maximum) {
                maximum = difference;
                maximum_index = index;
            }
        }
        std::cerr << label << " differs from full-prefix oracle: max_abs="
                  << maximum << " index=" << maximum_index << '\n';
        return false;
    }
    if (!std::all_of(saved.begin(), saved.end(),
                     [](float value) { return std::isfinite(value); })) {
        std::cerr << label << " contains a non-finite logit\n";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "expected checkpoint path\n";
            return 2;
        }
        gpt2::GPT2Model model(
            gpt2::ModelWeights::load_checkpoint(std::filesystem::path(argv[1])));
        const auto& config = model.weights().config();
        if (model.weights().checkpoint_version() != 1 ||
            model.weights().layout().total_count() != 124439808 ||
            config.vocabulary_size != 50257 || config.padded_vocabulary_size != 50257) {
            std::cerr << "real checkpoint metadata invariant failed\n";
            return 1;
        }

        gpt2::InferenceWorkspace workspace;
        gpt2::InferenceSession session(model, 5);

        const std::vector<std::int32_t> singleton{15496};
        const auto singleton_logits = session.prefill(singleton);
        if (!compare_step(model, workspace, singleton, singleton_logits,
                          "single-token prefill") ||
            argmax(singleton_logits) != 11) {
            std::cerr << "single-token real-checkpoint regression failed\n";
            return 1;
        }

        session.reset();
        std::vector<std::int32_t> prefix{15496, 11, 616};
        const auto prompt_logits = session.prefill(prefix);
        if (!compare_step(model, workspace, prefix, prompt_logits,
                          "three-token prefill") ||
            argmax(prompt_logits) != 1438) {
            std::cerr << "known prompt greedy token regression failed\n";
            return 1;
        }

        prefix.push_back(1438);
        const auto first_decode = session.decode(1438);
        if (!compare_step(model, workspace, prefix, first_decode,
                          "first cached decode") ||
            argmax(first_decode) != 318) {
            std::cerr << "known second greedy token regression failed\n";
            return 1;
        }

        prefix.push_back(318);
        const auto second_decode = session.decode(318);
        if (!compare_step(model, workspace, prefix, second_decode,
                          "second cached decode")) {
            return 1;
        }

        gpt2::LinearExecutor executor(4);
        gpt2::InferenceSession parallel_session(model, 5);
        if (!compare_step(model, workspace, singleton,
                          parallel_session.prefill(singleton, &executor),
                          "parallel single-token prefill")) {
            return 1;
        }
        parallel_session.reset();
        const std::vector<std::int32_t> prompt{15496, 11, 616};
        if (!compare_step(model, workspace, prompt,
                          parallel_session.prefill(prompt, &executor),
                          "parallel three-token prefill")) {
            return 1;
        }
        const std::vector<std::int32_t> first_extended{15496, 11, 616, 1438};
        if (!compare_step(model, workspace, first_extended,
                          parallel_session.decode(1438, &executor),
                          "parallel first decode") ||
            !compare_step(model, workspace, prefix,
                          parallel_session.decode(318, &executor),
                          "parallel second decode")) {
            return 1;
        }

        std::cout << "real checkpoint full/cached parity passed: version="
                  << model.weights().checkpoint_version()
                  << " parameters=" << model.weights().layout().total_count()
                  << " checked_prefix_lengths=1,3,4,5"
                  << " greedy_tokens=11,1438,318"
                  << " scalar_and_4_thread_comparison=bit-identical\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "real checkpoint test: " << error.what() << '\n';
        return 1;
    }
}
