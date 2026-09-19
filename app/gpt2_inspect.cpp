#include "gpt2/model.hpp"

#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " --checkpoint PATH [--layout]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path checkpoint;
        bool show_layout = false;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--checkpoint" && index + 1 < argc) {
                checkpoint = argv[++index];
            } else if (argument == "--layout") {
                show_layout = true;
            } else if (argument == "--help") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::invalid_argument("unknown or incomplete argument: " +
                                            std::string(argument));
            }
        }
        if (checkpoint.empty()) {
            print_usage(argv[0]);
            return 2;
        }

        const auto weights = gpt2::ModelWeights::load_checkpoint(checkpoint);
        const auto& config = weights.config();
        std::cout << "checkpoint_version: " << weights.checkpoint_version() << '\n'
                  << "max_seq_len: " << config.max_sequence_length << '\n'
                  << "vocab_size: " << config.vocabulary_size << '\n'
                  << "padded_vocab_size: " << config.padded_vocabulary_size << '\n'
                  << "num_layers: " << config.num_layers << '\n'
                  << "num_heads: " << config.num_heads << '\n'
                  << "channels: " << config.channels << '\n'
                  << "num_parameters: " << weights.layout().total_count() << '\n';

        if (show_layout) {
            std::cout << "\nparameter_layout:\n";
            for (const auto& tensor : weights.layout().tensors()) {
                std::cout << std::left << std::setw(11) << tensor.name
                          << " offset=" << tensor.offset
                          << " count=" << tensor.count
                          << " shape=[";
                for (std::size_t dimension = 0; dimension < tensor.rank; ++dimension) {
                    if (dimension != 0) {
                        std::cout << ',';
                    }
                    std::cout << tensor.shape[dimension];
                }
                std::cout << "]\n";
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gpt2_inspect: " << error.what() << '\n';
        return 1;
    }
}
