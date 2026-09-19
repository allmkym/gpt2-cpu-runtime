/*
 * Thin validation harness for the independent upstream llm.c CPU reference.
 *
 * Build with the pinned upstream checkout on the include path, for example:
 *   gcc -O3 -std=c11 -ffp-contract=off -Wno-unknown-pragmas \
 *     -I /path/to/llm.c validation/llmc_reference_logits.c \
 *     -lm -o /tmp/llmc_reference_logits
 *
 * The included train_gpt2.c remains governed by its upstream MIT license.
 */

#define main llmc_training_main
#include "train_gpt2.c"
#undef main

#include <stdint.h>

static int load_version1_checkpoint(GPT2* model, const char* path) {
    memset(model, 0, sizeof(*model));
    FILE* checkpoint = fopen(path, "rb");
    if (checkpoint == NULL) {
        perror("open checkpoint");
        return 1;
    }

    int32_t header[256];
    if (fread(header, sizeof(int32_t), 256, checkpoint) != 256) {
        fprintf(stderr, "truncated checkpoint header\n");
        fclose(checkpoint);
        return 1;
    }
    if (header[0] != 20240326 || header[1] != 1) {
        fprintf(stderr, "reference harness requires llm.c checkpoint version 1\n");
        fclose(checkpoint);
        return 1;
    }

    model->config.max_seq_len = header[2];
    model->config.vocab_size = header[3];
    model->config.padded_vocab_size = header[3];
    model->config.num_layers = header[4];
    model->config.num_heads = header[5];
    model->config.channels = header[6];
    fill_in_parameter_sizes(model->param_sizes, model->config);
    for (size_t index = 0; index < NUM_PARAMETER_TENSORS; ++index) {
        model->num_parameters += model->param_sizes[index];
    }
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    if (fread(model->params_memory, sizeof(float), model->num_parameters, checkpoint) !=
        model->num_parameters) {
        fprintf(stderr, "truncated checkpoint payload\n");
        fclose(checkpoint);
        free(model->params_memory);
        model->params_memory = NULL;
        return 1;
    }
    fclose(checkpoint);
    model->mean_loss = -1.0f;
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s CHECKPOINT OUTPUT_LOGITS\n", argv[0]);
        return 2;
    }

    GPT2 model;
    if (load_version1_checkpoint(&model, argv[1]) != 0) {
        return 1;
    }
    int tokens[] = {15496, 11, 616};
    const size_t sequence_length = sizeof(tokens) / sizeof(tokens[0]);
    gpt2_forward(&model, tokens, NULL, 1, sequence_length);

    FILE* output = fopen(argv[2], "wb");
    if (output == NULL) {
        perror("open logits output");
        gpt2_free(&model);
        return 1;
    }
    const float* last_logits =
        model.acts.logits + (sequence_length - 1) * model.config.padded_vocab_size;
    const size_t written = fwrite(last_logits, sizeof(float),
                                  model.config.vocab_size, output);
    fclose(output);
    if (written != (size_t)model.config.vocab_size) {
        fprintf(stderr, "failed to write reference logits\n");
        gpt2_free(&model);
        return 1;
    }

    size_t argmax = 0;
    for (size_t index = 1; index < (size_t)model.config.vocab_size; ++index) {
        if (last_logits[index] > last_logits[argmax]) {
            argmax = index;
        }
    }
    printf("llm.c reference: prefix=15496,11,616 logits=%d argmax=%zu\n",
           model.config.vocab_size, argmax);
    gpt2_free(&model);
    return 0;
}
