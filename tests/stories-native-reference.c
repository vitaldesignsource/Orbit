/* Regenerate the fixture with the original MIT llama2.c implementation:
 * cc -O2 -ffp-contract=off -I /path/to/llama2.c tests/stories-native-reference.c -lm -o /tmp/stories-reference
 * /tmp/stories-reference dist/machine/models/stories260K.bin dist/machine/models/tok512.bin > tests/fixtures/stories260k-reference.json
 * Reference upstream commit: 350e04fe35433e6d2941dce5a1f53308f87058eb
 */
#define TESTING
#include "run.c"

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    Transformer transformer;
    Tokenizer tokenizer;
    build_transformer(&transformer, argv[1]);
    build_tokenizer(&tokenizer, argv[2], transformer.config.vocab_size);
    const char *prompts[] = {"", "Once upon a time", "Lily's red ball.", "caf\xc3\xa9 \xf0\x9f\x8c\x99"};
    printf("{\"upstream\":\"karpathy/llama2.c@350e04fe35433e6d2941dce5a1f53308f87058eb\",\"promptTokens\":[");
    for (int p = 0; p < 4; p++) {
        int tokens[256], count;
        encode(&tokenizer, (char *)prompts[p], 1, 0, tokens, &count);
        printf("%s[", p ? "," : "");
        for (int i = 0; i < count; i++) printf("%s%d", i ? "," : "", tokens[i]);
        printf("]");
    }
    printf("],\"logits\":[");
    int next = 1;
    int tokens[200];
    for (int position = 0; position < 200; position++) {
        float *logits = forward(&transformer, next, position);
        if (position < 3) {
            printf("%s[", position ? "," : "");
            for (int i = 0; i < transformer.config.vocab_size; i++) printf("%s%.9g", i ? "," : "", logits[i]);
            printf("]");
        }
        next = sample_argmax(logits, transformer.config.vocab_size);
        tokens[position] = next;
    }
    printf("],\"greedyTokens\":[");
    for (int i = 0; i < 200; i++) printf("%s%d", i ? "," : "", tokens[i]);
    printf("]}\n");
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
