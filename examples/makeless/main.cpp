#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "utf8.h"

#include <algorithm>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <memory>

// Generate token index
//
// std::vector<char_utf8_t> string_split_utf8(const std::string & str) {
//     std::vector<char_utf8_t> result;
//     const char * w_prev = str.c_str();
//     const char * w_curr = str.c_str();
//     while (*w_curr) {
//         utf8::advance(w_curr, 1, str.c_str() + str.size());
//         result.emplace_back(std::string(w_prev, w_curr-w_prev));
//         w_prev = w_curr;
//     }

//     return result;
// }
//
// bool generate_token_index(const std::string & fname) {
//     auto fin = std::ifstream(fname);
//     if (!fin) {
//         fprintf(stderr, "failed to open dataset file %s\n", fname.c_str());
//         return false;
//     }

//     std::string name;
//     std::vector<std::string> names;
//     while (std::getline(fin, name)) {
//         names.emplace_back(std::move(name));
//     }

//     std::set<char_utf8_t> tokens;

//     for (const auto & name : names) {
//         auto more = string_split_utf8(name);
//         for (const auto & token : more) {
//             tokens.insert(token);
//         }
//     }

//     for (const auto & token : tokens) {
//         printf("%s\n", token.c_str());
//     }

//     return true;
// }


struct encoder_t {
    std::map<char32_t, unsigned> stoi;
    std::vector<char32_t> itos;
};


std::unique_ptr<encoder_t> encoder_load(const std::string & fname) {
    auto encoder = std::make_unique<encoder_t>();

    auto fin = std::ifstream(fname);
    if (!fin) {
        fprintf(stderr, "failed to open token index file %s\n", fname.c_str());
        return {};
    }

    std::istreambuf_iterator<char> it = fin.rdbuf();
    std::istreambuf_iterator<char> eos;

    size_t index = 0;

    while (it != eos) {
        const auto ch = utf8::next(it, eos);
        encoder->stoi[ch] = index++;
        encoder->itos.emplace_back(ch);
    }

    return encoder;
}


std::u32string makeless_data_load(const std::string & fname) {
    auto fin = std::ifstream(fname);
    if (!fin) {
        fprintf(stderr, "failed to open dataset file %s\n", fname.c_str());
        return {};
    }

    std::istreambuf_iterator<char> it(fin.rdbuf());
    std::istreambuf_iterator<char> eos;

    std::u32string s;
    while (it != eos) {
        s.push_back(utf8::next(it, eos));
    }

    return s;
}


#define MAKELESS_BATCH_SIZE 1024
#define MAKELESS_BLOCK_SIZE 3


int main(int argc, const char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s examples/makeless/chinese-names-corpus.txt examples/makeless/token-index.txt\n", argv[0]);
        exit(0);
    }

    // auto names = makeless_data_load(argv[1]);
    // GGML_ASSERT(!names.empty());

    // auto encoder = encoder_load(argv[2]);
    // GGML_ASSERT(encoder);

    // std::vector<unsigned> tokens;
    // std::transform(names.begin(), names.end(), std::back_inserter(tokens), [&](char32_t x){ return encoder->stoi[x]; });

    struct ggml_context * ctx;
    struct ggml_init_params params {
        /*.mem_size   =*/ 1 << 30,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    ctx = ggml_init(params);

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, MAKELESS_BLOCK_SIZE, MAKELESS_BATCH_SIZE);
    ggml_tensor * index = ggml_view_1d(ctx, input, MAKELESS_BATCH_SIZE * MAKELESS_BLOCK_SIZE, 0);
    ggml_tensor * C     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2271);
    ggml_tensor * emb   = ggml_get_rows(ctx, C, index);

    printf("shape of emb ne0=%ld ne1=%ld ne2=%ld ne3=%ld\n", emb->ne[0], emb->ne[1], emb->ne[2], emb->ne[3]);

    return 0;
}

