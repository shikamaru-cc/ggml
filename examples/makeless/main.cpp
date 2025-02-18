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
#include <random>

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
    std::map<char32_t, int32_t> stoi;
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

    int32_t index = 0;

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


#define TENSOR_DEBUG_SHAPE(t) \
    do { \
        printf(#t ": ne0=%ld ne1=%ld ne2=%ld ne3=%ld\n", (t)->ne[0], (t)->ne[1], (t)->ne[2], (t)->ne[3]); \
    } while (0)


void tensor_init_random(ggml_tensor * t) {
    static std::random_device rd{};
    static std::mt19937 gen{rd()};
    static std::normal_distribution<float> nd{0.0f, 1e-2f};

    GGML_ASSERT(t->type == GGML_TYPE_F32);
    const int64_t ne = ggml_nelements(t);
    std::vector<float> tmp(ne);

    for (int64_t i = 0; i < ne; ++i) {
        tmp[i] = nd(gen);
    }
    memcpy(t->data, tmp.data(), ggml_nbytes(t));
}


int main(int argc, const char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s examples/makeless/chinese-names-corpus.txt examples/makeless/token-index.txt\n", argv[0]);
        exit(0);
    }

    auto names = makeless_data_load(argv[1]);
    GGML_ASSERT(!names.empty());

    auto encoder = encoder_load(argv[2]);
    GGML_ASSERT(encoder);

    std::vector<int32_t> tokens;
    std::transform(names.begin(), names.end(), std::back_inserter(tokens), [&](char32_t x){ return encoder->stoi[x]; });

    // struct ggml_context * ctx;
    // {
    //     struct ggml_init_params params {
    //         /*.mem_size   =*/ 1 << 30,
    //         /*.mem_buffer =*/ NULL,
    //         /*.no_alloc   =*/ false,
    //     };
    //     ctx = ggml_init(params);
    // }

    // size_t n_batch  = 1024;
    // size_t n_block  = 3;
    // size_t n_vocab  = 2271;
    // size_t n_embed  = 2;
    // size_t n_hidden = 10000;
    // size_t n_thread = 32;

    // ggml_tensor * data      = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_block, tokens.size() / n_block);

    // ggml_tensor * input     = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_block, n_batch);
    // ggml_tensor * index     = ggml_view_1d(ctx, input, n_block * n_batch, 0);

    // ggml_tensor * C         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embed, n_vocab);
    // ggml_tensor * emb       = ggml_get_rows(ctx, C, index);
    // ggml_tensor * emb_view  = ggml_view_2d(ctx, emb, n_embed * n_block, n_batch, emb->nb[1], 0);

    // ggml_tensor * w1        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embed * n_block, n_hidden);
    // ggml_tensor * b1        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_hidden);

    // ggml_tensor * hpreact   = ggml_add(ctx, ggml_mul_mat(ctx, w1, emb_view), b1);
    // ggml_tensor * h         = ggml_tanh(ctx, hpreact);

    // ggml_tensor * w2        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hidden, n_vocab);
    // ggml_tensor * b2        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab);

    // ggml_set_param(ctx, C);
    // ggml_set_param(ctx, w2);
    // ggml_set_param(ctx, b2);
    // ggml_set_param(ctx, w1);
    // ggml_set_param(ctx, b1);

    // tensor_init_random(C);
    // tensor_init_random(w2);
    // tensor_init_random(b2);
    // tensor_init_random(w1);
    // tensor_init_random(b1);

    // ggml_tensor * logits    = ggml_add(ctx, ggml_mul_mat(ctx, w2, hpreact), b2);

    // ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);

    // ggml_build_forward_expand(gf, logits);
    // ggml_build_backward_expand(ctx, ctx, gf, false);

    // ggml_graph_reset(gf);
    // ggml_graph_compute_with_ctx(ctx, gf, n_thread);

    // printf("epoch %d, loss %f\n", 0, ggml_get_f32_1d(loss, 0));

    return 0;
}

