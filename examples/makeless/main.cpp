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
        if (ch == '\n') continue;
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

    const size_t n_batch  = 1000;
    const size_t n_block  = 3;
    const size_t n_vocab  = 2271;
    const size_t n_embed  = 2;
    const size_t n_hidden = 10000;
    const size_t n_thread = 32;

    auto names = makeless_data_load(argv[1]);
    GGML_ASSERT(!names.empty());

    auto encoder = encoder_load(argv[2]);
    GGML_ASSERT(encoder);
    GGML_ASSERT(encoder->stoi.size() == n_vocab);
    GGML_ASSERT(encoder->itos.size() == n_vocab);

    std::vector<int32_t> tokens;
    std::transform(names.begin(), names.end(), std::back_inserter(tokens), [&](char32_t x){ return encoder->stoi[x]; });

    const size_t n_data = tokens.size() - n_block;

    int32_t * data_input = new int32_t [n_block * n_data];
    float   * data_label = new float   [n_vocab * n_data];

    memset(data_label, 0, n_vocab * n_data * sizeof(*data_label));

    // for (size_t i = 0; i + n_block < tokens.size(); ++i) {
    for (size_t i = 0; i < n_batch; ++i) {
        int32_t * ptr_input = data_input + i * n_block;
        memcpy(ptr_input, tokens.data() + i, n_block * sizeof(*ptr_input));

        int32_t   idx_label = tokens[i + n_block];
        float   * ptr_label = data_label + i * n_vocab;

        ptr_label[idx_label] = 1.0f;
    }

    // printf("%d, %d, %d\n", tokens.size(), data_input.size(), data_label.size());

    struct ggml_context * ctx;
    {
        struct ggml_init_params params {
            /*.mem_size   =*/ 1 << 30,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        ctx = ggml_init(params);
    }

    // ggml_tensor * data      = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_block, n_batch);
    // memcpy(data->data,  data_input, ggml_nbytes(data));

    ggml_tensor * label     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_batch);
    memcpy(label->data, data_label, ggml_nbytes(label));

    ggml_tensor * input     = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_block, n_batch);
    memcpy(input->data, data_input, ggml_nbytes(input));

    ggml_tensor * index     = ggml_view_1d(ctx, input, n_block * n_batch, 0);

    ggml_tensor * C         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embed, n_vocab);
    ggml_tensor * emb       = ggml_get_rows(ctx, C, index);
    ggml_tensor * emb_view  = ggml_view_2d(ctx, emb, n_embed * n_block, n_batch, emb->nb[1], 0);

    ggml_tensor * w1        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embed * n_block, n_hidden);
    ggml_tensor * b1        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_hidden);

    ggml_tensor * hpreact   = ggml_add(ctx, ggml_mul_mat(ctx, w1, emb_view), b1);
    ggml_tensor * h         = ggml_relu(ctx, hpreact);

    ggml_tensor * w2        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hidden, n_vocab);
    ggml_tensor * b2        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab);

    ggml_set_param(ctx, C);
    ggml_set_param(ctx, w2);
    ggml_set_param(ctx, b2);
    ggml_set_param(ctx, w1);
    ggml_set_param(ctx, b1);

    tensor_init_random(C);
    tensor_init_random(w2);
    tensor_init_random(b2);
    tensor_init_random(w1);
    tensor_init_random(b1);

    ggml_tensor * logits = ggml_add(ctx, ggml_mul_mat(ctx, w2, h), b2);

    // forward
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);

    ggml_build_forward_expand(gf, logits);

    // forward + backward
    ggml_cgraph * gf_grad = ggml_graph_dup(ctx, gf);

    ggml_tensor * loss = ggml_cross_entropy_loss(ctx, logits, label);
    ggml_set_loss(loss);

    ggml_build_forward_expand(gf_grad, loss);
    ggml_build_backward_expand(ctx, ctx, gf_grad, false);

    // forward + backward + optimize
    ggml_cgraph * gf_opt = ggml_graph_dup(ctx, gf_grad);

    const float learning_rate = 0.1f;

    for (int i = ggml_graph_n_nodes(gf)-1; i >= 0; --i) {
        ggml_tensor * node = ggml_graph_node(gf_opt, i);
        ggml_tensor * grad = ggml_graph_get_grad(gf_opt, node);

        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            ggml_tensor * learned = ggml_sub_inplace(ctx, node, ggml_scale_inplace(ctx, grad, learning_rate));
            ggml_build_forward_expand(gf_opt, learned);
        }
    }

    // train
    printf("training start\n");

    for (int i = 0; i < 10; ++i) {
        ggml_graph_reset(gf_opt);
        ggml_graph_compute_with_ctx(ctx, gf_opt, n_thread);

        printf("epoch %d, loss %f\n", i, ggml_get_f32_1d(loss, 0));
    }

    delete data_input;
    delete data_label;

    return 0;
}

