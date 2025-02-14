#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <random>

static void tensor_randomize(ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<float> nd{0.0f, 1.0f};
    // std::normal_distribution<float> nd{0.0f, 1e-2f};

    const int64_t ne = ggml_nelements(t);
    std::vector<float> tmp(ne);

    for (int64_t i = 0; i < ne; ++i) {
        // tmp[i] = nd(gen);
        tmp[i] = 0.0f;
    }
    memcpy(t->data, tmp.data(), ggml_nbytes(t));
}

static void tensor_print_2d(ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);

    std::vector<float> out_data(ggml_nelements(t));
    memcpy(out_data.data(), t->data, ggml_nbytes(t));

    printf("=== Tensor ===\n");
    printf("%s\n", t->name);
    for (int j = 0; j < t->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }
        for (int i = 0; i < t->ne[0] /* cols */; i++) {
            printf(" %+.4f", out_data[j * t->ne[0] + i]);
        }
    }
    printf("\n");
    printf("========================================\n");
}

struct model_layer {
    struct ggml_tensor * w;
    struct ggml_tensor * b;

    model_layer(struct ggml_context * ctx, int n_input, int n_output) {
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_input, n_output);
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_output);
        tensor_randomize(w);
        tensor_randomize(b);
        ggml_set_param(ctx, w);
        ggml_set_param(ctx, b);
    }

    struct ggml_tensor * forward(struct ggml_context * ctx, struct ggml_tensor * x) {
        struct ggml_tensor * one  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_f32(one, 1.0f);

        // sigmoid
        struct ggml_tensor * y;
        y = ggml_add(ctx, ggml_mul_mat(ctx, w, x), b);

        struct ggml_tensor * ones = ggml_repeat(ctx, one, y);
        y = ggml_div(ctx, ones, ggml_add(ctx, ggml_exp(ctx, ggml_neg(ctx, y)), one));

        return y;
    }

    void learn(
        struct ggml_context * ctx          ,
        struct ggml_cgraph  * cgraph       ,
        struct ggml_cgraph  * cgraph_learn ,
        struct ggml_tensor  * learning_rate
    ) {
        struct ggml_tensor * grad_w  = ggml_view_tensor(ctx, ggml_graph_get_grad(cgraph, w));
        struct ggml_tensor * grad_b  = ggml_view_tensor(ctx, ggml_graph_get_grad(cgraph, b));

        struct ggml_tensor * learn_w = ggml_sub_inplace(ctx, w, ggml_mul(ctx, grad_w, learning_rate));
        struct ggml_tensor * learn_b = ggml_sub_inplace(ctx, b, ggml_mul(ctx, grad_b, learning_rate));

        ggml_build_forward_expand(cgraph_learn, learn_w);
        ggml_build_forward_expand(cgraph_learn, learn_b);
    }
};

struct model_layers {
    std::vector<struct model_layer> layers;

    model_layers(struct ggml_context * ctx, const std::vector<int>&& nins, int nout) {
        auto nins_cpy = nins;
        nins_cpy.emplace_back(nout);
        for (size_t i = 1; i < nins_cpy.size(); ++i) {
            layers.emplace_back(model_layer(ctx, nins_cpy[i-1], nins_cpy[i]));
        }
        for (size_t i = 0; i < layers.size(); ++i) {
            ggml_format_name(layers[i].w, "l%lu.w", i);
            ggml_format_name(layers[i].b, "l%lu.b", i);
        }
    }

    struct ggml_tensor * forward(struct ggml_context * ctx, struct ggml_tensor * x) {
        struct ggml_tensor * tensor = x;
        for (auto & layer : layers) {
            tensor = layer.forward(ctx, tensor);
        }
        return tensor;
    }

    void learn(
        struct ggml_context * ctx          ,
        struct ggml_cgraph  * cgraph       ,
        struct ggml_cgraph  * cgraph_learn ,
        struct ggml_tensor  * learning_rate
    ) {
        for (auto & layer : layers) {
            layer.learn(ctx, cgraph, cgraph_learn, learning_rate);
        }
    }
};

void print_model_layers(struct model_layers & mlp, struct ggml_cgraph * gf) {
    printf("=== Model ===\n");
    for (auto & layer : mlp.layers) {
        struct ggml_tensor * grad_w = ggml_graph_get_grad(gf, layer.w);
        struct ggml_tensor * grad_b = ggml_graph_get_grad(gf, layer.b);

        tensor_print_2d(layer.w);
        tensor_print_2d(grad_w);
        tensor_print_2d(layer.b);
        tensor_print_2d(grad_b);
    }
    printf("========================================\n");
}

int main(void) {
    float input[2*4] = {
        0, 0,
        0, 1,
        1, 0,
        1, 1,
    };
    float output[1*4] = {
        0,
        1,
        1,
        1,
    };

    struct ggml_context * ctx;
    {
        struct ggml_init_params params {
            /*.mem_size   =*/ 1 << 30,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        ctx = ggml_init(params);
    }

    auto model = model_layers(ctx, {2, 2, }, 1);

    struct ggml_tensor * x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 4);
    struct ggml_tensor * y_true = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 4);

    memcpy(x->data, input, ggml_nbytes(x));
    memcpy(y_true->data, output, ggml_nbytes(y_true));

    struct ggml_tensor * y_pred = model.forward(ctx, x);
    ggml_set_name(y_pred, "y_pred");

    struct ggml_tensor * loss;
    loss = ggml_sqr(ctx, ggml_sub(ctx, y_pred, y_true));
    loss = ggml_sum(ctx, loss);
    loss = ggml_scale(ctx, loss, 1.0f / ggml_nelements(y_pred));
    ggml_set_name(loss, "loss");
    ggml_set_loss(loss);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);

    ggml_build_forward_expand(gf, loss);
    ggml_build_backward_expand(ctx, ctx, gf, false);

    // ggml_graph_print(gf);
    ggml_graph_dump_dot(gf, 0, "gf.dot");

    struct ggml_tensor * learning_rate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_f32(learning_rate, 1.0f);

    struct ggml_cgraph * gf_learn = ggml_new_graph(ctx);
    model.learn(ctx, gf, gf_learn, learning_rate);

    ggml_graph_dump_dot(gf_learn, 0, "gf_learn.dot");

    for (int i = 0; i < 10000; ++i) {
        ggml_graph_reset(gf);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        // tensor_print_2d(loss);

        // print_model_layers(model, gf);

        ggml_graph_compute_with_ctx(ctx, gf_learn, 1);

        if (i % 100 == 0)
            printf("epoch %d, loss %f\n", i, ggml_get_f32_1d(loss, 0));
    }

    ggml_graph_reset(gf);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // print_model_layers(model, gf);
    // tensor_print_2d(y_pred);
}
