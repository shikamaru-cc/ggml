#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "utf8.h"

#include <fstream>
#include <vector>
#include <set>
#include <map>

// TODO: use uft8char_t

using char_utf8_t = std::string;

struct encoder_t {
    std::map<char_utf8_t, unsigned> stoi;
    std::vector<char_utf8_t> itos;
};

static encoder_t encoder_load(const std::string & fname) {
    encoder_t encoder;

    auto fin = std::ifstream(fname);
    if (!fin) {
        fprintf(stderr, "failed to open token index file %s\n", fname.c_str());
        return {};
    }

    encoder.stoi["."] = 0;
    encoder.itos.emplace_back(".");

    size_t index = 1;
    std::string token;
    while (std::getline(fin, token)) {
        encoder.stoi[token] = index++;
        encoder.itos.emplace_back(token);
    }

    return encoder;
}

std::vector<char_utf8_t> string_split_utf8(const std::string & str) {
    std::vector<char_utf8_t> result;

    const char * w_prev = str.c_str();
    const char * w_curr = str.c_str();
    while (*w_curr) {
        utf8::advance(w_curr, 1, str.c_str() + str.size());
        result.emplace_back(std::string(w_prev, w_curr-w_prev));
        w_prev = w_curr;
    }

    return result;
}

std::string makeless_data_load(const std::string & fname) {
    auto fin = std::ifstream(fname);
    if (!fin) {
        fprintf(stderr, "failed to open dataset file %s\n", fname.c_str());
        return {};
    }

    std::string name, names;
    while (std::getline(fin, name)) {
        name.append(".");
        names.append(std::move(name));
    }

    return names;
}

// Generate token index
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

int main(int argc, const char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s examples/makeless/chinese-names-corpus.txt examples/makeless/token-index.txt\n", argv[0]);
        exit(0);
    }

    auto encoder = encoder_load(argv[2]);

    auto names = makeless_data_load(argv[1]);
    GGML_ASSERT(!names.empty());

    // std::vector<unsigned>   ctt_token;
    // std::string             ctt_string;

    // for (size_t i = 1000; i < 1020; ++i) {
    //     const auto & name = names[i];
    //     printf("%s --> ", name.c_str());
    //     for (const auto & token : string_split_utf8(name)) {
    //         printf("<%u> ", encoder.stoi[token]);
    //     }
    //     printf("\n");
    // }
}

