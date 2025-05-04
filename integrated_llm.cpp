#include "integrated_llm.h"
#include "llama.h"
#include <string>
#include <vector>
#include <mutex>
#include <fstream>
#include <sstream>
#include <memory>
#include <algorithm>
#include <iostream>
#include <filesystem>

// Unique pointer alias for resource management
template<typename T, void(*Deleter)(T*)>
using resource_unique_ptr = std::unique_ptr<T, decltype(Deleter)>;

static resource_unique_ptr<llama_context, llama_free>* inference_context = nullptr;
static resource_unique_ptr<llama_model, llama_model_free>* loaded_model = nullptr;
static std::string last_model_response;
static std::mutex api_mutex;
static std::vector<std::string> knowledge_base;

extern "C" {

INTEGRATEDLLM_API int init(const char* model_path) {
    std::lock_guard<std::mutex> lock(api_mutex);
#ifdef _DEBUG
    std::cout << "[DEBUG] init called with model_path: " << model_path << std::endl;
#endif
    if (!model_path || !std::filesystem::exists(model_path)) {
#ifdef _DEBUG
        std::cout << "[DEBUG] Model file does not exist." << std::endl;
#endif
        return 10;
    }
    if (inference_context && inference_context->get()) {
#ifdef _DEBUG
        std::cout << "[DEBUG] Model already initialized." << std::endl;
#endif
        return 0;
    }
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
#ifdef _DEBUG
        std::cout << "[DEBUG] llama_model_load_from_file failed." << std::endl;
#endif
        return 1;
    }
    llama_context_params ctx_params = llama_context_default_params();
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
#ifdef _DEBUG
        std::cout << "[DEBUG] llama_init_from_model failed." << std::endl;
#endif
        llama_model_free(model);
        return 2;
    }
    loaded_model = new resource_unique_ptr<llama_model, llama_model_free>(model, llama_model_free);
    inference_context = new resource_unique_ptr<llama_context, llama_free>(ctx, llama_free);
#ifdef _DEBUG
    std::cout << "[DEBUG] Model initialized successfully." << std::endl;
#endif
    return 0;
}

INTEGRATEDLLM_API void add_knowledge(const char* doc) {
    std::lock_guard<std::mutex> lock(api_mutex);
    if (!doc) {
#ifdef _DEBUG
        std::cout << "[DEBUG] add_knowledge received null doc." << std::endl;
#endif
        return;
    }
#ifdef _DEBUG
    std::cout << "[DEBUG] Adding knowledge: " << doc << std::endl;
#endif
    knowledge_base.push_back(std::string(doc));
#ifdef _DEBUG
    std::cout << "[DEBUG] Knowledge addition done." << std::endl;
#endif
}

INTEGRATEDLLM_API void load_knowledge_file(const char* filepath) {
    std::lock_guard<std::mutex> lock(api_mutex);
    if (!filepath) {
#ifdef _DEBUG
        std::cout << "[DEBUG] load_knowledge_file received null filepath." << std::endl;
#endif
        return;
    }
#ifdef _DEBUG
    std::cout << "[DEBUG] Loading knowledge file: " << filepath << std::endl;
#endif
    std::ifstream infile(filepath);
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty()) {
            knowledge_base.push_back(line);
        }
    }
#ifdef _DEBUG
    std::cout << "[DEBUG] Knowledge file loading done." << std::endl;
#endif
}

static std::string retrieve_context(const std::string& question) {
    int max_score = 0;
    std::string best_match;
    for (const auto& doc : knowledge_base) {
        int score = 0;
        std::istringstream iss_q(question);
        std::string word;
        while (iss_q >> word) {
            if (doc.find(word) != std::string::npos) {
                score++;
            }
        }
        if (score > max_score) {
            max_score = score;
            best_match = doc;
        }
    }
    return best_match;
}

INTEGRATEDLLM_API char* query(const char* prompt) {
    std::lock_guard<std::mutex> lock(api_mutex);
#ifdef _DEBUG
    std::cout << "[DEBUG] Entered query()" << std::endl;
#endif
    if (!inference_context || !inference_context->get()) {
        last_model_response = "LLM not initialized.";
        char* result = (char*)malloc(last_model_response.size() + 1);
        if (result) {
            memcpy(result, last_model_response.c_str(), last_model_response.size() + 1);
        }
        return result;
    }
    std::string context = retrieve_context(prompt);
    std::string full_prompt = prompt;
    if (!context.empty()) {
        full_prompt += " " + context;
    }
    const llama_model* model = loaded_model ? loaded_model->get() : nullptr;
    const llama_vocab* vocab = model ? llama_model_get_vocab(model) : nullptr;
    if (!vocab) {
        last_model_response = "Vocab not available.";
        char* result = (char*)malloc(last_model_response.size() + 1);
        if (result) {
            memcpy(result, last_model_response.c_str(), last_model_response.size() + 1);
        }
        return result;
    }

    int n_vocab = llama_vocab_n_tokens(vocab);  // Initialize n_vocab here
    std::vector<llama_token> tokens(full_prompt.size() + 50);
    int32_t n_tokens = llama_tokenize(
        vocab,
        full_prompt.c_str(),
        (int32_t)full_prompt.size(),
        tokens.data(),
        (int32_t)tokens.size(),
        true,
        true
    );
    if (n_tokens <= 0) {
        last_model_response = "Tokenization failed.";
        char* result = (char*)malloc(last_model_response.size() + 1);
        if (result) {
            memcpy(result, last_model_response.c_str(), last_model_response.size() + 1);
        }
        return result;
    }
    tokens.resize(n_tokens);
    /*
    // Always prepend BOS token if missing (TinyLlama expects <s> at start)
    if (tokens.empty() || tokens[0] != 1) {
        tokens.insert(tokens.begin(), 1); // 1 is <s> for most LLaMA-family models
        std::cout << "[DEBUG] Prepended BOS token <s> (1) to tokens." << std::endl;
    }*/
    // Debug: Print tokens after BOS adjustment
    std::cout << "[DEBUG] Tokens after BOS adjustment: ";
    for (int i = 0; i < tokens.size(); ++i) std::cout << tokens[i] << " ";
    std::cout << std::endl;


    // Fix: Check prompt length against context window
    if (tokens.size() > 512) {
        last_model_response = "Prompt too long for model context window.";
        char* result = (char*)malloc(last_model_response.size() + 1);
        if (result) {
            memcpy(result, last_model_response.c_str(), last_model_response.size() + 1);
        }
        return result;
    }
    // Debug: Print tokens
    std::cout << "[DEBUG] Tokens: ";
    for (int i = 0; i < tokens.size(); ++i) std::cout << tokens[i] << " ";
    std::cout << std::endl;


    // DEBUG: print token count before decode
    std::cout << "[DEBUG] tokens.size() before decode: " << tokens.size() << std::endl;
    // Fix: Set all fields in batch as required by llama.cpp API
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (int i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i]; // Set token ID
        batch.pos[i] = i;           // Set position in sequence
        batch.n_seq_id[i] = 1;      // Number of sequence IDs for this token
        batch.seq_id[i][0] = 0;     // Sequence ID (single sequence)
    }
    batch.n_tokens = tokens.size(); // Set number of tokens in batch
    // DEBUG: Print batch state
    std::cout << "[DEBUG] Batch tokens: ";
    for (int i = 0; i < batch.n_tokens; ++i) std::cout << batch.token[i] << " ";
    std::cout << std::endl;
    std::cout << "[DEBUG] Batch positions: ";
    for (int i = 0; i < batch.n_tokens; ++i) std::cout << batch.pos[i] << " ";
    std::cout << std::endl;
    // DEBUG: run decode and print result
    int decode_result = llama_decode(inference_context->get(), batch);
    std::cout << "[DEBUG] llama_decode result: " << decode_result << std::endl;
    // Fix: Only decode once and check result
    if (decode_result) {
        llama_batch_free(batch);
        last_model_response = "Failed to evaluate prompt.";
        char* result = (char*)malloc(last_model_response.size() + 1);
        if (result) {
            memcpy(result, last_model_response.c_str(), last_model_response.size() + 1);
        }
        return result;
    }
    llama_batch_free(batch);


    last_model_response.clear();
    // changes here: reverted to sampler-based greedy sampling, but add a check for sum of logits before calling llama_sampler_apply
    // The error was: assertion failed: sum > 0.0 at ops.cpp:4725, caused when logits are all zero (decode failed or logits invalid)
    llama_sampler* sampler = llama_sampler_init_greedy(); // changes here: restore sampler object
    std::vector<llama_token_data> token_data(n_vocab);
    int n_past = tokens.size();
    std::vector<llama_token> input_token(1); // Declare input_token as a vector of size 1

    for (int i = 0; i < 32; ++i) {
        const float* logits = llama_get_logits(inference_context->get());

#ifdef _DEBUG
        // DEBUG: print all logits
        std::cout << "[DEBUG] First 10 logits: ";
        for (int dbg = 0; dbg <  std::min(10, n_vocab); ++dbg) std::cout << logits[dbg] << " ";
        std::cout << std::endl;
        // Print logits pointer
        std::cout << "[DEBUG] logits ptr: " << (void*)logits << std::endl;
        // Print min/max logits
        float min_logit = logits[0], max_logit = logits[0];
        for (int t = 1; t < n_vocab; ++t) {
            if (logits[t] < min_logit) min_logit = logits[t];
            if (logits[t] > max_logit) max_logit = logits[t];
        }
        std::cout << "[DEBUG] min_logit: " << min_logit << ", max_logit: " << max_logit << std::endl;
        // Print sum of exp(logits)
        double sum_exp = 0.0;
        for (int t = 0; t < n_vocab; ++t) {
            sum_exp += std::exp(logits[t]);
        }
        std::cout << "[DEBUG] sum_exp_logits: " << sum_exp << std::endl;

        // Print n_vocab and logits bounds
        std::cout << "[DEBUG] n_vocab: " << n_vocab << std::endl;
        std::cout << "[DEBUG] logits[0]: " << logits[0] << ", logits[n_vocab-1]: " << logits[n_vocab-1] << std::endl;
#endif
        float sum_logits = 0.0f;
        for (int t = 0; t < n_vocab; ++t) {
            token_data[t] = { t, logits[t], 0.0f };
            sum_logits += logits[t];
        }
#ifdef _DEBUG
        // Print first 10 token_data entries
        for (int i = 0; i < std::min(10, n_vocab); ++i) {
            std::cout << "[DEBUG] token_data[" << i << "]: id=" << token_data[i].id << ", logit=" << token_data[i].logit << std::endl;
        }
        std::cout << "[DEBUG] sum_logits: " << sum_logits << std::endl;
#endif
        // Fix: Add NaN check for logits sum
        if (std::isnan(sum_logits)) {
            std::cout << "[DEBUG] sum_logits is NaN, breaking out of generation loop." << std::endl;
            break;
        }
        /*if (sum_logits <= 0.0f) {
            std::cout << "[DEBUG] Sum of logits is zero or negative, breaking out of generation loop." << std::endl;
            break;
        }*/

        llama_token_data_array token_data_array = { token_data.data(), (size_t)n_vocab, -1, false };
        llama_sampler_apply(sampler, &token_data_array);
#ifdef _DEBUG
        std::cout << "[DEBUG] token_data_array.selected: " << token_data_array.selected << std::endl;
#endif
        if (token_data_array.selected < 0 || token_data_array.selected >= n_vocab) {
#ifdef _DEBUG
            std::cout << "[DEBUG] Sampler failed to select a valid token. Breaking out of generation loop." << std::endl;
#endif
            break;
        }
        llama_token token = token_data_array.data[token_data_array.selected].id;
#ifdef _DEBUG
        std::cout << "[DEBUG] Sampled token: " << token << std::endl;
#endif
        if (token == llama_vocab_eos(vocab)) {
            break;
        }
        const char* token_str = llama_vocab_get_text(vocab, token);
        if (token_str) {
            last_model_response += token_str;
        }
        input_token[0] = token;
#ifdef _DEBUG
        std::cout << "[DEBUG] About to decode token: " << token << std::endl;
        if (token < 0 || token >= n_vocab) {
            std::cout << "[ERROR] Sampled token out of vocab range! Breaking." << std::endl;
            break;
        }
#endif
        llama_batch next_batch = llama_batch_init(1, 0, 1);
        next_batch.token[0] = token;
        next_batch.pos[0] = n_past++;
        next_batch.n_seq_id[0] = 1;
        next_batch.seq_id[0][0] = 0;
        next_batch.n_tokens = 1;
#ifdef _DEBUG
        std::cout << "[DEBUG] next_batch.token[0]: " << next_batch.token[0] << std::endl;
        std::cout << "[DEBUG] next_batch.pos[0]: " << next_batch.pos[0] << std::endl;
        std::cout << "[DEBUG] next_batch.n_seq_id[0]: " << next_batch.n_seq_id[0] << std::endl;
        std::cout << "[DEBUG] next_batch.seq_id[0][0]: " << next_batch.seq_id[0][0] << std::endl;
        std::cout << "[DEBUG] next_batch.n_tokens: " << next_batch.n_tokens << std::endl;
#endif
        int decode_result_next = llama_decode(inference_context->get(), next_batch);
#ifdef _DEBUG
        std::cout << "[DEBUG] llama_decode (next) result: " << decode_result_next << std::endl;
#endif
        if (decode_result_next) {
            llama_batch_free(next_batch);
            break;
        }
        llama_batch_free(next_batch);
    }
    llama_sampler_free(sampler); // changes here: free the sampler object

    char* result = (char*)malloc(last_model_response.size() + 1);
    if (result) {
        memcpy(result, last_model_response.c_str(), last_model_response.size() + 1);
    }
    return result;
}

INTEGRATEDLLM_API void free_response(char* ptr) {
    free(ptr);
}

} // extern "C"
