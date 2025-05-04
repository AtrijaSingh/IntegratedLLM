#pragma once
#ifdef _WIN32
#define INTEGRATEDLLM_API __declspec(dllexport)
#else
#define INTEGRATEDLLM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

INTEGRATEDLLM_API int init(const char* model_path); // returns 0 on success
INTEGRATEDLLM_API void add_knowledge(const char* doc); // add a knowledge string
INTEGRATEDLLM_API void load_knowledge_file(const char* filepath); // load knowledge from file
INTEGRATEDLLM_API char* query(const char* prompt); // returns pointer to response
INTEGRATEDLLM_API void free_response(char* ptr);

#ifdef __cplusplus
}
#endif
