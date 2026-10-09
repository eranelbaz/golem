// C ABI over llama.cpp's chat layer (common/chat.h): template rendering with
// tools, tool-call grammars, and output parsing. Every C++ exception is caught
// here and reported as a nonzero return. Owned strings are freed with
// golem_chat_str_free.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct llama_model;
struct llama_vocab;
struct llama_sampler;

typedef struct golem_chat golem_chat;

// Templates from `model` metadata (chatml when it ships none), or from
// `tmpl_override` when non-NULL and non-empty (`model` may then be NULL).
// NULL on failure.
golem_chat * golem_chat_init(const struct llama_model * model, const char * tmpl_override);

// messages/tools/tool_choice are OpenAI JSON (tools/tool_choice may be NULL;
// tool_choice is "auto"|"none"|"required" or {"type":"function","function":{"name":..}}).
// out_prompt/out_grammar are malloc'd; out_handle indexes the parser params kept
// in `chat` until golem_chat_release. Returns 0 on success.
int golem_chat_apply(golem_chat * chat, const char * messages_json, const char * tools_json,
                     const char * tool_choice, char ** out_prompt, char ** out_grammar, int * out_handle);

// Grammar sampler for `handle` (lazy with its triggers, or eager with the
// generation prompt prefilled), or *out = NULL when the request has no
// grammar. The caller owns the sampler. Returns 0 on success.
int golem_chat_grammar_sampler(golem_chat * chat, int handle, const struct llama_vocab * vocab,
                               struct llama_sampler ** out);

// Parses generated `text` into an OpenAI assistant message JSON
// {content, reasoning_content, tool_calls}. Returns 0 on success.
int golem_chat_parse(golem_chat * chat, int handle, const char * text, int is_partial, char ** out_msg_json);

// Frees the stored parser params for that request.
void golem_chat_release(golem_chat * chat, int handle);

void golem_chat_str_free(char * s);
void golem_chat_free(golem_chat * chat);

#ifdef __cplusplus
}
#endif
