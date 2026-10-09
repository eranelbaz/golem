#include "golem_chat.h"

#include "chat.h"
#include "common.h"
#include "llama.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <vector>

namespace {

struct request {
    common_chat_parser_params parser;
    std::string grammar;
    bool grammar_lazy = false;
    std::vector<common_grammar_trigger> triggers;
};

char * dup(const std::string & s) {
    char * p = static_cast<char *>(std::malloc(s.size() + 1));
    if (p) {
        std::memcpy(p, s.data(), s.size());
        p[s.size()] = '\0';
    }
    return p;
}

// A required call means no answer text: upstream's root rule still admits
// free content before the call marker (`until-N?`), which a model can fill
// until max_tokens. Dropping it forces the call right after any reasoning.
std::string force_call(const std::string & grammar) {
    static const std::regex content(R"( until-\d+\?)");
    std::string out;
    size_t pos = 0;
    while (pos < grammar.size()) {
        size_t eol  = grammar.find('\n', pos);
        size_t end  = eol == std::string::npos ? grammar.size() : eol + 1;
        auto   line = grammar.substr(pos, end - pos);
        out += line.rfind("root ::=", 0) == 0 ? std::regex_replace(line, content, "") : line;
        pos = end;
    }
    return out;
}

common_chat_tool_choice parse_tool_choice(const char * tc, std::string & named) {
    if (!tc || !*tc) {
        return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }
    auto j = common_json::parse(tc);
    if (j.is_string()) {
        return common_chat_tool_choice_parse_oaicompat(j.get<std::string>());
    }
    named = j.at("function").at("name").get<std::string>();
    return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
}

}  // namespace

struct golem_chat {
    common_chat_templates_ptr tmpls;
    std::map<int, request> requests;
    int next = 0;
};

extern "C" {

golem_chat * golem_chat_init(const llama_model * model, const char * tmpl_override) {
    try {
        auto * c = new golem_chat();
        c->tmpls = common_chat_templates_init(model, tmpl_override ? tmpl_override : "");
        return c;
    } catch (...) {
        return nullptr;
    }
}

int golem_chat_apply(golem_chat * chat, const char * messages_json, const char * tools_json,
                     const char * tool_choice, char ** out_prompt, char ** out_grammar, int * out_handle) {
    if (!chat || !messages_json || !out_prompt || !out_grammar || !out_handle) {
        return 1;
    }
    try {
        common_chat_templates_inputs in;
        in.messages         = common_chat_msgs_parse_oaicompat(common_json::parse(messages_json));
        in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        std::string named;
        in.tool_choice = parse_tool_choice(tool_choice, named);
        if (tools_json && *tools_json) {
            in.tools = common_chat_tools_parse_oaicompat(common_json::parse(tools_json));
        }
        // A named choice is `required` over the one named tool: the grammar can
        // then only produce a call to it.
        if (!named.empty()) {
            std::vector<common_chat_tool> only;
            for (auto & t : in.tools) {
                if (t.name == named) {
                    only.push_back(t);
                }
            }
            if (only.empty()) {
                return 2;
            }
            in.tools = std::move(only);
        }
        in.parallel_tool_calls = false;
        auto params            = common_chat_templates_apply(chat->tmpls.get(), in);

        request r;
        r.parser                  = common_chat_parser_params(params);
        r.parser.reasoning_format = in.reasoning_format;
        r.parser.parse_tool_calls = !in.tools.empty() && in.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
        if (!params.parser.empty()) {
            r.parser.parser.load(params.parser);
        }
        r.grammar      = in.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED && !params.grammar_lazy ?
                             force_call(params.grammar) :
                             params.grammar;
        r.grammar_lazy = params.grammar_lazy;
        r.triggers     = params.grammar_triggers;

        char * p = dup(params.prompt);
        char * g = dup(params.grammar);
        if (!p || !g) {
            std::free(p);
            std::free(g);
            return 3;
        }
        int h = chat->next++;
        chat->requests[h] = std::move(r);
        *out_prompt       = p;
        *out_grammar      = g;
        *out_handle       = h;
        return 0;
    } catch (...) {
        return 4;
    }
}

int golem_chat_grammar_sampler(golem_chat * chat, int handle, const llama_vocab * vocab, llama_sampler ** out) {
    if (!chat || !vocab || !out) {
        return 1;
    }
    *out = nullptr;
    try {
        auto it = chat->requests.find(handle);
        if (it == chat->requests.end()) {
            return 2;
        }
        const request & r = it->second;
        if (r.grammar.empty()) {
            return 0;
        }
        // Trigger translation mirrors common/sampling.cpp common_sampler_init.
        std::vector<std::string> patterns;
        std::vector<llama_token> tokens;
        for (const auto & t : r.triggers) {
            switch (t.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                    patterns.push_back(regex_escape(t.value));
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                    patterns.push_back(t.value);
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
                    const auto & v = t.value;
                    std::string a  = "^$";
                    if (!v.empty()) {
                        a = (v.front() != '^' ? "^" : "") + v + (v.back() != '$' ? "$" : "");
                    }
                    patterns.push_back(a);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                    tokens.push_back(t.token);
                    break;
            }
        }
        std::vector<const char *> pc;
        for (const auto & p : patterns) {
            pc.push_back(p.c_str());
        }
        llama_sampler * s =
            r.grammar_lazy ?
                llama_sampler_init_grammar_lazy_patterns(vocab, r.grammar.c_str(), "root", pc.data(), pc.size(),
                                                         tokens.data(), tokens.size()) :
                llama_sampler_init_grammar(vocab, r.grammar.c_str(), "root");
        if (!s) {
            return 3;
        }
        // The template already placed the generation prompt in the prompt, so
        // an eager tool-call grammar must advance past it (common_sampler_init).
        // Leading tokens that only spell a tokenizer-inserted space prefix are
        // skipped: upstream skips one, but a byte-fallback vocab spells `▁` as
        // several byte tokens.
        if (!r.grammar_lazy && !r.parser.generation_prompt.empty()) {
            const auto & gp = r.parser.generation_prompt;
            auto toks       = common_tokenize(vocab, gp, false, true);
            std::vector<std::string> pieces;
            for (auto t : toks) {
                pieces.push_back(common_token_to_piece(vocab, t, true));
            }
            size_t skip = 0;
            for (; skip <= toks.size(); skip++) {
                std::string rest;
                for (size_t i = skip; i < toks.size(); i++) {
                    rest += pieces[i];
                }
                if (rest == gp) {
                    break;
                }
            }
            if (skip > toks.size()) {
                llama_sampler_free(s);
                return 5;
            }
            for (size_t i = skip; i < toks.size(); i++) {
                llama_sampler_accept(s, toks[i]);
            }
        }
        *out = s;
        return 0;
    } catch (...) {
        return 4;
    }
}

int golem_chat_parse(golem_chat * chat, int handle, const char * text, int is_partial, char ** out_msg_json) {
    if (!chat || !text || !out_msg_json) {
        return 1;
    }
    try {
        auto it = chat->requests.find(handle);
        if (it == chat->requests.end()) {
            return 2;
        }
        auto msg = common_chat_parse(text, is_partial != 0, it->second.parser);
        msg.role = "assistant";
        char * s = dup(msg.to_json_oaicompat().dump());
        if (!s) {
            return 3;
        }
        *out_msg_json = s;
        return 0;
    } catch (...) {
        return 4;
    }
}

void golem_chat_release(golem_chat * chat, int handle) {
    if (chat) {
        chat->requests.erase(handle);
    }
}

void golem_chat_str_free(char * s) {
    std::free(s);
}

void golem_chat_free(golem_chat * chat) {
    delete chat;
}

}  // extern "C"
