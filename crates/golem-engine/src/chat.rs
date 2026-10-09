//! Chat rendering and output parsing through llama.cpp's chat layer
//! (`common/chat.h`, via `golem-llama-sys/shim/golem_chat.h`): every prompt —
//! tools or not — renders here, so one transcript always yields one prefix.
#![allow(unsafe_code)]

use std::ffi::{CStr, CString, c_char};
use std::rc::Rc;

use golem_llama_sys::raw;

use crate::error::EngineError;

/// Owned `golem_chat`; `Drop` calls `golem_chat_free`.
struct Handle(*mut raw::golem_chat);

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: the pointer came from golem_chat_init and is freed exactly once here.
        unsafe { raw::golem_chat_free(self.0) };
    }
}

/// Takes ownership of a shim-malloc'd string.
///
/// # Safety
/// `p` is NULL or a NUL-terminated string from the shim, not yet freed.
unsafe fn take(p: *mut c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: per the contract above.
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    // SAFETY: shim-allocated, freed exactly once.
    unsafe { raw::golem_chat_str_free(p) };
    s
}

fn cstr(s: &str, what: &str) -> Result<CString, EngineError> {
    CString::new(s).map_err(|_| EngineError::ChatTemplate(format!("{what} contains NUL")))
}

/// The session's chat templates. `!Send`: the shim's request table is
/// unsynchronized and only the decode thread touches it.
#[derive(Clone)]
pub struct ChatTemplates(Rc<Handle>);

/// A rendered request: the prompt plus the parser (and grammar) kept for it.
pub struct Rendered {
    /// Template output to tokenize.
    pub prompt: String,
    /// Parser for this request's generated text.
    pub parser: OutputParser,
    pub(crate) has_grammar: bool,
}

/// Parses one request's generated text; dropping it releases the request's
/// parser params in the shim.
pub struct OutputParser {
    chat: Rc<Handle>,
    handle: i32,
}

impl ChatTemplates {
    /// Templates from `src`, else the model metadata (chatml when it ships
    /// none).
    pub(crate) fn from_model(
        model: *const raw::llama_model,
        src: Option<&str>,
    ) -> Result<Self, EngineError> {
        let src = src.map(|s| cstr(s, "template")).transpose()?;
        // SAFETY: model is a live llama_model; a NULL override selects its metadata.
        let p = unsafe {
            raw::golem_chat_init(model, src.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()))
        };
        Self::wrap(p)
    }

    /// Templates from a jinja source, without a model (no BOS/EOS tokens).
    pub fn from_source(src: &str) -> Result<Self, EngineError> {
        let src = cstr(src, "template")?;
        // SAFETY: NULL model is allowed with a non-empty override; src is live.
        let p = unsafe { raw::golem_chat_init(std::ptr::null(), src.as_ptr()) };
        Self::wrap(p)
    }

    fn wrap(p: *mut raw::golem_chat) -> Result<Self, EngineError> {
        if p.is_null() {
            return Err(EngineError::TemplateUnavailable);
        }
        Ok(Self(Rc::new(Handle(p))))
    }

    /// Renders OpenAI `messages` (+ optional `tools`/`tool_choice` JSON) with
    /// the generation prompt appended.
    pub fn render(
        &self,
        messages_json: &str,
        tools_json: Option<&str>,
        tool_choice: Option<&str>,
    ) -> Result<Rendered, EngineError> {
        let m = cstr(messages_json, "messages")?;
        let t = tools_json.map(|t| cstr(t, "tools")).transpose()?;
        let c = tool_choice.map(|c| cstr(c, "tool_choice")).transpose()?;
        let (mut prompt, mut grammar, mut handle) = (std::ptr::null_mut(), std::ptr::null_mut(), 0);
        // SAFETY: the handle is live; inputs are live NUL-terminated strings or
        // NULL; outputs are written only on success.
        let rc = unsafe {
            raw::golem_chat_apply(
                self.0.0,
                m.as_ptr(),
                t.as_ref().map_or(std::ptr::null(), |t| t.as_ptr()),
                c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                &mut prompt,
                &mut grammar,
                &mut handle,
            )
        };
        if rc != 0 {
            return Err(EngineError::ChatTemplate(format!(
                "golem_chat_apply failed with code {rc}"
            )));
        }
        // SAFETY: success → both are shim-allocated strings.
        let (prompt, grammar) = unsafe { (take(prompt), take(grammar)) };
        Ok(Rendered {
            prompt,
            parser: OutputParser {
                chat: self.0.clone(),
                handle,
            },
            has_grammar: !grammar.is_empty(),
        })
    }
}

impl OutputParser {
    /// Parses `text` into an OpenAI assistant message
    /// `{content, reasoning_content?, tool_calls?}`.
    pub fn parse(&self, text: &str, partial: bool) -> Result<serde_json::Value, EngineError> {
        let t = cstr(text, "generated text")?;
        let mut out = std::ptr::null_mut();
        // SAFETY: handle and request are live (released only in Drop); t is live.
        let rc = unsafe {
            raw::golem_chat_parse(
                self.chat.0,
                self.handle,
                t.as_ptr(),
                i32::from(partial),
                &mut out,
            )
        };
        if rc != 0 {
            return Err(EngineError::ChatTemplate(format!(
                "golem_chat_parse failed with code {rc}"
            )));
        }
        // SAFETY: success → shim-allocated string.
        let s = unsafe { take(out) };
        serde_json::from_str(&s).map_err(|e| EngineError::ChatTemplate(e.to_string()))
    }

    /// The request's grammar sampler, `None` without a grammar; the caller
    /// owns the returned sampler.
    pub(crate) fn grammar_sampler(
        &self,
        vocab: *const raw::llama_vocab,
    ) -> Result<Option<*mut raw::llama_sampler>, EngineError> {
        let mut s = std::ptr::null_mut();
        // SAFETY: handle, request and vocab are live.
        let rc =
            unsafe { raw::golem_chat_grammar_sampler(self.chat.0, self.handle, vocab, &mut s) };
        if rc != 0 {
            return Err(EngineError::ChatTemplate(format!(
                "golem_chat_grammar_sampler failed with code {rc}"
            )));
        }
        Ok((!s.is_null()).then_some(s))
    }
}

impl Drop for OutputParser {
    fn drop(&mut self) {
        // SAFETY: the handle is live (Rc) and the request is released once.
        unsafe { raw::golem_chat_release(self.chat.0, self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GLM: &str = include_str!("../tests/fixtures/glm-5.3-flash.jinja");

    fn tools() -> String {
        serde_json::json!([{"type":"function","function":{"name":"get_weather","parameters":{
            "type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}])
        .to_string()
    }

    fn user() -> String {
        serde_json::json!([{"role":"user","content":"weather in Paris?"}]).to_string()
    }

    /// `auto` parsing on the target template: canned model text with a GLM
    /// `<tool_call>` block parses to an OpenAI tool call.
    #[test]
    fn glm_auto_parses_canned_tool_call() {
        let t = ChatTemplates::from_source(GLM).unwrap();
        let r = t.render(&user(), Some(&tools()), Some("\"auto\"")).unwrap();
        assert!(
            r.prompt.contains("get_weather"),
            "tools rendered into the prompt"
        );
        let msg = r
            .parser
            .parse(
                "need weather</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
                false,
            )
            .unwrap();
        let tc = &msg["tool_calls"][0]["function"];
        assert_eq!(tc["name"], "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(tc["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, serde_json::json!({"city":"Paris"}));
        assert_eq!(msg["reasoning_content"], "need weather");
    }

    #[test]
    fn glm_plain_answer_has_no_tool_calls() {
        let t = ChatTemplates::from_source(GLM).unwrap();
        let r = t.render(&user(), Some(&tools()), None).unwrap();
        let msg = r.parser.parse("</think>It is sunny.", false).unwrap();
        assert_eq!(msg["content"], "It is sunny.");
        assert!(msg.get("tool_calls").is_none());
    }

    #[test]
    fn named_choice_outside_tools_is_an_error() {
        let t = ChatTemplates::from_source(GLM).unwrap();
        let choice = r#"{"type":"function","function":{"name":"nope"}}"#;
        assert!(t.render(&user(), Some(&tools()), Some(choice)).is_err());
    }
}
