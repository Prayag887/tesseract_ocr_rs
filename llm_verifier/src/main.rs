//! Standalone sidecar process: reads OCR lines (text + rough position) from
//! stdin, one JSON request per line, and writes back structured fields as
//! JSON, one response per line.
//!
//! Lives in its own crate/build (never in the same dependency graph as the
//! main `tesseract` binary's `opencv` dependency) because `llama-cpp-sys-2`
//! pulls in `bindgen`, and unifying `bindgen`'s `clang-sys` with
//! `opencv-rust`'s own `clang-sys` usage in a single cargo build triggers a
//! known, unresolved upstream panic ("a `libclang` shared library is not
//! loaded on this thread" — opencv-rust issue #680). Two separate builds
//! means two separate dependency resolutions, so the conflict can't happen.
//!
//! Protocol is line-delimited JSON over stdin/stdout deliberately, not a
//! one-shot CLI per request: loading a 1.5B GGUF model takes real time, so
//! this process starts once and stays resident for the life of the parent.

use std::io::{self, BufRead, Write};
use std::num::NonZeroU32;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use serde::{Deserialize, Serialize};

const MAX_OUTPUT_TOKENS: i32 = 800;
// Default n_ctx is 512 — too small for a document's worth of OCR lines plus
// output. Must exceed prompt tokens + MAX_OUTPUT_TOKENS or decode runs out
// of KV cache space mid-generation.
const N_CTX: u32 = 4096;

#[derive(Deserialize)]
struct Request {
    lines: Vec<OcrLineIn>,
    /// Known output field labels for the detected document type, or `null`
    /// for open-ended label:value extraction on an unrecognized type.
    labels: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct OcrLineIn {
    text: String,
    x: i32,
    y: i32,
}

#[derive(Serialize)]
struct Response {
    fields: Vec<FieldOut>,
}

#[derive(Serialize, Deserialize)]
struct FieldOut {
    label: String,
    value: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .expect("usage: llm_verifier <path-to-gguf>");

    let backend = LlamaBackend::init().expect("failed to init llama backend");
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .expect("failed to load model");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line.expect("failed to read stdin line");
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<Request>(&line) {
            Ok(request) => match handle_request(&backend, &model, request) {
                Ok(fields) => serde_json::to_string(&Response { fields }),
                Err(message) => serde_json::to_string(&ErrorResponse { error: message }),
            },
            Err(error) => serde_json::to_string(&ErrorResponse {
                error: format!("invalid request: {error}"),
            }),
        }
        .expect("failed to serialize response");

        writeln!(stdout, "{response}").expect("failed to write stdout");
        stdout.flush().expect("failed to flush stdout");
    }
}

fn handle_request(
    backend: &LlamaBackend,
    model: &LlamaModel,
    request: Request,
) -> Result<Vec<FieldOut>, String> {
    let prompt = build_prompt(&request.lines, request.labels.as_deref());
    let schema = build_schema(request.labels.as_deref());
    if std::env::var("LLM_DEBUG").is_ok() {
        eprintln!("=== schema ===\n{schema}\n=== prompt ===\n{prompt}");
    }

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_batch(N_CTX);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|error| format!("failed to create context: {error}"))?;

    let tokens_list = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|error| format!("failed to tokenize: {error}"))?;

    let mut batch = LlamaBatch::new(4096, 1);
    let last_index = i32::try_from(tokens_list.len())
        .map_err(|_| "prompt too long".to_owned())?
        - 1;
    for (i, token) in (0_i32..).zip(&tokens_list) {
        batch
            .add(*token, i, &[0], i == last_index)
            .map_err(|error| format!("failed to build batch: {error}"))?;
    }
    ctx.decode(&mut batch)
        .map_err(|error| format!("decode failed: {error}"))?;

    // GBNF via `LlamaSampler::grammar` hits a known, unresolved llama.cpp
    // bug ("Unexpected empty grammar stack after accepting piece: {" —
    // ggml-org/llama.cpp#18173): the JSON grammar's own parser dies right
    // after the opening brace. llguidance is a separate constrained-decoding
    // engine (not llama.cpp's native grammar code), unaffected by that bug.
    let grammar = TopLevelGrammar::from_tagged_str("json", &schema)
        .map_err(|error| format!("invalid schema: {error}"))?;
    let tok_env = LlamaSampler::llguidance_tok_env(model);
    let factory =
        ParserFactory::new_simple(&tok_env).map_err(|error| format!("llguidance init: {error}"))?;
    let parser = factory
        .create_parser(grammar)
        .map_err(|error| format!("llguidance parser: {error}"))?;
    let matcher = Matcher::new(Ok(parser));
    let llg_sampler = LlamaSampler::from(matcher);
    let mut sampler = LlamaSampler::chain_simple([llg_sampler, LlamaSampler::greedy()]);

    let mut n_cur = batch.n_tokens();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();

    while n_cur <= tokens_list.len() as i32 + MAX_OUTPUT_TOKENS {
        if std::env::var("LLM_DEBUG").is_ok() {
            eprintln!("[debug] sampling token at n_cur={n_cur}");
        }
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        if std::env::var("LLM_DEBUG").is_ok() {
            eprintln!("[debug] sampled token {token:?}");
        }
        sampler.accept(token);

        if token == model.token_eos() {
            break;
        }

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|error| format!("failed to detokenize: {error}"))?;
        output.push_str(&piece);

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|error| format!("failed to build batch: {error}"))?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|error| format!("decode failed: {error}"))?;
    }

    if std::env::var("LLM_DEBUG").is_ok() {
        eprintln!("=== raw output ===\n{output}");
    }
    parse_output(&output, request.labels.as_deref())
}

fn build_prompt(lines: &[OcrLineIn], labels: Option<&[String]>) -> String {
    let mut lines_text = String::new();
    for line in lines {
        lines_text.push_str(&format!("{} @ ({}, {})\n", line.text, line.x, line.y));
    }

    let instructions = match labels {
        Some(labels) => format!(
            "You are given OCR text lines read off a scanned identity document, \
             each shown as `text @ (x, y)` where (x, y) is that line's top-left \
             pixel position. Some fields are printed as \"Label: Value\" on one \
             line; others show the label on its own line with the value on a \
             separate nearby line (often just below it). Match each value to its \
             correct field using both the text and its position, regardless of \
             which layout this document uses. Extract exactly these fields: {}. \
             If a field's value cannot be found, use an empty string for it. \
             Respond with only the JSON object, no explanation.",
            labels.join(", ")
        ),
        None => "You are given OCR text lines read off a scanned document, each \
                  shown as `text @ (x, y)` where (x, y) is that line's top-left \
                  pixel position. Find every label/value pair on the document — a \
                  label is a short caption or heading, and its value is the data \
                  printed next to it or on a separate nearby line below it. \
                  Respond with only a JSON array of {\"label\": ..., \"value\": ...} \
                  objects, no explanation."
            .to_owned(),
    };

    format!(
        "<|im_start|>system\n{instructions}<|im_end|>\n<|im_start|>user\n{lines_text}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn build_schema(labels: Option<&[String]>) -> String {
    match labels {
        Some(labels) => {
            let properties: serde_json::Map<String, serde_json::Value> = labels
                .iter()
                .map(|label| (label.clone(), serde_json::json!({"type": "string"})))
                .collect();
            serde_json::json!({
                "type": "object",
                "properties": properties,
                "required": labels,
            })
            .to_string()
        }
        None => serde_json::json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["label", "value"],
            }
        })
        .to_string(),
    }
}

fn normalize_key(key: &str) -> String {
    key.trim()
        .to_uppercase()
        .replace(['_', '-'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_output(output: &str, labels: Option<&[String]>) -> Result<Vec<FieldOut>, String> {
    match labels {
        Some(labels) => {
            let object: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(output)
                    .map_err(|error| format!("model produced invalid JSON: {error}: {output}"))?;
            // The model (or the schema-to-grammar step) sometimes turns
            // "NATIONAL ID NUMBER" into "NATIONAL_ID_NUMBER" in its output
            // keys despite the schema asking for spaces — normalize both
            // sides before matching rather than trust exact key equality.
            let normalized: std::collections::HashMap<String, &serde_json::Value> = object
                .iter()
                .map(|(key, value)| (normalize_key(key), value))
                .collect();
            let fields = labels
                .iter()
                .filter_map(|label| {
                    let value = normalized.get(&normalize_key(label))?.as_str()?.trim();
                    (!value.is_empty()).then(|| FieldOut {
                        label: label.clone(),
                        value: value.to_owned(),
                    })
                })
                .collect();
            Ok(fields)
        }
        None => {
            let items: Vec<FieldOut> = serde_json::from_str(output)
                .map_err(|error| format!("model produced invalid JSON: {error}: {output}"))?;
            Ok(items
                .into_iter()
                .filter(|field| !field.label.trim().is_empty() && !field.value.trim().is_empty())
                .collect())
        }
    }
}
