//! Talks to the `llm_verifier` sidecar process (see `llm_verifier/src/main.rs`
//! for why it's a separate crate/binary rather than an in-process
//! dependency) over line-delimited JSON on its stdin/stdout.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::ocr::{Field, OcrLine, line_left, line_top};

const SIDECAR_BINARY: &str = "llm_verifier/target/release/llm_verifier";
const MODEL_PATH: &str = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";

pub struct LlmVerifier {
    // The whole child (not just its pipes) is held so the process is killed
    // when this drops, and so a dead/crashed sidecar is diagnosable from
    // `child.try_wait()` rather than silently reading EOF forever.
    state: Mutex<SidecarHandle>,
}

struct SidecarHandle {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[derive(Serialize)]
struct Request<'a> {
    lines: Vec<LineOut<'a>>,
    labels: Option<&'a [&'a str]>,
}

#[derive(Serialize)]
struct LineOut<'a> {
    text: &'a str,
    x: i32,
    y: i32,
}

#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    fields: Vec<Field>,
    #[serde(default)]
    error: Option<String>,
}

impl LlmVerifier {
    pub fn spawn() -> Result<Self, AppError> {
        let handle = Self::spawn_child()?;
        Ok(Self {
            state: Mutex::new(handle),
        })
    }

    fn spawn_child() -> Result<SidecarHandle, AppError> {
        let mut child = Command::new(SIDECAR_BINARY)
            .arg(MODEL_PATH)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(AppError::Io)?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
        Ok(SidecarHandle {
            child,
            stdin,
            stdout,
        })
    }

    /// Extracts fields for a known document type's field labels, or does
    /// open-ended label:value extraction when `labels` is `None`.
    pub fn extract(&self, lines: &[OcrLine], labels: Option<&[&str]>) -> Result<Vec<Field>, AppError> {
        let request = Request {
            lines: lines
                .iter()
                .map(|line| LineOut {
                    text: &line.text,
                    x: line_left(line),
                    y: line_top(line),
                })
                .collect(),
            labels,
        };
        let request_line = serde_json::to_string(&request)
            .map_err(|error| AppError::LlmProtocol(format!("failed to encode request: {error}")))?;

        let mut state = self.state.lock().expect("llm sidecar mutex poisoned");

        if let Ok(Some(status)) = state.child.try_wait() {
            return Err(AppError::LlmProtocol(format!(
                "llm_verifier sidecar exited early with {status}"
            )));
        }

        writeln!(state.stdin, "{request_line}").map_err(AppError::Io)?;
        state.stdin.flush().map_err(AppError::Io)?;

        let mut response_line = String::new();
        state
            .stdout
            .read_line(&mut response_line)
            .map_err(AppError::Io)?;
        if response_line.is_empty() {
            return Err(AppError::LlmProtocol(
                "llm_verifier sidecar closed its stdout".to_owned(),
            ));
        }

        let response: Response = serde_json::from_str(&response_line).map_err(|error| {
            AppError::LlmProtocol(format!("invalid sidecar response: {error}: {response_line}"))
        })?;

        if let Some(error) = response.error {
            return Err(AppError::LlmProtocol(error));
        }
        Ok(response.fields)
    }
}
