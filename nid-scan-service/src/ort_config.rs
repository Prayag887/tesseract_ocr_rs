//! One place where every ONNX Runtime session in this service gets its memory
//! behavior set.
//!
//! Sessions used to be built from a bare `Session::builder()` — ORT's
//! defaults, which assume one model with a whole machine to itself. This
//! container runs five sessions (det, textline-orientation, the recognition
//! worker pool, and `scanner`'s corner detector) next to two sibling services
//! on one small VM, where those defaults cost far more than they buy:
//!
//! - The **CPU arena allocator** (on by default) never returns freed
//!   activations to malloc. RSS settles at the high-water mark of the largest
//!   image the process has ever seen and stays there for the container's life.
//!   Disabled at the environment level below, so every session inherits it.
//! - **Memory pattern** planning (on by default) pre-reserves activation
//!   buffers sized for the largest input shape a session has run. Detection
//!   input is variable and capped only by `OCR_DET_MAX_SIDE`, so a single
//!   large upload permanently reserves buffers sized for it.
//! - `intra_threads` defaults to the logical CPU count (8 on the current
//!   host). That is *per session*: five sessions meant up to 40 ORT worker
//!   threads per container, each carrying its own allocation state, to serve
//!   the at-most-`OCR_MAX_CONCURRENCY` (2) requests actually in flight. The
//!   environment's global thread pool replaces them with one shared pool.
//!
//! Every knob is env-overridable so a regression can be bisected against the
//! old behavior without a rebuild: `ORT_CPU_ARENA=true`,
//! `ORT_MEMORY_PATTERN=true`, `ORT_INTRA_THREADS=8`, `ORT_INTER_THREADS=8`
//! restores what the defaults did.

use ort::environment::GlobalThreadPoolOptions;
use ort::ep::CPU;
use ort::session::builder::{PrepackedWeights, SessionBuilder};

use crate::error::AppError;

/// Intra-op threads for the shared global pool. This is the whole process's
/// budget, not a per-session one. Two matches `OCR_MAX_CONCURRENCY`'s default:
/// there is no point having more parallelism inside an operator than there are
/// requests allowed to be in flight.
const DEFAULT_INTRA_THREADS: usize = 2;
/// Inter-op threads. Only used when a session runs independent branches of its
/// graph concurrently, which needs `with_parallel_execution` — never enabled
/// for any session here — so one is enough.
const DEFAULT_INTER_THREADS: usize = 1;

/// Installs the process-wide ONNX Runtime environment.
///
/// Must be called before the first session is built: `ort` creates a default
/// environment lazily on first use, and
/// [`commit`](ort::environment::EnvironmentBuilder::commit) is a no-op once
/// that has happened — it returns `false` rather than failing, which is why
/// the flag is logged instead of ignored. A `committed=false` line means some
/// earlier code already built a session and none of this took effect.
pub fn init_runtime() -> Result<(), AppError> {
    let intra_threads = env_usize("ORT_INTRA_THREADS", DEFAULT_INTRA_THREADS).clamp(1, 64);
    let inter_threads = env_usize("ORT_INTER_THREADS", DEFAULT_INTER_THREADS).clamp(1, 64);
    let cpu_arena = env_bool("ORT_CPU_ARENA", false);

    let thread_pool = GlobalThreadPoolOptions::default()
        .with_intra_threads(intra_threads)?
        .with_inter_threads(inter_threads)?;

    let committed = ort::init()
        .with_name("ocr")
        .with_telemetry(false)
        // Registered on the environment rather than on each builder so a
        // session added later cannot silently miss it. A session-level
        // execution provider would take precedence over this, so no session
        // here sets one.
        .with_execution_providers([CPU::default().with_arena_allocator(cpu_arena).build()])
        .with_global_thread_pool(thread_pool)
        .commit();

    tracing::info!(
        intra_threads,
        inter_threads,
        cpu_arena,
        committed,
        "ONNX Runtime environment configured"
    );
    Ok(())
}

/// The builder every session in this service is created from.
///
/// Deliberately does *not* set `intra_threads`/`inter_threads`: with the
/// environment's global thread pool active, ONNX Runtime ignores the
/// session-level thread counts, so setting them here would read as if it
/// controlled something it doesn't. [`init_runtime`] owns them.
pub fn session_builder() -> Result<SessionBuilder, AppError> {
    let memory_pattern = env_bool("ORT_MEMORY_PATTERN", false);
    Ok(SessionBuilder::new()?.with_memory_pattern(memory_pattern)?)
}

/// Builder for one recognition-pool worker.
///
/// The pool is N sessions over the *same* `devanagari_rec.onnx`. Handing every
/// worker the same [`PrepackedWeights`] container makes ORT prepack that
/// model's weights once and share the result across the pool, rather than each
/// session materializing its own copy of identical repacked buffers — the
/// largest duplicate allocation in this process, since the pool is the one
/// place a single model is loaded more than once.
pub fn rec_session_builder(weights: &PrepackedWeights) -> Result<SessionBuilder, AppError> {
    Ok(session_builder()?.with_prepacked_weights(weights)?)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<bool>().ok())
        .unwrap_or(default)
}
