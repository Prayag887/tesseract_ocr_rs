//! Shared ONNX Runtime setup for every session. ORT's defaults assume one model per machine;
//! here they cost memory, so:
//! - CPU arena off (it never returns freed activations),
//! - memory-pattern planning off (reserves buffers for the largest input ever seen),
//! - one small global thread pool instead of a per-session pool sized to the CPU count.
//! Restore the defaults with `ORT_CPU_ARENA=true ORT_MEMORY_PATTERN=true ORT_INTRA_THREADS=8
//! ORT_INTER_THREADS=8`.

use ort::environment::GlobalThreadPoolOptions;
use ort::ep::CPU;
use ort::session::builder::{PrepackedWeights, SessionBuilder};

use crate::error::AppError;

/// Process-wide intra-op threads; matches the default `OCR_MAX_CONCURRENCY` (2).
const DEFAULT_INTRA_THREADS: usize = 2;
/// Inter-op threads; unused without parallel execution, which no session enables.
const DEFAULT_INTER_THREADS: usize = 1;

/// Installs the global ORT environment. Must run before the first session is built, or it's a
/// no-op (logged as `committed=false`).
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
        // On the environment, so every session inherits it.
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

/// Base builder for every session. No thread settings: the global pool overrides them.
pub fn session_builder() -> Result<SessionBuilder, AppError> {
    let memory_pattern = env_bool("ORT_MEMORY_PATTERN", false);
    Ok(SessionBuilder::new()?.with_memory_pattern(memory_pattern)?)
}

/// Recognition-pool worker; shared [`PrepackedWeights`] so the pool stores one copy of the weights.
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
