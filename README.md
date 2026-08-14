# Document scanner (pure Rust, ONNX)

Detects and perspective-corrects a document photo, then reads it — bilingual
Nepali (Devanagari) and English identity documents — into structured
label/value JSON.

**No Python.** Detection, crop, text detection, text recognition, and field
extraction all run in-process in this one Rust binary, via OpenCV's `dnn`
module loading ONNX models directly. There is no service to start, no
sidecar to keep alive, no model download step — the ONNX weights are
committed in [`models/`](./models) and loaded at startup.

## Run it

```sh
DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib cargo run --release
```

Open <http://127.0.0.1:3000>.

(`DYLD_FALLBACK_LIBRARY_PATH` is a macOS-only requirement for OpenCV's linker
to find `libclang`; drop it on Linux.)

## Architecture

1. Axum accepts an image with a 25 MiB request limit (`POST /scan`).
2. `docaligner_lcnet100.onnx` (heatmap regression, one channel per corner)
   finds the document's 4 corners; OpenCV perspective-warps and crops it.
3. `POST /extract/{id}` runs the crop through three more ONNX models in
   sequence: `ppocrv5_det.onnx` (text-line detection, DB algorithm),
   `textline_ori.onnx` (per-line 0°/180° orientation), `devanagari_rec.onnx`
   (bilingual Devanagari + English text recognition, CTC decode).
4. A declarative field-signature table (`src/ocr.rs`) maps recognized
   lines to structured output fields (name, date of birth, ID number, ...)
   by keyword, shape, and position — deterministic, no model involved, and
   extensible to a new document type by adding a table entry rather than
   writing new code.

All four `.onnx` files are OpenCV `dnn::Net`s, each guarded by its own
`Mutex` and loaded once at process startup.

## Optional legacy backend

An HTTP client for the original Python PaddleX sidecar (`PaddleOcrClient`)
is still in the codebase as a rollback path, selected via `OCR_BACKEND`:

| Value | Behavior |
|---|---|
| `local` (default) | In-process ONNX inference, this Rust binary only. |
| `paddlex` | Calls out to a PaddleX `/ocr` HTTP service — requires the Python setup this project no longer needs by default. Not documented further here; only kept for A/B comparison against the native path. |

## Configuration

| Variable | Default | Meaning |
|---|---:|---|
| `OCR_BACKEND` | `local` | `local` (native ONNX) or `paddlex` (legacy HTTP sidecar). |
| `PADDLE_OCR_MIN_CONFIDENCE` | `0.45` | Recognition confidence threshold below which a line is dropped. |
| `OCR_DEBUG_LINES` | unset | If set, logs every recognized OCR line (text + confidence) at INFO level — useful for diagnosing missing/wrong fields. |

## Checks

```sh
cargo fmt --all -- --check
DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib cargo check --all-targets
DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib cargo clippy --all-targets --all-features -- -D warnings
DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib cargo test --all-targets
```
