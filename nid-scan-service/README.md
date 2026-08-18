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

### Building natively on RHEL / Rocky

Rocky 10 is the floor, not a preference: `ort` downloads a prebuilt
`libonnxruntime` that needs glibc >= 2.38 and GCC >= 13, and Rocky 9 ships
glibc 2.34 / GCC 11. Two package names differ from the Debian equivalents —
zlib is `zlib-ng-compat-devel` (there is no `zlib-devel` on RHEL 10), and
`clang-devel`, `llvm-devel` and `openexr-devel` live in CRB, which is
disabled by default:

```sh
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --set-enabled crb
sudo dnf -y install gcc gcc-c++ make cmake ninja-build git pkgconf-pkg-config \
  clang clang-devel llvm-devel \
  libjpeg-turbo-devel libpng-devel libtiff-devel libwebp-devel openexr-devel \
  zlib-ng-compat-devel
export LIBCLANG_PATH=/usr/lib64
```

OpenCV 5 still has to be built from source (EPEL is on 4.x, and this project
needs the `geometry` module) — see [`Dockerfile`](./Dockerfile) for the exact
`cmake` invocation. Install it to `/usr/local` with
`-DCMAKE_INSTALL_LIBDIR=lib64`, then register it with the loader, which on
RHEL does not search `/usr/local/lib64` by default:

```sh
echo /usr/local/lib64 | sudo tee /etc/ld.so.conf.d/opencv5.conf
sudo ldconfig
export PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig
```

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
| `OCR_ORIENTATION_BATCH_SIZE` | `8` | Number of fixed-size text-line orientation inputs inferred together. |
| `OCR_RECOGNITION_BATCH_SIZE` | `1` | Recognition batch size. Keep at `1` unless the deployment backend and document corpus have been accuracy-tested at a larger value. |
| `OCR_RECOGNITION_WORKERS` | automatic | Independent recognition models used to process text lines concurrently. Defaults to logical CPUs divided by `OCR_MAX_CONCURRENCY`, capped at `8`; set explicitly to tune memory/latency for a deployment CPU. |
| `OCR_MAX_CONCURRENCY` | `2` | Maximum simultaneous scan, crop, or OCR jobs per service process; excess requests wait without occupying more inference capacity. |
| `OCR_DEBUG_LINES` | unset | If set, logs every recognized OCR line (text + confidence) at INFO level — useful for diagnosing missing/wrong fields. |

## Checks

```sh
cargo fmt --all -- --check
DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib cargo check --all-targets
DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib cargo clippy --all-targets --all-features -- -D warnings
DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib cargo test --all-targets
```
