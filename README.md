# PP-OCRv5 document scanner

This test application detects and perspective-corrects a document with OpenCV,
then sends the corrected image to a lightweight PP-OCRv5 service. For bilingual
Nepali and English identity documents, configure PaddleOCR with the
`devanagari_PP-OCRv5_mobile_rec` recognition model. It supports Devanagari,
Nepali, English, and numbers and is approximately 7.5 MB.

## Architecture

1. Axum accepts an image with a 25 MiB request limit.
2. OpenCV detects and crops the document on Tokio's blocking pool.
3. Rust sends the JPEG to a persistent PP-OCRv5 pipeline through `POST /ocr`.
4. The API returns full text plus each recognized line's confidence and polygon.
5. Rust returns a compact JSON response containing only best-effort label/value fields.

## Start the PP-OCRv5 service

Install PaddleOCR in an isolated Python environment:

```sh
python3 -m venv .venv_paddleocr
source .venv_paddleocr/bin/activate
python -m pip install --upgrade pip
python -m pip install paddlepaddle paddleocr
paddlex --install serving
```

The repository includes [`OCR.yaml`](./OCR.yaml), pinned to the lightweight
PP-OCRv5 detection and bilingual Devanagari/English recognition models:

```yaml
pipeline_name: OCR
use_doc_preprocessor: false
SubModules:
  TextDetection:
    model_name: PP-OCRv5_mobile_det
  TextRecognition:
    model_name: devanagari_PP-OCRv5_mobile_rec
Serving:
  visualize: false
```

Start the service on loopback:

```sh
.venv_paddleocr/bin/paddlex --serve --pipeline OCR.yaml --device cpu --host 127.0.0.1 --port 8080
```

The first start downloads the selected Paddle models into Paddle's model cache;
the Rust repository does not contain or load model weights.

Verify that the endpoint exists:

```sh
curl -sS http://127.0.0.1:8080/ocr \
  -H 'content-type: application/json' \
  --data '{"file":"aGVsbG8=","fileType":1,"visualize":false}'
```

The example is not an image, so a structured Paddle error is expected.
Connection refusal means the service is not running.

## Run Rust

```sh
DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib cargo run
```

Open <http://127.0.0.1:3000>.

Configuration:

| Variable | Default | Meaning |
|---|---:|---|
| `PADDLE_OCR_URL` | `http://127.0.0.1:8080/ocr` | PP-OCRv5 endpoint |
| `PADDLE_OCR_API_KEY` | unset | Optional bearer token |
| `PADDLE_OCR_TIMEOUT_SECS` | `30` | End-to-end upstream timeout |
| `PADDLE_OCR_QUEUE_TIMEOUT_MS` | `2000` | Maximum local capacity wait |
| `PADDLE_OCR_MAX_CONCURRENCY` | `1` | Concurrent OCR requests |
| `PADDLE_OCR_MIN_CONFIDENCE` | `0.45` | Recognition confidence threshold |

Raise concurrency only after load-testing the Paddle deployment. The Rust client
keeps HTTP connections pooled and applies connect, queue, and total timeouts.

## Rust memory behavior

- The image is borrowed as `&[u8]` throughout OCR submission.
- The Base64 string required by Paddle's JSON API is allocated exactly once and
  borrowed by the serialized request.
- Paddle response strings are moved into output structures rather than cloned.
- Output vectors reserve capacity using the response sizes.
- Visualization images are disabled to reduce response memory and latency.

## Checks

```sh
cargo fmt --all -- --check
DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib cargo check --all-targets
DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib cargo clippy --all-targets --all-features -- -D warnings
DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib cargo test --all-targets
```
