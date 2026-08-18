# Models

- `PP-OCRv6_medium_det.onnx` — text-region detection. Not shared with the
  NID/passport services (they use `ppocrv5_det.onnx`) — swapped in here
  after a real accuracy comparison against this service's own low-resolution
  test scans; see the comment on `DET_MODEL_PATH` in `src/local_ocr.rs` for
  what was tried and why this one won. Downloaded from
  `PaddlePaddle/PP-OCRv6_medium_det_onnx` on Hugging Face. Script-agnostic,
  like the v5 detector it replaced.
- `textline_ori.onnx`, `docaligner_lcnet100.onnx` — copies of the
  NID/passport services' models, unchanged (upright/rotated text-line
  classification, document-corner detection — both script-agnostic).
- `devanagari_rec.onnx` + `devanagari_rec_dict.txt` — copy of the
  NID/passport services' PP-OCRv5 Devanagari recognizer, unchanged. Its
  vocab also extends the full Latin base charset (digits, A-Z), so it reads
  both scripts printed on a Nepali citizenship certificate without a second
  recognizer. No larger Devanagari recognition model exists in PaddleOCR's
  model zoo to upgrade to (PP-OCRv6 dropped Devanagari from its unified
  recognizer).
