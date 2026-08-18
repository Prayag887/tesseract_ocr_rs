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

Removed models (kept here as a record so they aren't re-added without the
measurements that argued against them):

- `super-resolution-10.onnx` (ESPCN 3x) and `docshadow_sd7k.onnx`
  (DocShadow, 114 MB) were both trialled as pre-detection enhancement.
  Scored against a hand-checked ground truth over three real card pairs
  (48 fields): neither models 35/48, upscale alone 33/48, docshadow alone
  33/48, both together 40/48 — so each was a *net loss* on its own and the
  pair bought +5 fields. The cost was ~2x peak RSS (3.8 GB vs 1.8 GB) and
  ~7x latency, because upscaling triples the image and the detector then
  works on 11.0 MP instead of 2.9 MP. Removed on that trade.
