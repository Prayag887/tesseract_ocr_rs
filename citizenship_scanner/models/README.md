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
- `super-resolution-10.onnx` — optional pre-detection upscaler for crops
  under `MIN_PREPROCESS_SHORT_SIDE`, gated behind `CITIZENSHIP_OCR_UPSCALE`
  (default off — see `src/upscale.rs`). The classic ESPCN-style sub-pixel
  CNN from the official ONNX Model Zoo, downloaded from
  `onnxmodelzoo/super-resolution-10` on Hugging Face (Apache 2.0). Fixed
  224x224 input, 3x upscale, Y-channel (luma) only — `upscale.rs` tiles the
  crop into 224x224 blocks and resizes Cr/Cb separately. Tested against two
  real scans before wiring in: makes the image visibly sharper but is *not*
  a clear OCR-accuracy win — it changes which misreads the recognizer
  makes rather than removing them, and on one real front-page scan it
  caused a wrong-attribution bug (a relative's name matched as the
  holder's) that hadn't occurred without it. Left disabled by default for
  that reason; flip `CITIZENSHIP_OCR_UPSCALE=true` to test it yourself.
