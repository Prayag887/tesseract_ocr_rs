# Models

All four files here are copies of the NID service's models — no retraining
or separate sourcing needed:

- `ppocrv5_det.onnx`, `textline_ori.onnx`, `docaligner_lcnet100.onnx` —
  script-agnostic (text-region detection, upright/rotated classification,
  document-corner detection).
- `devanagari_rec.onnx` + `devanagari_rec_dict.txt` — despite the name,
  this dict is PP-OCRv5's Devanagari vocab, which extends the full Latin
  base charset. It already contains every character an MRZ can print
  (digits, A-Z, `<` filler) — see `local_ocr.rs`'s `REC_MODEL_PATH` comment.
  MRZ-specific handling (which lines are MRZ rows, character-set filtering,
  TD3 field/checksum decode) lives entirely in `local_ocr.rs` and `mrz.rs`,
  not in the model.
