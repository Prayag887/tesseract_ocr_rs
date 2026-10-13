//! Real-scan regression test: `test_images/golden/<name>.json` is the expected extraction of
//! `test_images/<name>.{png,jpeg,jpg}` through the same scan -> extract flow as `routes`.
//! Must stay identical unless a change is meant to alter results (then regenerate snapshots).
//! Needs local, gitignored images:
//!
//! ```sh
//! OCR_DET_MAX_SIDE=1600 cargo test --release golden -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use crate::extraction::Extractor;
use crate::scanner::{self, DocDetector};

const IMAGES_DIR: &str = "test_images";
const GOLDEN_DIR: &str = "test_images/golden";

fn image_for(name: &str) -> Option<PathBuf> {
    ["png", "jpeg", "jpg"]
        .iter()
        .map(|ext| Path::new(IMAGES_DIR).join(format!("{name}.{ext}")))
        .find(|path| path.exists())
}

#[test]
#[ignore = "needs local test_images/ and ~1 minute; see module docs"]
fn real_scans_match_golden_snapshots() {
    assert_eq!(
        std::env::var("OCR_DET_MAX_SIDE").as_deref(),
        Ok("1600"),
        "run with OCR_DET_MAX_SIDE=1600 (production value the snapshots use)"
    );
    crate::ort_config::init_runtime().expect("ONNX Runtime init");
    let detector = DocDetector::load().expect("document detector");
    let debug_dir = std::env::temp_dir().join("citizenship_golden_debug");
    std::fs::create_dir_all(&debug_dir).expect("debug dir");
    let extractor = Extractor::load(debug_dir).expect("OCR models");

    let mut snapshots: Vec<PathBuf> = std::fs::read_dir(GOLDEN_DIR)
        .expect("test_images/golden/ missing")
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    snapshots.sort();
    assert!(!snapshots.is_empty(), "no snapshots in {GOLDEN_DIR}");

    let mut failures = Vec::new();
    for snapshot in &snapshots {
        let name = snapshot.file_stem().unwrap().to_string_lossy().into_owned();
        let image_path = image_for(&name).unwrap_or_else(|| panic!("no image for snapshot {name}"));
        let side = if name.contains("back") { "back" } else { "front" };

        let original = std::fs::read(&image_path).expect("read image");
        let scan = scanner::scan_document(&detector, &original).expect("scan");
        let original_for_fallback = scan.corners_detected.then_some(original.as_slice());
        let started = std::time::Instant::now();
        let document = extractor
            .extract_document(&scan.jpeg_bytes, original_for_fallback, Some(side))
            .expect("extract");

        let actual = serde_json::to_value(&document).unwrap();
        let expected: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(snapshot).unwrap()).unwrap();
        let verdict = if actual == expected { "ok" } else { "MISMATCH" };
        println!("{verdict:8} {name} ({:.1}s)", started.elapsed().as_secs_f32());
        if actual != expected {
            failures.push(format!(
                "{name}\n  expected: {expected}\n  actual:   {actual}"
            ));
        }
    }
    assert!(failures.is_empty(), "golden mismatches:\n{}", failures.join("\n"));
}
