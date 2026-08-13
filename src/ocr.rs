use std::process::Command;
use std::sync::LazyLock;

use opencv::core::{Size, Vector};
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc};
use regex::Regex;
use serde::Serialize;

use crate::error::AppError;

/// A single label → value pair read off the document, in the order the
/// labels appear top-to-bottom. A plain `Vec` of pairs (not a `HashMap`)
/// because label text is whatever the document happens to print — nothing
/// here is a fixed schema, so there's no fixed set of keys to hang a map
/// off of, and preserving document order is more useful for display than
/// alphabetical/hash order would be.
#[derive(Serialize)]
pub struct Field {
    pub label: String,
    pub value: String,
}

/// Runs Tesseract (`eng+nep`, sparse-text mode — the ID card's watermark
/// background merges label text into unreadable blobs under the default
/// "assume a uniform block" mode; sparse mode looks for isolated text
/// regions instead, which is a much closer match for a card layout) on
/// `image_path` and pulls out every label/value pair it can find.
///
/// This makes no assumption about which document or which side of it is
/// being read: it doesn't look for "NATIONALITY" or "SEX" by name, it
/// looks for *the shape* a label takes on this kind of document (a short
/// run of Latin capitals — how every label on both sides of this card is
/// printed) and pairs each one with whatever text sits nearest below it.
/// Swap in the back of the card, or a different document with the same
/// labels-in-capitals convention, and it extracts whatever's actually
/// there instead of a fixed set of fields that may not apply.
///
/// Runs as a subprocess rather than linking libtesseract directly — avoids
/// stacking a second FFI/bindgen build dependency on top of OpenCV's; the
/// `tesseract` CLI already ships as a stable, versioned interface and the
/// process-spawn overhead (a few ms) is negligible next to OCR itself.
pub fn extract_fields(image_path: &std::path::Path) -> Result<Vec<Field>, AppError> {
    let upscaled_path = upscale_for_ocr(image_path)?;

    let output = Command::new("tesseract")
        .arg(&upscaled_path)
        .arg("stdout")
        .args(["-l", "eng+nep", "--psm", "11", "tsv"])
        .output()
        .map_err(AppError::Ocr);
    let _ = std::fs::remove_file(&upscaled_path);
    let output = output?;

    if !output.status.success() {
        return Err(AppError::OcrFailed(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }

    let tsv = String::from_utf8_lossy(&output.stdout);
    let lines = group_lines(&tsv);
    Ok(parse_fields(&lines))
}

/// 2x upscale factor fed to Tesseract. The smallest text on this card (a
/// vertically-printed ID number, Devanagari numerals) is only a handful of
/// pixels tall at the crop's native resolution — too small for Tesseract's
/// character shapes to resolve reliably. Verified experimentally: the same
/// crop upscaled 2x turned a garbled misread of that number into an exact
/// match, with no measurable loss of accuracy on text that was already
/// reading fine at native size.
const OCR_UPSCALE_FACTOR: f64 = 2.0;

/// Writes a 2x-upscaled copy of `image_path` to a temp file and returns its
/// path; the caller is responsible for deleting it once Tesseract is done.
fn upscale_for_ocr(image_path: &std::path::Path) -> Result<std::path::PathBuf, AppError> {
    let image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)
        .map_err(AppError::Processing)?;
    let size = image.size().map_err(AppError::Processing)?;

    let mut upscaled = Mat::default();
    imgproc::resize(
        &image,
        &mut upscaled,
        Size::new(
            (size.width as f64 * OCR_UPSCALE_FACTOR).round() as i32,
            (size.height as f64 * OCR_UPSCALE_FACTOR).round() as i32,
        ),
        0.0,
        0.0,
        imgproc::INTER_CUBIC,
    )
    .map_err(AppError::Processing)?;

    let out_path = std::env::temp_dir().join(format!("ocr-{}.png", uuid::Uuid::new_v4()));
    let mut buf = Vector::<u8>::new();
    imgcodecs::imencode(".png", &upscaled, &mut buf, &Vector::new()).map_err(AppError::Processing)?;
    std::fs::write(&out_path, buf.as_slice()).map_err(AppError::Io)?;

    Ok(out_path)
}

/// One OCR'd line: its words joined in reading order, plus the position
/// (top-left corner of the line's bounding box) used both to keep lines in
/// reading order and to tell which column a line belongs to.
struct Line {
    top: i32,
    left: i32,
    text: String,
}

/// Minimum word confidence (Tesseract's own 0-100 score) to trust. Low-conf
/// hits are almost always the card's decorative watermark/background
/// pattern getting misread as text, not an OCR engine being unsure about
/// real text — including them just adds noise to the label/value matching
/// below.
const MIN_WORD_CONFIDENCE: f32 = 40.0;

/// Groups Tesseract's word-level TSV rows into lines using its own
/// `block/par/line` grouping (columns 3-5) — even in sparse mode this
/// reliably keeps horizontally-aligned words (e.g. "DATE" "OF" "BIRTH")
/// together, so there's no need to re-derive line grouping from raw y
/// coordinates.
fn group_lines(tsv: &str) -> Vec<Line> {
    let mut grouped: Vec<((i32, i32, i32), (i32, i32), Vec<(i32, String)>)> = Vec::new();

    for row in tsv.lines().skip(1) {
        let cols: Vec<&str> = row.split('\t').collect();
        if cols.len() < 12 {
            continue;
        }
        let Ok(block) = cols[2].parse::<i32>() else {
            continue;
        };
        let Ok(par) = cols[3].parse::<i32>() else {
            continue;
        };
        let Ok(line_num) = cols[4].parse::<i32>() else {
            continue;
        };
        let Ok(word_num) = cols[5].parse::<i32>() else {
            continue;
        };
        let Ok(left) = cols[6].parse::<i32>() else {
            continue;
        };
        let Ok(top) = cols[7].parse::<i32>() else {
            continue;
        };
        let Ok(conf) = cols[10].parse::<f32>() else {
            continue;
        };
        let text = cols[11].trim();
        if conf < MIN_WORD_CONFIDENCE || text.is_empty() {
            continue;
        }

        let key = (block, par, line_num);
        match grouped.iter_mut().find(|(k, _, _)| *k == key) {
            Some((_, min_pos, words)) => {
                min_pos.0 = min_pos.0.min(top);
                min_pos.1 = min_pos.1.min(left);
                words.push((word_num, text.to_string()));
            }
            None => grouped.push((key, (top, left), vec![(word_num, text.to_string())])),
        }
    }

    let mut lines: Vec<Line> = grouped
        .into_iter()
        .map(|(_, (top, left), mut words)| {
            words.sort_by_key(|(word_num, _)| *word_num);
            let text = words
                .into_iter()
                .map(|(_, w)| w)
                .collect::<Vec<_>>()
                .join(" ");
            Line { top, left, text }
        })
        .collect();
    lines.sort_by_key(|l| l.top);
    lines
}

/// A field label on this card is printed as a short run of Latin capitals
/// ("NATIONALITY", "DATE OF ISSUE", "SEX", "PERMANENT ADDRESS", ...) —
/// every value, by contrast, is either title/mixed-case Latin, a number, or
/// Devanagari. That single, content-agnostic shape is what identifies a
/// label; the label *text itself* is never matched against a fixed list,
/// so whatever labels actually appear on whichever side of whichever
/// document gets read.
///
/// Matched as a substring (`find`), not anchored to the whole line
/// (`^...$`) — bilingual labels on this card (Nepali caption + English
/// caption printed on the same physical row, e.g. "नागरिकताको किसिम |
/// CITIZENSHIP TYPE") land in Tesseract's OCR as one combined line, and an
/// anchored match would reject the whole thing for containing Devanagari.
/// Extracting just the matched run doubles as cleanup: the label ends up
/// as "CITIZENSHIP TYPE", not the raw bilingual line.
static LABEL_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Z][A-Z .,'&/-]*[A-Z]").expect("valid regex"));

/// Below this, a token like "M" or "OK" would satisfy [`LABEL_PATTERN`] and
/// get mistaken for a label instead of a value — genuine labels on this
/// card are all several characters long ("SEX" is the shortest real one).
const MIN_LABEL_LEN: usize = 3;

/// Returns the label text if `text` contains one, cleaned to just the
/// matched capitals run (see [`LABEL_PATTERN`] for why that's not simply
/// the whole line).
fn label_in(text: &str) -> Option<&str> {
    LABEL_PATTERN
        .find(text)
        .map(|m| m.as_str())
        .filter(|m| m.chars().count() >= MIN_LABEL_LEN)
}

/// Rejects near-empty OCR noise (stray punctuation, a short misread
/// fragment) from being accepted as a value just because it happened to be
/// the nearest non-label line — a real value has some actual content.
/// A plain "≥2 alphanumeric characters" bar isn't tight enough on its own:
/// it let a 2-letter garbled fragment ("Ax", misread background texture)
/// through as a value on a real scan. But single/double-character values
/// are also completely legitimate — a gender letter (M/F/O), a 2-digit
/// code — so short values are still allowed when they're purely digits or
/// a single letter, just not an arbitrary short run of Latin letters
/// (which real field values essentially never are; real short words are
/// several characters even in Devanagari, where syllable-per-character
/// packs more into fewer codepoints).
fn looks_like_value(text: &str) -> bool {
    let alnum_count = text.chars().filter(|c| c.is_alphanumeric()).count();
    if alnum_count == 0 {
        return false;
    }
    let trimmed = text.trim();
    let digit_count = trimmed.chars().filter(|c| c.is_ascii_digit()).count();
    if digit_count >= 2 && trimmed.chars().all(|c| c.is_ascii_digit() || !c.is_alphanumeric()) {
        return true; // numeric codes ("01", NIN, ...) — a lone stray digit is noise, not a code
    }
    if alnum_count == 1 {
        return trimmed
            .chars()
            .find(|c| c.is_alphabetic())
            .is_some_and(|c| c.is_uppercase());
    }
    alnum_count >= 3
}

// These three are distances in the *OCR'd* image's pixel space, which is
// [`OCR_UPSCALE_FACTOR`]x the crop's real resolution since that's what
// actually gets fed to Tesseract — they need to scale with it, or an
// upscale-factor change silently shrinks the effective search windows
// (caught in testing: raising the factor made a label lose its already-
// correctly-positioned value because the gap between them, unchanged in
// real terms, now measured as more OCR pixels than the fixed threshold
// allowed).
/// How far below a label its value is expected to sit. Generous on
/// purpose — cards get scanned at a range of resolutions.
const MAX_ROW_DISTANCE: i32 = (140.0 * OCR_UPSCALE_FACTOR) as i32;
/// How far consecutive lines of an already-started, wrapped value (e.g. an
/// address spanning two printed lines) may sit from each other. Tighter
/// than [`MAX_ROW_DISTANCE`] since these are lines known to belong to the
/// same value, not a value being searched for — they're expected to sit
/// right under one another.
const MAX_CONTINUATION_GAP: i32 = (70.0 * OCR_UPSCALE_FACTOR) as i32;
/// A value can wrap onto at most this many printed lines before the search
/// gives up extending it — caps how far a bad match can run away.
const MAX_VALUE_LINES: usize = 3;
/// How far a value's left edge may drift from its label's left edge and
/// still count as "the same column". This card's layout is two-column
/// (e.g. NATIONALITY on the left, NIN on the right, at similar heights) —
/// without this, "nearest line below" regularly grabs a value from the
/// *other* column that merely happens to sit at a closer y.
const MAX_COLUMN_DRIFT: i32 = (260.0 * OCR_UPSCALE_FACTOR) as i32;

/// A value that's itself a `YYYY-MM-DD` date, embedded in a longer OCR'd
/// line — pulled out on its own when present. Not a document-specific
/// rule: any label whose nearest line happens to contain a date benefits,
/// regardless of what the label says. Restricted to ASCII digits
/// deliberately: Rust's `regex` crate is Unicode-aware by default, so a
/// plain `\d` would also match Devanagari digits (U+0966-U+096F) — this
/// card prints most dates twice, once in each script, and the point of
/// this pattern is to prefer the Latin-digit copy over its Devanagari twin
/// when both land on the same OCR'd line.
static DATE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").expect("valid regex"));

/// This card prints several fields (names, address, dates, ID numbers) in
/// both Devanagari and Latin script, side by side or stacked, as two full
/// alternate renderings rather than one embedded token. Both are real,
/// useful content — a Nepali name has no "correct" Latin spelling to fall
/// back to — so rather than picking one script and discarding the other,
/// each label gets a value from *each* script that has a plausible nearby
/// candidate: same label, two `Field` entries when both scripts are
/// present, one when only one is.
static DEVANAGARI_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\u{0900}-\u{097F}]").expect("valid regex"));

fn parse_fields(lines: &[Line]) -> Vec<Field> {
    let mut fields = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let Some(label) = label_in(&line.text) else {
            continue;
        };

        let candidates = || {
            lines[i + 1..]
                .iter()
                .take_while(|l| l.top - line.top <= MAX_ROW_DISTANCE)
                .filter(|l| label_in(&l.text).is_none() && looks_like_value(&l.text))
        };
        // Require the same column (this card's layout is multi-column, so
        // the nearest line by row alone is often a value from an unrelated
        // field — picking whatever's nearest regardless of column reliably
        // grabs the wrong one).
        let same_column =
            || candidates().filter(|l| (l.left - line.left).abs() <= MAX_COLUMN_DRIFT);

        let latin = same_column().find(|l| !DEVANAGARI_PATTERN.is_match(&l.text));
        let devanagari = same_column().find(|l| DEVANAGARI_PATTERN.is_match(&l.text));

        for value_line in [latin, devanagari].into_iter().flatten() {
            fields.push(Field {
                label: label.to_string(),
                value: collect_wrapped_value(lines, i, value_line),
            });
        }

        // A same-column match can still miss entirely: a value printed in
        // two scripts on one OCR line (e.g. a date in both Devanagari and
        // Latin digits) gets grouped with a `left` anchored to whichever
        // script is more to the left, which can legitimately fall outside
        // the label's column even though it's the right line. For that
        // specific, self-validating case — not for arbitrary text, which
        // would just as often grab the wrong neighboring field — fall back
        // to a date pattern found anywhere in the row window regardless of
        // column.
        if latin.is_none() && devanagari.is_none() {
            if let Some(date) = candidates().find_map(|l| DATE_PATTERN.find(&l.text)) {
                fields.push(Field {
                    label: label.to_string(),
                    value: date.as_str().to_string(),
                });
            }
        }
    }

    fields
}

/// Starting from `first` (the closest matching line to the label), keeps
/// pulling in following lines that look like a continuation of the same
/// value — same column, close enough below the previous line to be a
/// wrapped second row rather than the next field — up to
/// [`MAX_VALUE_LINES`]. Handles values that print across two lines (an
/// address is the case that motivated this; a long name would hit the same
/// thing) without needing to know in advance which labels tend to wrap.
fn collect_wrapped_value(lines: &[Line], label_index: usize, first: &Line) -> String {
    let mut parts = vec![first.text.as_str()];
    let mut prev = first;

    for line in &lines[label_index + 1..] {
        if parts.len() >= MAX_VALUE_LINES {
            break;
        }
        // Skip lines already consumed up to and including `first`.
        if line.top <= first.top {
            continue;
        }
        if line.top - prev.top > MAX_CONTINUATION_GAP {
            break;
        }
        if label_in(&line.text).is_some() || !looks_like_value(&line.text) {
            break;
        }
        if (line.left - first.left).abs() > MAX_COLUMN_DRIFT {
            break;
        }
        // Don't mix scripts into one value — a Devanagari line right below
        // a Latin one is virtually always the *other* rendering of the
        // same field (see the script-preference note above [`parse_fields`]),
        // not a second line of the same rendering.
        if DEVANAGARI_PATTERN.is_match(&line.text) != DEVANAGARI_PATTERN.is_match(&first.text) {
            break;
        }
        parts.push(line.text.as_str());
        prev = line;
    }

    let joined = DATE_PATTERN
        .find(first.text.as_str())
        .map(|m| m.as_str().to_string());
    match joined {
        // A wrapped-value search only makes sense for text; a date is a
        // single self-contained token, so isolate it same as elsewhere
        // rather than appending unrelated following lines to it.
        Some(date) => date,
        None => parts.join(" "),
    }
}
