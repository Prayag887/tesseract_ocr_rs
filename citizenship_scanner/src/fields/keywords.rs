//! Fuzzy label search in OCR text, label -> value lookup, and the watermark/boilerplate filter.

use super::generic::looks_like_label;
use super::layout::{Bounds, OcrLine};

/// Levenshtein distance over chars (not bytes — Devanagari glyphs are multi-byte).
fn edit_distance(a: &[char], b: &[char]) -> usize {
    let (n, m) = (a.len(), b.len());
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// Extra window room for stray spaces inside a word (free, see `find_keyword`), e.g. "Citize ns hip".
const WHITESPACE_SLACK: usize = 6;

/// Finds `keyword` in `text` within `max_distance` edits, ignoring whitespace on both sides.
/// Returns the matched byte range.
pub(super) fn find_keyword(text: &str, keyword: &str, max_distance: usize) -> Option<(usize, usize)> {
    let keyword_chars: Vec<char> = keyword.chars().filter(|c| !c.is_whitespace()).collect();
    let text_chars: Vec<char> = text.chars().collect();
    let klen = keyword_chars.len();
    if klen == 0 {
        return None;
    }
    let min_window = klen.saturating_sub(max_distance).max(1);
    let max_window = klen + max_distance + WHITESPACE_SLACK;

    // Ties go to the window closest to the keyword's length, so a substitution ("R.M" ~ "R.N")
    // beats a deletion ("R.") that would leave "N." behind as the value.
    let mut best: Option<(usize, usize, usize)> = None;
    for start in 0..text_chars.len() {
        for window in min_window..=max_window {
            let end = start + window;
            if end > text_chars.len() {
                break;
            }
            let stripped: Vec<char> =
                text_chars[start..end].iter().filter(|c| !c.is_whitespace()).copied().collect();
            let distance = edit_distance(&stripped, &keyword_chars);
            if distance > max_distance {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, _, best_distance)) if distance < best_distance => true,
                Some((best_start, best_end, best_distance)) if distance == best_distance => {
                    let best_window = best_end - best_start;
                    window.abs_diff(klen) < best_window.abs_diff(klen)
                }
                _ => false,
            };
            if better {
                best = Some((start, end, distance));
            }
        }
    }

    best.map(|(start, end, _)| {
        let byte_start: usize = text_chars[..start].iter().map(|c| c.len_utf8()).sum();
        let byte_end: usize = text_chars[..end].iter().map(|c| c.len_utf8()).sum();
        (byte_start, byte_end)
    })
}

/// `find_keyword` with distance by length: exact under 4 chars, else 1 edit.
/// (At 1 edit, "पति" matches inside "प्रमाणपत्र" on every card.)
pub(super) fn contains_keyword(text: &str, keyword: &str) -> bool {
    let max_distance = if keyword.chars().count() < 4 { 0 } else { 1 };
    find_keyword(text, keyword, max_distance).is_some()
}

/// One tolerance level of `value_after_keyword`.
fn value_after_keyword_pass(
    lines: &[OcrLine],
    keyword: &str,
    stop_keywords: &[&str],
    max_distance: usize,
) -> Option<String> {
    for (index, line) in lines.iter().enumerate() {
        let text = &line.text;
        // Boilerplate lines are never labels; skip before fuzzy matching can mistake one for a label.
        if looks_like_noise(text) {
            continue;
        }
        let Some((_start, kw_end)) = find_keyword(text, keyword, max_distance) else { continue };
        let after_kw = &text[kw_end..];
        let stop_pos = stop_keywords.iter().filter_map(|kw| after_kw.find(kw)).min().unwrap_or(after_kw.len());
        let scope = &after_kw[..stop_pos];
        let value = match scope.find(':') {
            Some(colon_pos) => &scope[colon_pos + 1..],
            None => scope,
        };
        let value = value.trim().trim_matches(|c: char| c == '.' || c == '*' || c.is_whitespace());
        if !value.is_empty() && !looks_like_noise(value) {
            return Some(value.to_owned());
        }
        if let Some(value) = nearest_value_near(lines, index, stop_keywords) {
            return Some(value);
        }
    }
    None
}

/// Value after `keyword`: text after its own colon, cut at the first `stop_keywords`; if the
/// line has no value, the nearest box right/below (labels and values are often separate boxes).
/// Exact matches are tried on every line before 1-edit matches on any line, so e.g. "Day"
/// can't fuzzy-match "Dat(e of Birth)" ahead of the real "Day:" row.
pub(super) fn value_after_keyword(lines: &[OcrLine], keyword: &str, stop_keywords: &[&str]) -> Option<String> {
    // Don't raise to distance 2: tried, more wrong-block matches than fixes on real scans.
    (0..=1).find_map(|max_distance| value_after_keyword_pass(lines, keyword, stop_keywords, max_distance))
}

/// Watermark/boilerplate fragments that never appear in a real value. Matched against the value
/// left after the keyword is consumed, hence only the tail of the district-office footer.
const NOISE_PHRASES: &[&str] = &[
    "नागरिकताको प्रमाण",
    "नागरिकता ऐन",
    "नेपाली नागरिकता",
    "नेपाल सरकार",
    "गृह मन्त्रालय",
    "जिल्ला प्रशासन",
    "कार्यालयमा वा प्रहरी कार्यालयमा",
    "बुझाईदिनुहोला",
    // English preamble: close enough to "Citizenship Certificate No" to fuzzy-match it.
    "issued this",
    "following details",
];

pub(super) fn looks_like_noise(text: &str) -> bool {
    NOISE_PHRASES.iter().any(|phrase| text.contains(phrase))
}

/// Nearest non-label box holding a label's value: same row to the right first, else the row below.
fn nearest_value_near(lines: &[OcrLine], label_index: usize, stop_keywords: &[&str]) -> Option<String> {
    let label_bounds = Bounds::from_line(&lines[label_index]);
    let mut best_right: Option<(i32, usize)> = None;
    let mut best_below: Option<(i32, usize)> = None;
    let trace = std::env::var("OCR_DEBUG_LINES").is_ok();
    if trace {
        eprintln!(
            "[nearest_value_near] label={:?} bounds=(l{},t{},r{},b{})",
            lines[label_index].text, label_bounds.left, label_bounds.top, label_bounds.right, label_bounds.bottom
        );
    }

    for (index, line) in lines.iter().enumerate() {
        if index == label_index || looks_like_label(&line.text) {
            continue;
        }
        if stop_keywords.iter().any(|kw| line.text.contains(kw)) {
            continue;
        }
        let bounds = Bounds::from_line(line);
        // Same row = overlap over half the label's height; any overlap lets a header row win.
        let overlap = bounds.bottom.min(label_bounds.bottom) - bounds.top.max(label_bounds.top);
        let same_row = overlap > label_bounds.height() / 2;
        if trace {
            eprintln!(
                "  candidate={:?} bounds=(l{},t{},r{},b{}) same_row={}",
                line.text, bounds.left, bounds.top, bounds.right, bounds.bottom, same_row
            );
        }
        // Only needs to start right of the label's left edge — padded boxes overlap a little.
        if same_row && bounds.left > label_bounds.left {
            let distance = bounds.left - label_bounds.left;
            if best_right.is_none_or(|(best, _)| distance < best) {
                best_right = Some((distance, index));
            }
        } else if !same_row
            && bounds.top >= label_bounds.bottom.saturating_sub(label_bounds.height().saturating_div(2))
        {
            let distance = bounds.top - label_bounds.bottom;
            if best_below.is_none_or(|(best, _)| distance < best) {
                best_below = Some((distance, index));
            }
        }
    }

    let chosen = best_right.or(best_below)?.1;
    let value = lines[chosen].text.trim().trim_matches(|c: char| c == '.' || c == '*' || c.is_whitespace());
    (!value.is_empty() && !looks_like_noise(value)).then(|| value.to_owned())
}
