//! Visual (printed, not MRZ) label/value field extraction for the passport
//! photo page — same generic spatial label/value engine the NID service
//! uses (`ocr.rs`), ported here rather than shared as a dependency, and
//! tuned with a `PASSPORT` document profile instead of NID/citizenship
//! ones. This is what recovers fields the MRZ physically cannot encode
//! (date of issue, place of birth, issuing authority) — see `mrz.rs` for
//! why those are absent from the machine-readable zone.

use std::cmp::Ordering;

use serde::Serialize;

const FIELD_LABEL_MIN_CONFIDENCE: f32 = 0.90;
const FIELD_VALUE_MIN_CONFIDENCE: f32 = 0.75;
const MAX_FIELD_ALIGNMENT_DRIFT: i32 = 120;

/// Label vocabulary for this passport layout's bilingual (Devanagari/
/// English) field headers — scoped tightly to passport terms (not NID's
/// list) to avoid misclassifying printed values ("NEPALESE", "SAPTARI",
/// surnames/given names) as labels.
const LABEL_WORDS: &[&str] = &[
    "TYPE",
    "COUNTRY",
    "CODE",
    "PASSPORT",
    "SURNAME",
    "GIVEN",
    "NAMES",
    "NATIONALITY",
    "DATE",
    "OF",
    "BIRTH",
    "SEX",
    "ISSUE",
    "EXPIRY",
    "CITIZENSHIP",
    "NO",
    "PLACE",
    "ISSUING",
    "AUTHORITY",
    "HOLDERS",
    "SIGNATURE",
];

#[derive(Debug, PartialEq)]
pub struct OcrLine {
    pub text: String,
    pub confidence: f32,
    pub polygon: [[i32; 2]; 4],
}

#[derive(Debug, Serialize, PartialEq, Clone)]
pub struct Field {
    pub label: String,
    pub value: String,
}

const MONTH_ABBREVIATIONS: &[&str] = &[
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
];

/// Lines shaped like `18 FEB 1992` — this passport template prints exactly
/// three such dates (birth, issue, expiry) in that top-to-bottom order.
/// Used as a positional fallback when a date's own label garbles badly
/// enough that keyword matching can't find it (seen in practice: bilingual
/// header text recognizes far less reliably than the plain-value lines
/// below it) — most valuable for date of issue, since the MRZ can't
/// provide that one at all as a fallback of its own.
pub fn date_shaped_values(lines: &[OcrLine]) -> Vec<String> {
    lines
        .iter()
        .filter(|line| is_dd_mon_yyyy(&line.text))
        .map(|line| line.text.trim().to_owned())
        .collect()
}

fn is_dd_mon_yyyy(text: &str) -> bool {
    dd_mon_yyyy_parts(text).is_some()
}

/// Day must lead, year must trail, and a recognized month abbreviation
/// must appear somewhere between them — but anything else between day and
/// year is ignored rather than rejecting the whole line. Needed for
/// bilingual foreign passports, where the printed date is e.g. a
/// non-Latin month name alongside its English abbreviation
/// (`05 <script>/JUL 23`, split on '/' same as whitespace): the non-Latin
/// script routinely OCRs as garbage through this Devanagari-vocab
/// recognizer, but it was never data worth keeping anyway — only the
/// English abbreviation is.
fn dd_mon_yyyy_parts(text: &str) -> Option<(u32, u32, i32)> {
    let tokens: Vec<&str> = text
        .split(|c: char| c.is_whitespace() || c == '/')
        .filter(|token| !token.is_empty())
        .collect();
    let (first, rest) = tokens.split_first()?;
    let (last, middle) = rest.split_last()?;
    if middle.is_empty() {
        return None;
    }
    let day_ok = (1..=2).contains(&first.len()) && first.chars().all(|c| c.is_ascii_digit());
    let year_ok = (last.len() == 2 || last.len() == 4) && last.chars().all(|c| c.is_ascii_digit());
    if !day_ok || !year_ok {
        return None;
    }
    let month = middle
        .iter()
        .find_map(|token| MONTH_ABBREVIATIONS.iter().position(|m| *m == token.to_uppercase()))?
        as u32
        + 1;
    let day: u32 = first.parse().ok()?;
    let year: i32 = if last.len() == 2 {
        // Unlike an MRZ birth date, a printed date of issue/expiry on a
        // passport being scanned today is never plausibly pre-2000 — no
        // century-ambiguity handling needed, always the current century.
        2000 + last.parse::<i32>().ok()?
    } else {
        last.parse().ok()?
    };
    Some((day, month, year))
}

/// Converts an OCR-read `18 FEB 1992`-shaped value to ISO 8601
/// (`1992-02-18`), matching the format the MRZ-derived dates already use —
/// same document, one consistent date shape throughout the response.
/// Returns the original text unchanged if it isn't actually date-shaped.
pub fn dd_mon_yyyy_to_iso(text: &str) -> String {
    match dd_mon_yyyy_parts(text) {
        Some((day, month, year)) => format!("{year:04}-{month:02}-{day:02}"),
        None => text.to_owned(),
    }
}

pub fn reading_order(a: &OcrLine, b: &OcrLine) -> Ordering {
    let a_top = a.polygon.iter().map(|point| point[1]).min().unwrap_or(0);
    let b_top = b.polygon.iter().map(|point| point[1]).min().unwrap_or(0);
    let a_left = a.polygon.iter().map(|point| point[0]).min().unwrap_or(0);
    let b_left = b.polygon.iter().map(|point| point[0]).min().unwrap_or(0);
    a_top.cmp(&b_top).then_with(|| a_left.cmp(&b_left))
}

struct FieldSignature {
    output_label: &'static str,
    english_keywords: &'static [&'static str],
}

const PASSPORT_FIELDS: &[FieldSignature] = &[
    FieldSignature { output_label: "TYPE", english_keywords: &["TYPE"] },
    FieldSignature { output_label: "COUNTRY CODE", english_keywords: &["COUNTRY CODE"] },
    FieldSignature { output_label: "PASSPORT NUMBER", english_keywords: &["PASSPORT NO"] },
    FieldSignature { output_label: "SURNAME", english_keywords: &["SURNAME"] },
    FieldSignature { output_label: "GIVEN NAMES", english_keywords: &["GIVEN NAMES"] },
    FieldSignature { output_label: "NATIONALITY", english_keywords: &["NATIONALITY"] },
    FieldSignature { output_label: "DATE OF BIRTH", english_keywords: &["DATE OF BIRTH"] },
    FieldSignature { output_label: "SEX", english_keywords: &["SEX"] },
    FieldSignature { output_label: "DATE OF ISSUE", english_keywords: &["DATE OF ISSUE"] },
    FieldSignature { output_label: "DATE OF EXPIRY", english_keywords: &["DATE OF EXPIRY"] },
    FieldSignature { output_label: "CITIZENSHIP NUMBER", english_keywords: &["CITIZENSHIP NO"] },
    FieldSignature { output_label: "PLACE OF BIRTH", english_keywords: &["PLACE OF BIRTH"] },
    FieldSignature { output_label: "ISSUING AUTHORITY", english_keywords: &["ISSUING AUTHORITY"] },
];

const DETECT_KEYWORDS: &[&str] = &[
    "PASSPORT",
    "SURNAME",
    "GIVEN NAMES",
    "NATIONALITY",
    "DATE OF BIRTH",
    "राहदानी",
];
const MIN_DETECT_MATCHES: usize = 2;

pub fn parse_fields(lines: &[OcrLine]) -> Vec<Field> {
    let generic = parse_generic_fields(lines);
    let matches = DETECT_KEYWORDS
        .iter()
        .filter(|keyword| has_line_ci(lines, keyword))
        .count();
    if matches < MIN_DETECT_MATCHES {
        return generic;
    }

    let mut fields = Vec::with_capacity(PASSPORT_FIELDS.len());
    for signature in PASSPORT_FIELDS {
        for keyword in signature.english_keywords {
            // Passport values are printed as a single OCR line, unlike
            // NID's occasional value-split-across-two-boxes layout — so
            // prefer the single-nearest-line lookup first here. The
            // generic parser's same-row merge (built for that NID case)
            // would otherwise glue nearby background/microprint noise onto
            // the value on this document's busier photo page.
            if let Some(value) = relaxed_field_value(lines, &[keyword])
                .or_else(|| field_value(&generic, keyword).map(str::to_owned))
            {
                push_owned_field(&mut fields, signature.output_label, Some(value));
                break;
            }
        }
    }
    fields
}

fn parse_generic_fields(lines: &[OcrLine]) -> Vec<Field> {
    let mut fields = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if let Some((label, value)) = line.text.split_once(':')
            && let Some(field) = make_field(label, value)
        {
            fields.push((index, field));
        }
    }

    for mut label_group in connected_line_groups(lines, is_field_label) {
        label_group.sort_unstable_by_key(|&index| line_left(&lines[index]));
        let bounds = group_bounds(lines, &label_group);
        let Some(anchor) = lines
            .iter()
            .enumerate()
            .filter(|(index, candidate)| {
                !label_group.contains(index)
                    && candidate.confidence >= FIELD_VALUE_MIN_CONFIDENCE
                    && !is_field_label(candidate)
            })
            .filter(|(_, candidate)| {
                is_below(bounds, candidate)
                    && vertical_gap_from(bounds, candidate) <= bounds.height().saturating_mul(4)
                    && left_distance(bounds.left, candidate) <= MAX_FIELD_ALIGNMENT_DRIFT
            })
            .min_by_key(|(_, candidate)| {
                (
                    vertical_gap_from(bounds, candidate),
                    left_distance(bounds.left, candidate),
                )
            })
            .map(|(index, _)| index)
        else {
            continue;
        };

        let mut value_group = value_group_from(lines, anchor, |line| {
            line.confidence >= FIELD_VALUE_MIN_CONFIDENCE && !is_field_label(line)
        });
        value_group.sort_unstable_by_key(|&index| line_left(&lines[index]));

        let label = join_line_text(lines, &label_group);
        let value = join_line_text(lines, &value_group);
        if let Some(field) = make_field(&label, &value) {
            fields.push((*label_group.iter().min().unwrap_or(&anchor), field));
        }
    }
    fields.sort_unstable_by_key(|(index, _)| *index);
    fields.into_iter().map(|(_, field)| field).collect()
}

fn has_line_ci(lines: &[OcrLine], needle: &str) -> bool {
    let needle = needle.to_uppercase();
    lines
        .iter()
        .any(|line| line.text.to_uppercase().contains(&needle))
}

fn relaxed_field_value(lines: &[OcrLine], keywords: &[&str]) -> Option<String> {
    let label_index = lines.iter().position(|line| {
        let upper = line.text.to_uppercase();
        keywords.iter().any(|keyword| upper.contains(keyword))
    })?;
    let label_line = &lines[label_index];

    if let Some((_, value)) = label_line.text.split_once(':') {
        let value = value.trim();
        if !value.is_empty() {
            return Some(value.to_owned());
        }
    }

    let bounds = Bounds::from_line(label_line);
    lines
        .iter()
        .enumerate()
        .filter(|(index, line)| {
            *index != label_index
                && is_below(bounds, line)
                && !looks_like_label(&line.text)
                && left_distance(bounds.left, line) <= MAX_FIELD_ALIGNMENT_DRIFT * 2
        })
        .min_by_key(|(_, line)| vertical_gap_from(bounds, line))
        .map(|(_, line)| line.text.trim().to_owned())
}

fn field_value<'a>(fields: &'a [Field], label: &str) -> Option<&'a str> {
    fields
        .iter()
        .find(|field| field.label.contains(label))
        .map(|field| field.value.as_str())
}

fn push_borrowed_field(fields: &mut Vec<Field>, label: &str, value: Option<&str>) {
    if let Some(value) = value
        && let Some(field) = make_field(label, value)
    {
        fields.push(field);
    }
}

fn push_owned_field(fields: &mut Vec<Field>, label: &str, value: Option<String>) {
    if let Some(value) = value {
        push_borrowed_field(fields, label, Some(&value));
    }
}

fn is_field_label(line: &OcrLine) -> bool {
    line.confidence >= FIELD_LABEL_MIN_CONFIDENCE && looks_like_label(&line.text)
}

fn looks_like_label(text: &str) -> bool {
    let mut saw_label_word = false;
    for word in text.split_whitespace() {
        let cleaned: String = word.chars().filter(char::is_ascii_alphabetic).collect();
        if cleaned.is_empty() {
            continue;
        }
        if cleaned.chars().any(|character| character.is_ascii_lowercase()) {
            return false;
        }
        if cleaned.len() == 1 {
            continue;
        }
        if !LABEL_WORDS.contains(&cleaned.as_str()) {
            return false;
        }
        saw_label_word = true;
    }
    saw_label_word
}

fn connected_line_groups(
    lines: &[OcrLine],
    include: impl Fn(&OcrLine) -> bool + Copy,
) -> Vec<Vec<usize>> {
    let mut remaining = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| include(line).then_some(index))
        .collect::<Vec<_>>();
    let mut groups = Vec::new();
    while let Some(seed) = remaining.pop() {
        let group = connected_group(lines, seed, &mut remaining);
        groups.push(group);
    }
    groups
}

fn value_group_from(
    lines: &[OcrLine],
    seed: usize,
    include: impl Fn(&OcrLine) -> bool,
) -> Vec<usize> {
    let mut group = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            (index != seed && include(line) && lines_share_value_row(&lines[seed], line))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    group.push(seed);
    group
}

fn connected_group(lines: &[OcrLine], seed: usize, remaining: &mut Vec<usize>) -> Vec<usize> {
    let mut group = vec![seed];
    while let Some(position) = remaining.iter().position(|candidate| {
        group
            .iter()
            .any(|member| lines_share_row(&lines[*member], &lines[*candidate]))
    }) {
        group.push(remaining.swap_remove(position));
    }
    group
}

fn lines_share_row(a: &OcrLine, b: &OcrLine) -> bool {
    let a_bounds = Bounds::from_line(a);
    let b_bounds = Bounds::from_line(b);
    let overlaps_vertically = a_bounds.top < b_bounds.bottom && b_bounds.top < a_bounds.bottom;
    let horizontal_gap = if a_bounds.right < b_bounds.left {
        b_bounds.left - a_bounds.right
    } else if b_bounds.right < a_bounds.left {
        a_bounds.left - b_bounds.right
    } else {
        0
    };
    overlaps_vertically
        && horizontal_gap <= a_bounds.height().max(b_bounds.height()).saturating_mul(2)
}

fn lines_share_value_row(a: &OcrLine, b: &OcrLine) -> bool {
    let a_bounds = Bounds::from_line(a);
    let b_bounds = Bounds::from_line(b);
    let center_distance = (a_bounds.top + a_bounds.bottom)
        .abs_diff(b_bounds.top + b_bounds.bottom)
        .saturating_div(2);
    let max_height = a_bounds.height().max(b_bounds.height());
    let horizontal_gap = if a_bounds.right < b_bounds.left {
        b_bounds.left - a_bounds.right
    } else if b_bounds.right < a_bounds.left {
        a_bounds.left - b_bounds.right
    } else {
        0
    };
    center_distance <= max_height.saturating_div(2) as u32 && horizontal_gap <= max_height
}

#[derive(Clone, Copy)]
struct Bounds {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

impl Bounds {
    fn from_line(line: &OcrLine) -> Self {
        Self {
            left: line_left(line),
            top: line.polygon.iter().map(|point| point[1]).min().unwrap_or(0),
            right: line.polygon.iter().map(|point| point[0]).max().unwrap_or(0),
            bottom: line.polygon.iter().map(|point| point[1]).max().unwrap_or(0),
        }
    }

    fn height(self) -> i32 {
        self.bottom.saturating_sub(self.top).max(1)
    }
}

fn group_bounds(lines: &[OcrLine], indexes: &[usize]) -> Bounds {
    indexes.iter().fold(
        Bounds {
            left: i32::MAX,
            top: i32::MAX,
            right: i32::MIN,
            bottom: i32::MIN,
        },
        |bounds, &index| {
            let line = Bounds::from_line(&lines[index]);
            Bounds {
                left: bounds.left.min(line.left),
                top: bounds.top.min(line.top),
                right: bounds.right.max(line.right),
                bottom: bounds.bottom.max(line.bottom),
            }
        },
    )
}

fn line_left(line: &OcrLine) -> i32 {
    line.polygon.iter().map(|point| point[0]).min().unwrap_or(0)
}

fn vertical_gap_from(label: Bounds, value: &OcrLine) -> i32 {
    let value_top = value.polygon.iter().map(|point| point[1]).min().unwrap_or(0);
    value_top.saturating_sub(label.bottom)
}

fn is_below(label: Bounds, value: &OcrLine) -> bool {
    let value_top = value.polygon.iter().map(|point| point[1]).min().unwrap_or(0);
    value_top >= label.bottom.saturating_sub(label.height().saturating_div(2))
}

fn left_distance(label_left: i32, value: &OcrLine) -> i32 {
    label_left.abs_diff(line_left(value)).try_into().unwrap_or(i32::MAX)
}

fn join_line_text(lines: &[OcrLine], indexes: &[usize]) -> String {
    let capacity = indexes
        .iter()
        .map(|&index| lines[index].text.len())
        .sum::<usize>()
        .saturating_add(indexes.len().saturating_sub(1));
    let mut text = String::with_capacity(capacity);
    for &index in indexes {
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(&lines[index].text);
    }
    text
}

fn make_field(label: &str, value: &str) -> Option<Field> {
    let label = label.trim().trim_matches('*').trim();
    let value = value.trim().trim_matches('*').trim();
    (!label.is_empty() && !value.is_empty()).then(|| Field {
        label: label.to_owned(),
        value: value.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, confidence: f32, left: i32, top: i32) -> OcrLine {
        line_with_width(text, confidence, left, top, 100)
    }

    fn line_with_width(text: &str, confidence: f32, left: i32, top: i32, width: i32) -> OcrLine {
        OcrLine {
            text: text.to_owned(),
            confidence,
            polygon: [
                [left, top],
                [left + width, top],
                [left + width, top + 20],
                [left, top + 20],
            ],
        }
    }

    #[test]
    fn extracts_passport_visual_fields_from_a_two_column_layout() {
        let lines = vec![
            line_with_width("PASSPORT", 0.99, 10, 10, 200),
            line("TYPE", 0.99, 10, 60),
            line("COUNTRY CODE", 0.99, 500, 60),
            line("P", 0.99, 12, 90),
            line("NPL", 0.99, 500, 90),
            line("PASSPORT NO", 0.99, 500, 130),
            line("05516586", 0.99, 500, 160),
            line("SURNAME", 0.99, 10, 130),
            line("YADAV", 0.99, 12, 160),
            line("GIVEN NAMES", 0.99, 10, 200),
            line_with_width("RAM KUMAR", 0.99, 12, 230, 150),
            line("NATIONALITY", 0.99, 10, 270),
            line("NEPALESE", 0.99, 12, 300),
            line("DATE OF BIRTH", 0.99, 10, 340),
            line("18 FEB 1992", 0.99, 12, 370),
            line("DATE OF ISSUE", 0.99, 10, 480),
            line("06 APR 2011", 0.99, 12, 510),
            line("DATE OF EXPIRY", 0.99, 10, 550),
            line("05 APR 2021", 0.99, 12, 580),
            line("PLACE OF BIRTH", 0.99, 500, 200),
            line("SAPTARI", 0.99, 500, 230),
            line("CITIZENSHIP NO", 0.99, 500, 270),
            line("161074-77", 0.99, 500, 300),
            line("ISSUING AUTHORITY", 0.99, 500, 340),
            line_with_width("MOFA CENTRAL PASSPORT OFFICE", 0.99, 500, 370, 300),
        ];

        let fields = parse_fields(&lines);
        let get = |label: &str| {
            fields
                .iter()
                .find(|field| field.label == label)
                .map(|field| field.value.as_str())
        };

        assert_eq!(get("SURNAME"), Some("YADAV"));
        assert_eq!(get("GIVEN NAMES"), Some("RAM KUMAR"));
        assert_eq!(get("NATIONALITY"), Some("NEPALESE"));
        assert_eq!(get("DATE OF BIRTH"), Some("18 FEB 1992"));
        assert_eq!(get("DATE OF ISSUE"), Some("06 APR 2011"));
        assert_eq!(get("DATE OF EXPIRY"), Some("05 APR 2021"));
        assert_eq!(get("PLACE OF BIRTH"), Some("SAPTARI"));
        assert_eq!(get("CITIZENSHIP NUMBER"), Some("161074-77"));
        assert_eq!(get("PASSPORT NUMBER"), Some("05516586"));
    }

    #[test]
    fn converts_printed_dates_to_iso_matching_the_mrz_dates() {
        assert_eq!(dd_mon_yyyy_to_iso("06 APR 2011"), "2011-04-06");
        assert_eq!(dd_mon_yyyy_to_iso("18 FEB 1992"), "1992-02-18");
        // Not date-shaped: returned unchanged rather than mangled.
        assert_eq!(dd_mon_yyyy_to_iso("SAPTARI"), "SAPTARI");
    }

    #[test]
    fn ignores_a_non_latin_month_name_misread_as_garbage() {
        // Real case: a foreign passport prints day / non-Latin-script month
        // name / English month abbreviation / 2-digit year. The Devanagari
        // recognizer garbles the non-Latin script into noise ("AMn") — it
        // was never usable data, only the "JUL" abbreviation matters.
        assert_eq!(dd_mon_yyyy_to_iso("05 AMn/JUL 23"), "2023-07-05");
    }

    #[test]
    fn finds_all_three_dates_in_reading_order() {
        let lines = vec![
            line("18 FEB 1992", 0.99, 12, 370),
            line("06 APR 2011", 0.99, 12, 510),
            line("05 APR 2021", 0.99, 12, 580),
        ];
        assert_eq!(
            date_shaped_values(&lines),
            vec!["18 FEB 1992", "06 APR 2011", "05 APR 2021"]
        );
    }
}
