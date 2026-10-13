//! Debug-only (`OCR_DEBUG_LINES`) label/value pairing of leftover OCR text. [`looks_like_label`]
//! is also used by `keywords` to skip neighbouring label boxes.

use super::layout::{Bounds, OcrLine, line_left};

/// Leftover label/value pair; logged only.
#[derive(Debug, PartialEq, Clone)]
pub(super) struct Field {
    pub(super) label: String,
    pub(super) value: String,
}

const FIELD_LABEL_MIN_CONFIDENCE: f32 = 0.90;

const FIELD_VALUE_MIN_CONFIDENCE: f32 = 0.75;

const MAX_FIELD_ALIGNMENT_DRIFT: i32 = 120;

const LABEL_WORDS: &[&str] = &[
    "GOVERNMENT", "NEPAL", "CITIZENSHIP", "CERTIFICATE", "NO", "FULL", "NAME", "SEX", "DATE", "OF",
    "BIRTH", "PLACE", "PERMANENT", "ADDRESS", "DISTRICT", "WARD", "YEAR", "MONTH", "DAY", "RM",
    "TYPE", "ISSUED", "ISSUING", "OFFICER", "AUTHORITY", "SIGNATURE", "HOLDER",
];

pub(super) fn parse_generic_fields(lines: &[OcrLine]) -> Vec<Field> {
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
                (vertical_gap_from(bounds, candidate), left_distance(bounds.left, candidate))
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

fn is_field_label(line: &OcrLine) -> bool {
    line.confidence >= FIELD_LABEL_MIN_CONFIDENCE && looks_like_label(&line.text)
}

pub(super) fn looks_like_label(text: &str) -> bool {
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

fn connected_line_groups(lines: &[OcrLine], include: impl Fn(&OcrLine) -> bool + Copy) -> Vec<Vec<usize>> {
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

fn value_group_from(lines: &[OcrLine], seed: usize, include: impl Fn(&OcrLine) -> bool) -> Vec<usize> {
    let mut group = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            (index != seed && include(line) && lines_share_value_row(&lines[seed], line)).then_some(index)
        })
        .collect::<Vec<_>>();
    group.push(seed);
    group
}

fn connected_group(lines: &[OcrLine], seed: usize, remaining: &mut Vec<usize>) -> Vec<usize> {
    let mut group = vec![seed];
    while let Some(position) = remaining
        .iter()
        .position(|candidate| group.iter().any(|member| lines_share_row(&lines[*member], &lines[*candidate])))
    {
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
    overlaps_vertically && horizontal_gap <= a_bounds.height().max(b_bounds.height()).saturating_mul(2)
}

fn lines_share_value_row(a: &OcrLine, b: &OcrLine) -> bool {
    let a_bounds = Bounds::from_line(a);
    let b_bounds = Bounds::from_line(b);
    let center_distance = (a_bounds.top + a_bounds.bottom).abs_diff(b_bounds.top + b_bounds.bottom).saturating_div(2);
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

fn group_bounds(lines: &[OcrLine], indexes: &[usize]) -> Bounds {
    indexes.iter().fold(
        Bounds { left: i32::MAX, top: i32::MAX, right: i32::MIN, bottom: i32::MIN },
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
    let capacity = indexes.iter().map(|&index| lines[index].text.len()).sum::<usize>().saturating_add(indexes.len().saturating_sub(1));
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
    (!label.is_empty() && !value.is_empty()).then(|| Field { label: label.to_owned(), value: value.to_owned() })
}
