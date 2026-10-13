//! Raw value -> schema form: gender, ward numbers, Devanagari digits, AD/BS dates.

use super::keywords::{contains_keyword, value_after_keyword};
use super::layout::OcrLine;

const MONTH_ABBREVIATIONS: &[&str] = &[
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
];

/// "Male"/"Female"/"Other" from either script; anything else (incl. misreads) -> `None`.
pub(super) fn normalize_gender(raw: &str) -> Option<String> {
    match raw.trim() {
        "पुरुष" | "पुरूष" | "Male" | "male" | "M" | "m" => Some("Male".to_owned()),
        "महिला" | "Female" | "female" | "F" | "f" => Some("Female".to_owned()),
        "अन्य" | "Other" | "other" => Some("Other".to_owned()),
        _ => None,
    }
}

pub(super) fn combine_ad_date(year: Option<String>, month: Option<String>, day: Option<String>) -> Option<String> {
    let year: i32 = year?.trim().parse().ok()?;
    let month_upper = month?.trim().to_uppercase();
    let month_num = MONTH_ABBREVIATIONS.iter().position(|m| *m == month_upper)? as u32 + 1;
    let day: u32 = day?.trim().parse().ok()?;
    if !(1..=31).contains(&day) {
        return None;
    }
    Some(format!("{year:04}/{month_num:02}/{day:02}"))
}

pub(super) fn devanagari_digits_to_ascii(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '०' => '0',
            '१' => '1',
            '२' => '2',
            '३' => '3',
            '४' => '4',
            '५' => '5',
            '६' => '6',
            '७' => '7',
            '८' => '8',
            '९' => '9',
            other => other,
        })
        .collect()
}

/// BS birth date by position: the साल row holds exactly three numbers, Y-M-D. Requires exactly
/// three and a 4-digit year, so a stray neighbouring number is rejected, not reordered.
pub(super) fn bs_date_from_row(lines: &[OcrLine]) -> Option<String> {
    let row = lines.iter().find(|line| contains_keyword(&line.text, "साल"))?;
    let converted = devanagari_digits_to_ascii(&row.text);
    let numbers: Vec<&str> = converted
        .split(|c: char| !c.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .collect();
    let [year, month, day] = numbers.as_slice() else { return None };
    if year.len() != 4 {
        return None;
    }
    bs_date(year, month, day)
}

/// `YYYY/MM/DD` BS date; rejects impossible months/days (a misread "90" month looks valid otherwise).
pub(super) fn bs_date(year: &str, month: &str, day: &str) -> Option<String> {
    let month_num: u32 = month.parse().ok()?;
    let day_num: u32 = day.parse().ok()?;
    if !(1..=12).contains(&month_num) || !(1..=32).contains(&day_num) {
        return None;
    }
    Some(format!("{year}/{month_num:02}/{day_num:02}"))
}

/// Exactly 8 digits -> `YYYY/MM/DD`, ignoring separators (OCR mixes "-" and ".").
pub(super) fn parse_bs_date(text: &str) -> Option<String> {
    let digits: String = devanagari_digits_to_ascii(text).chars().filter(char::is_ascii_digit).collect();
    if digits.len() != 8 {
        return None;
    }
    Some(format!("{}/{}/{}", &digits[0..4], &digits[4..6], &digits[6..8]))
}

/// 1-2 ASCII digits.
fn is_plausible_ward_number(text: &str) -> bool {
    !text.is_empty() && text.len() <= 2 && text.chars().all(|c| c.is_ascii_digit())
}

/// Ward digits from the text after "वडा"/"Ward" (which may still carry "नं."/"No.").
pub(super) fn ward_number(value: &str) -> Option<String> {
    // Strip "No" before the lookalike mapping, or its "o" becomes 0 ("Ward No.2" -> "02").
    let value = devanagari_digits_to_ascii(value);
    let value = value.replace("No", "").replace("no", "").replace("नं", "");

    // Must already look like a short token before mapping; else "Bard(iya)" -> "iya" -> ward "1".
    let core: String = value.chars().filter(|c| c.is_alphanumeric()).collect();
    if core.is_empty() || core.chars().count() > 2 {
        return None;
    }

    // Glyph lookalikes (l -> 1, S -> 5, ...): safe only because a ward is digits-only.
    let digits: String = core
        .chars()
        .map(|c| match c {
            'l' | 'I' | 'i' | '|' => '1',
            'O' | 'o' | 'D' => '0',
            'S' | 's' => '5',
            'Z' | 'z' => '2',
            'B' => '8',
            'G' => '6',
            other => other,
        })
        .filter(char::is_ascii_digit)
        .collect();
    is_plausible_ward_number(&digits).then_some(digits)
}

/// [`value_after_keyword`] reduced to its ASCII digits (back page `Year:`/`Day:`).
pub(super) fn number_after_keyword(lines: &[OcrLine], keyword: &str) -> Option<String> {
    let value = value_after_keyword(lines, keyword, &[])?;
    let digits: String = value.chars().filter(char::is_ascii_digit).collect();
    (!digits.is_empty()).then_some(digits)
}

/// Digits (Devanagari or ASCII -> ASCII) right after `keyword`, e.g. `साल २०६०`.
pub(super) fn devanagari_number_after(lines: &[OcrLine], keyword: &str) -> Option<String> {
    for line in lines {
        let converted = devanagari_digits_to_ascii(&line.text);
        let Some(pos) = converted.find(keyword) else { continue };
        let after = &converted[pos + keyword.len()..];
        let digits: String = after.chars().skip_while(|c| !c.is_ascii_digit()).take_while(char::is_ascii_digit).collect();
        if !digits.is_empty() {
            return Some(digits);
        }
    }
    None
}
