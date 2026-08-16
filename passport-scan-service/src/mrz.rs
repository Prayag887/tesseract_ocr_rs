//! ICAO 9303 TD3 (passport) machine-readable zone: two 44-char lines,
//! fixed-column fields + mod-10 weighted checksums. This gives the
//! checksum-verified core of a passport's identity data — but the MRZ
//! physically cannot encode some fields a passport prints (date of issue,
//! place of birth, issuing authority: TD3 has no columns for them at all,
//! by ICAO spec, not an extraction gap). `local_ocr.rs` fills those in from
//! the printed page (`fields.rs`) and merges everything into one flat
//! `PassportDocument`, so this module's own result type stays private.

use crate::error::AppError;

const LINE_LEN: usize = 44;
/// ICAO 9303 checksum weights, repeating every 3 characters.
const WEIGHTS: [u32; 3] = [7, 3, 1];

/// Everything the MRZ itself encodes, checksum-verified. Not the public
/// API type — `local_ocr.rs` folds this into `PassportDocument` alongside
/// fields the MRZ can't carry (date of issue, full name as printed).
#[derive(Debug, PartialEq)]
pub struct MrzFields {
    pub document_type: String,
    pub issuing_country: String,
    pub surname: String,
    pub given_names: String,
    pub passport_number: String,
    pub nationality: String,
    /// ISO 8601 (YYYY-MM-DD), century resolved relative to today.
    pub date_of_birth: String,
    pub sex: String,
    /// ISO 8601 (YYYY-MM-DD), century resolved relative to today.
    pub date_of_expiry: String,
    pub personal_number: String,
    pub expired: bool,
    pub passport_number_check_ok: bool,
    pub date_of_birth_check_ok: bool,
    pub date_of_expiry_check_ok: bool,
    pub personal_number_check_ok: bool,
    pub composite_check_ok: bool,
}

fn check_digit(field: &str) -> u32 {
    field
        .chars()
        .enumerate()
        .map(|(i, c)| char_value(c) * WEIGHTS[i % 3])
        .sum::<u32>()
        % 10
}

fn char_value(c: char) -> u32 {
    match c {
        '0'..='9' => c as u32 - '0' as u32,
        'A'..='Z' => c as u32 - 'A' as u32 + 10,
        _ => 0, // '<' and anything unrecognized
    }
}

/// `expected` is the character actually printed in the check-digit
/// position; '<' there (a blank/unset field, e.g. no personal number) is
/// always treated as valid regardless of the computed digit — ICAO 9303
/// permits omitting the check digit when the field itself is blank.
fn verify(field: &str, expected: char) -> bool {
    if expected == '<' {
        return true;
    }
    expected.to_digit(10) == Some(check_digit(field))
}

/// ICAO 9303: a name field's real components are single-`<`-separated; the
/// field terminates at the first double `<` (or end of string), and
/// everything after that is `<` padding, not content. Stopping there —
/// rather than treating every `<` as equivalent spacing — matters because
/// the padding tail is exactly where OCR most often misreads mangled
/// filler characters as stray letters (seen in practice: a name field's
/// padding recognized as noise like `<<E<SS<<<<<<<<<<<<S<SKE`, which a
/// naive "every `<` is a space" join would append as if it were real name
/// content instead of discarding).
fn clean_name(raw: &str) -> String {
    let field = raw.split("<<").next().unwrap_or(raw);
    field
        .split('<')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn clean_field(raw: &str) -> String {
    raw.trim_matches('<').to_owned()
}

/// Resolves an MRZ `YYMMDD` two-digit year against the century closest to
/// today, allowing up to 10 years into the future (covers a passport's
/// expiry date; a birth date this "future" would just fail the sanity of
/// being a birth date, which isn't this function's job to judge).
fn resolve_year(yy: i32, today_year: i32) -> i32 {
    let century = (today_year / 100) * 100;
    let candidate = century + yy;
    if candidate > today_year + 10 {
        candidate - 100
    } else {
        candidate
    }
}

/// Formats an MRZ `YYMMDD` digit run as ISO 8601, or returns it unchanged
/// if it isn't a well-formed date (garbled OCR) rather than fabricating one.
fn format_mrz_date(yymmdd: &str, today: (i32, u32, u32)) -> String {
    let digits: Option<Vec<u32>> = yymmdd.chars().map(|c| c.to_digit(10)).collect();
    let Some(digits) = digits.filter(|d| d.len() == 6) else {
        return yymmdd.to_owned();
    };
    let yy = (digits[0] * 10 + digits[1]) as i32;
    let month = digits[2] * 10 + digits[3];
    let day = digits[4] * 10 + digits[5];
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return yymmdd.to_owned();
    }
    let year = resolve_year(yy, today.0);
    format!("{year:04}-{month:02}-{day:02}")
}

fn is_past(iso_date: &str, today: (i32, u32, u32)) -> bool {
    let Some((y, rest)) = iso_date.split_once('-') else {
        return false;
    };
    let Some((m, d)) = rest.split_once('-') else {
        return false;
    };
    let (Ok(y), Ok(m), Ok(d)) = (y.parse::<i32>(), m.parse::<u32>(), d.parse::<u32>()) else {
        return false;
    };
    (y, m, d) < today
}

/// Days-from-civil-date / civil-date-from-days conversion (Howard Hinnant's
/// public-domain algorithm) — avoids pulling in a date/time crate for the
/// one thing needed here: today's (year, month, day) in UTC, to resolve
/// MRZ two-digit years and flag an expired passport.
fn today_ymd() -> (i32, u32, u32) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let z = (secs / 86400) as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m, d)
}

/// `line1`/`line2` must already be exactly 44 uppercase MRZ characters each
/// (A-Z, 0-9, '<') — the caller is responsible for picking the two OCR
/// lines that look like MRZ rows and normalizing their text first.
pub fn parse_td3(line1: &str, line2: &str) -> Result<MrzFields, AppError> {
    if line1.chars().count() != LINE_LEN || line2.chars().count() != LINE_LEN {
        return Err(AppError::MrzNotFound);
    }
    let l2: Vec<char> = line2.chars().collect();

    let document_type = clean_field(&line1[0..2]);
    let (surname_raw, given_raw) = line1[5..44].split_once("<<").unwrap_or((&line1[5..44], ""));

    let passport_number = &line2[0..9];
    let passport_number_check = l2[9];
    let nationality_code = clean_field(&line2[10..13]);
    let date_of_birth_raw = &line2[13..19];
    let date_of_birth_check = l2[19];
    let sex = match l2[20] {
        'M' => "M",
        'F' => "F",
        _ => "X",
    };
    let date_of_expiry_raw = &line2[21..27];
    let date_of_expiry_check = l2[27];
    let personal_number = &line2[28..42];
    let personal_number_check = l2[42];
    let composite_check = l2[43];

    let composite_field = format!(
        "{passport_number}{passport_number_check}{date_of_birth_raw}{date_of_birth_check}{date_of_expiry_raw}{date_of_expiry_check}{personal_number}{personal_number_check}"
    );

    let today = today_ymd();
    let date_of_birth = format_mrz_date(date_of_birth_raw, today);
    let date_of_expiry = format_mrz_date(date_of_expiry_raw, today);

    Ok(MrzFields {
        document_type,
        issuing_country: clean_field(&line1[2..5]),
        surname: clean_name(surname_raw),
        given_names: clean_name(given_raw),
        passport_number: clean_field(passport_number),
        nationality: nationality_code,
        date_of_birth,
        sex: sex.to_owned(),
        date_of_expiry: date_of_expiry.clone(),
        personal_number: clean_field(personal_number),
        expired: is_past(&date_of_expiry, today),
        passport_number_check_ok: verify(passport_number, passport_number_check),
        date_of_birth_check_ok: verify(date_of_birth_raw, date_of_birth_check),
        date_of_expiry_check_ok: verify(date_of_expiry_raw, date_of_expiry_check),
        personal_number_check_ok: verify(personal_number, personal_number_check),
        composite_check_ok: verify(&composite_field, composite_check),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real ICAO 9303 Doc 9303-3 worked example.
    const LINE1: &str = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<";
    const LINE2: &str = "L898902C36UTO7408122F1204159ZE184226B<<<<<10";

    #[test]
    fn parses_the_icao_reference_example() {
        let doc = parse_td3(LINE1, LINE2).expect("valid MRZ");
        assert_eq!(doc.surname, "ERIKSSON");
        assert_eq!(doc.given_names, "ANNA MARIA");
        assert_eq!(doc.passport_number, "L898902C3");
        assert_eq!(doc.nationality, "UTO");
        assert_eq!(doc.date_of_birth, "1974-08-12");
        assert_eq!(doc.sex, "F");
        assert_eq!(doc.date_of_expiry, "2012-04-15");
        assert!(doc.expired); // 2012 is long past relative to any real "today" this runs on
        assert!(doc.passport_number_check_ok);
        assert!(doc.date_of_birth_check_ok);
        assert!(doc.date_of_expiry_check_ok);
        assert!(doc.composite_check_ok);
    }

    #[test]
    fn flags_a_tampered_passport_number() {
        let tampered = "L898902C46UTO7408122F1204159ZE184226B<<<<<10";
        let doc = parse_td3(LINE1, tampered).expect("still 44 chars, just wrong checksum");
        assert!(!doc.passport_number_check_ok);
        assert!(!doc.composite_check_ok);
    }

    #[test]
    fn rejects_wrong_length_lines() {
        assert!(parse_td3("TOO SHORT", LINE2).is_err());
    }

    #[test]
    fn resolves_near_future_two_digit_years_into_the_current_century() {
        // A 6-year passport issued in 2024 expiring "30" should read 2030,
        // not 1930 — within the +10y future window.
        assert_eq!(format_mrz_date("300705", (2026, 8, 16)), "2030-07-05");
        // Clearly a birth year, far outside the future window.
        assert_eq!(format_mrz_date("910824", (2026, 8, 16)), "1991-08-24");
    }
}
