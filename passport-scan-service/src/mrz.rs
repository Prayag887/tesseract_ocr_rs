//! ICAO 9303 TD3 (passport) machine-readable zone: two 44-char lines,
//! fixed-column fields + mod-10 weighted checksums. Completely different
//! extraction strategy from the NID service's label/value spatial parser —
//! MRZ has no labels at all, just position and checksum.

use serde::Serialize;

use crate::error::AppError;

const LINE_LEN: usize = 44;
/// ICAO 9303 checksum weights, repeating every 3 characters.
const WEIGHTS: [u32; 3] = [7, 3, 1];

#[derive(Debug, Serialize, PartialEq)]
pub struct MrzDocument {
    pub document_type: String,
    pub issuing_country: String,
    pub surname: String,
    pub given_names: String,
    pub passport_number: String,
    pub nationality: String,
    pub date_of_birth: String,
    pub sex: String,
    pub date_of_expiry: String,
    pub personal_number: String,
    pub passport_number_check_ok: bool,
    pub date_of_birth_check_ok: bool,
    pub date_of_expiry_check_ok: bool,
    pub personal_number_check_ok: bool,
    pub composite_check_ok: bool,
}

/// Weighted mod-10 check digit: digits are their own value, A-Z = 10-35,
/// '<' = 0, weights cycle 7/3/1 across the input — ICAO 9303 Appendix A.
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

fn clean_name(raw: &str) -> String {
    raw.trim_matches('<').replace('<', " ").trim().to_owned()
}

fn clean_field(raw: &str) -> String {
    raw.trim_matches('<').to_owned()
}

/// `line1`/`line2` must already be exactly 44 uppercase MRZ characters each
/// (A-Z, 0-9, '<') — the caller is responsible for picking the two OCR
/// lines that look like MRZ rows and normalizing their text first.
pub fn parse_td3(line1: &str, line2: &str) -> Result<MrzDocument, AppError> {
    if line1.len() != LINE_LEN || line2.len() != LINE_LEN {
        return Err(AppError::MrzNotFound);
    }
    let l2: Vec<char> = line2.chars().collect();

    let document_type = clean_field(&line1[0..2]);
    let issuing_country = clean_field(&line1[2..5]);
    let (surname_raw, given_raw) = line1[5..44].split_once("<<").unwrap_or((&line1[5..44], ""));

    let passport_number = &line2[0..9];
    let passport_number_check = l2[9];
    let nationality = clean_field(&line2[10..13]);
    let date_of_birth = &line2[13..19];
    let date_of_birth_check = l2[19];
    let sex = match l2[20] {
        'M' => "M",
        'F' => "F",
        _ => "X",
    };
    let date_of_expiry = &line2[21..27];
    let date_of_expiry_check = l2[27];
    let personal_number = &line2[28..42];
    let personal_number_check = l2[42];
    let composite_check = l2[43];

    let composite_field = format!(
        "{passport_number}{passport_number_check}{date_of_birth}{date_of_birth_check}{date_of_expiry}{date_of_expiry_check}{personal_number}{personal_number_check}"
    );

    Ok(MrzDocument {
        document_type,
        issuing_country,
        surname: clean_name(surname_raw),
        given_names: clean_name(given_raw),
        passport_number: clean_field(passport_number),
        nationality,
        date_of_birth: date_of_birth.to_owned(),
        sex: sex.to_owned(),
        date_of_expiry: date_of_expiry.to_owned(),
        personal_number: clean_field(personal_number),
        passport_number_check_ok: verify(passport_number, passport_number_check),
        date_of_birth_check_ok: verify(date_of_birth, date_of_birth_check),
        date_of_expiry_check_ok: verify(date_of_expiry, date_of_expiry_check),
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
        assert_eq!(doc.date_of_birth, "740812");
        assert_eq!(doc.sex, "F");
        assert_eq!(doc.date_of_expiry, "120415");
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
}
