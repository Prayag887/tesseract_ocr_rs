//! Primary field parser ([`extract`]) for the current card layout. Old yellow cards: `legacy_layout.rs` (fallback only).

use super::document::{CitizenshipDocument, sanitize};
use super::generic::parse_generic_fields;
use super::keywords::{contains_keyword, value_after_keyword};
use super::layout::{NoneIfEmpty, OcrLine, block_between, second_occurrence_block};
use super::values::{bs_date, bs_date_from_row, combine_ad_date, devanagari_digits_to_ascii, devanagari_number_after, normalize_gender, number_after_keyword, parse_bs_date, ward_number};

/// Local-body labels -> type name appended to the value. Longest first: shorter ones are
/// substrings of longer ones ("न.पा" in "म.न.पा", "Municipality" in "Rural Municipality").
const LOCAL_BODY_LABELS: &[(&str, &str)] = &[
    ("उ.म.न.पा", "Sub-Metropolitan City"),
    ("म.न.पा", "Metropolitan City"),
    ("गा.वि.स", "VDC"),
    ("गा.पा", "Rural Municipality"),
    ("न.पा", "Municipality"),
    ("Sub-Metropolitan", "Sub-Metropolitan City"),
    ("Metropolitan", "Metropolitan City"),
    ("Rural Municipality", "Rural Municipality"),
    ("VDC", "VDC"),
    ("Municipality", "Municipality"),
    ("R. M", "Rural Municipality"),
    ("R.M", "Rural Municipality"),
];

/// Value of whichever local-body label `block` prints, with its type appended ("Aarubote VDC").
fn local_body_in(block: &[OcrLine]) -> Option<String> {
    for (keyword, type_name) in LOCAL_BODY_LABELS {
        let Some(value) = value_after_keyword(block, keyword, &["वडा", "Ward"]) else { continue };
        let value = value.trim();
        // Check the name before appending the type; "ल Municipality" would pass later length checks.
        if value.chars().count() < 2 {
            continue;
        }
        return Some(format!("{value} {type_name}"));
    }
    None
}

/// Printed placeholder for "not applicable" (unmarried holder's spouse).
const NOT_APPLICABLE: &str = "XXX";

pub fn extract(lines: &[OcrLine]) -> CitizenshipDocument {
    let mut doc = CitizenshipDocument::default();

    // Holder-only fields (number, name, sex) search only lines before the first relative marker:
    // "नाम थर" and "ना.प्र.नं" repeat in the father/mother/spouse rows. Fuzzy match (a false exclude
    // is harmless), and truncate rather than filter — a relative's number box has no marker itself.
    let owner_end = lines
        .iter()
        .position(|line| ["बाबुको", "आमाको", "पति"].iter().any(|kw| contains_keyword(&line.text, kw)))
        .unwrap_or(lines.len());
    let owner_lines: Vec<OcrLine> = lines[..owner_end].to_vec();

    // "ना.प्र.न" retries a label misread by two edits. Stops are the labels printed right of the number.
    const NUMBER_STOPS: &[&str] = &["Sex", "Ser", "लिङ्ग", "Full Name", "नाम थर"];
    doc.citizenship_number = value_after_keyword(&owner_lines, "ना.प्र.नं", NUMBER_STOPS)
        .or_else(|| value_after_keyword(&owner_lines, "ना.प्र.न", NUMBER_STOPS))
        .or_else(|| value_after_keyword(&owner_lines, "Citizenship Certificate No", NUMBER_STOPS))
        .map(|v| devanagari_digits_to_ascii(v.trim()));

    let full_name_english =
        value_after_keyword(&owner_lines, "Full Name", &[]).map(|v| v.trim_end_matches('.').trim().to_owned());
    let full_name_devanagari =
        value_after_keyword(&owner_lines, "नाम थर", &["लिङ्ग", "ना.प्र.नं", "ना.कि", "जन्म"]);
    doc.full_name = full_name_english.or(full_name_devanagari);

    // "Ser" is a real misread of "Sex".
    doc.gender = value_after_keyword(&owner_lines, "Sex", &[])
        .or_else(|| value_after_keyword(&owner_lines, "Ser", &[]))
        .or_else(|| value_after_keyword(&owner_lines, "लिङ्ग", &[]))
        .and_then(|raw| normalize_gender(&raw));

    let year = number_after_keyword(lines, "Year");
    let month = value_after_keyword(lines, "Month", &[]);
    let day = number_after_keyword(lines, "Day");
    doc.date_of_birth_ad = combine_ad_date(year, month, day);

    let bs_year = devanagari_number_after(lines, "साल");
    let bs_month = devanagari_number_after(lines, "महिना");
    let bs_day = devanagari_number_after(lines, "गते");
    doc.date_of_birth_bs = match (bs_year, bs_month, bs_day) {
        (Some(y), Some(m), Some(d)) => bs_date(&y, &m, &d),
        // One bad label (e.g. "गते" read as "वत्ते"): take the three numbers on the साल row in order.
        _ => bs_date_from_row(lines),
    };

    // Short "स्थान"/"बासस्थान" fallbacks: a mangled header empties the whole block. Safe because
    // birth place prints first and the permanent header ends the birth block.
    let birth_block_devanagari = block_between(lines, "जन्म स्थान", &["स्थायी बासस्थान", "स्थायी ठेगाना"])
        .none_if_empty()
        .or_else(|| block_between(lines, "स्थान", &["बासस्थान", "ठेगाना"]).none_if_empty())
        .unwrap_or(&[]);
    let birth_block_english = block_between(lines, "Birth Place", &["Permanent Address"]);
    let birth_block = if birth_block_devanagari.is_empty() { birth_block_english } else { birth_block_devanagari };
    doc.birth_district = value_after_keyword(birth_block, "जिल्ला", &[])
        .or_else(|| value_after_keyword(birth_block, "District", &[]));
    doc.birth_municipal = local_body_in(birth_block);
    doc.birth_ward = value_after_keyword(birth_block, "वडा", &[])
        .or_else(|| value_after_keyword(birth_block, "Ward", &[]))
        .and_then(|v| ward_number(&v));

    // "बासस्थान" alone: same fallback as above.
    let permanent_block_devanagari = block_between(lines, "स्थायी बासस्थान", &["जन्म मिति", "बाबुको"])
        .none_if_empty()
        .or_else(|| block_between(lines, "स्थायी ठेगाना", &["जन्म मिति", "बाबुको"]).none_if_empty())
        .or_else(|| block_between(lines, "बासस्थान", &["जन्म मिति", "बाबुको"]).none_if_empty());
    let permanent_block_english = block_between(lines, "Permanent Address", &["नागरिकता", "Citizenship Type"]);
    let permanent_block = permanent_block_devanagari.unwrap_or(&[]);
    let permanent_block = if permanent_block.is_empty() { permanent_block_english } else { permanent_block };
    // Last resort: "District" prints exactly twice; the second one starts the permanent block.
    let permanent_block = if permanent_block.is_empty() {
        second_occurrence_block(
            lines,
            &["District", "जिल्ला"],
            &["जन्म मिति", "बाबुको", "नागरिकता", "Citizenship Type"],
        )
    } else {
        permanent_block
    };
    doc.permanent_district = value_after_keyword(permanent_block, "जिल्ला", &[])
        .or_else(|| value_after_keyword(permanent_block, "District", &[]));
    doc.permanent_municipal = local_body_in(permanent_block);
    doc.permanent_ward = value_after_keyword(permanent_block, "वडा", &[])
        .or_else(|| value_after_keyword(permanent_block, "Ward", &[]))
        .and_then(|v| ward_number(&v));

    // Bare "बाबुको"/"आमाको": the header is often split across two boxes.
    let father_block = block_between(lines, "बाबुको नाम थर", &["आमाको नाम थर"])
        .none_if_empty()
        .or_else(|| block_between(lines, "बाबुको", &["आमाको"]).none_if_empty())
        .unwrap_or(&[]);
    doc.father_name = value_after_keyword(father_block, "नाम थर", &["ना.प्र.नं", "ना.कि"]);

    let mother_block = block_between(lines, "आमाको नाम थर", &["पति", "पत्नी"])
        .none_if_empty()
        .or_else(|| block_between(lines, "आमाको", &["पति", "पत्नी"]).none_if_empty())
        .unwrap_or(&[]);
    doc.mother_name = value_after_keyword(mother_block, "नाम थर", &["ना.प्र.नं", "ना.कि"]);

    let spouse_block = block_between(lines, "पति", &[]);
    doc.spouse_name = value_after_keyword(spouse_block, "नामथर", &["ना.प्र.नं", "ना.कि"])
        .or_else(|| value_after_keyword(spouse_block, "नाम थर", &["ना.प्र.नं", "ना.कि"]))
        .filter(|v| !v.is_empty() && !v.eq_ignore_ascii_case(NOT_APPLICABLE));

    doc.citizenship_type = value_after_keyword(lines, "नागरिकता किसिम", &[])
        .or_else(|| value_after_keyword(lines, "नागरिकताको किसिम", &[]))
        .or_else(|| value_after_keyword(lines, "Citizenship Type", &[]))
        .or_else(|| value_after_keyword(father_block, "ना.कि", &[]))
        .or_else(|| value_after_keyword(mother_block, "ना.कि", &[]));

    let officer_block = block_between(lines, "जारी गर्ने अधिकारी", &[]);
    // "जारी मिति" is unique on the page, so search everywhere when the officer block header is unread.
    doc.date_of_issue_bs = value_after_keyword(officer_block, "जारी मिति", &[])
        .or_else(|| value_after_keyword(lines, "जारी मिति", &[]))
        .and_then(|v| parse_bs_date(v.trim()));

    // Debug only — not part of the response.
    if std::env::var("OCR_DEBUG_LINES").is_ok() {
        for field in parse_generic_fields(lines) {
            tracing::debug!(label = %field.label, value = %field.value, "unmatched OCR field");
        }
    }

    sanitize(doc)
}
