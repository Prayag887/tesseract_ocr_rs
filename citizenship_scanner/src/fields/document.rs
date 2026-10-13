//! The API output type, plus what works on finished documents: [`sanitize`], [`combine`],
//! and the low-yield check ([`populated_fields`], [`LOW_YIELD_FIELDS`]).

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
#[serde(default)]
pub struct CitizenshipDocument {
    pub citizenship_number: Option<String>,
    /// As printed; English (back) preferred over Devanagari (front) by [`combine`].
    pub full_name: Option<String>,
    /// Exactly "Male", "Female" or "Other".
    pub gender: Option<String>,
    /// `YYYY/MM/DD` Gregorian, from the back page.
    pub date_of_birth_ad: Option<String>,
    /// `YYYY/MM/DD` Bikram Sambat, from the front page.
    pub date_of_birth_bs: Option<String>,
    pub birth_district: Option<String>,
    /// Name plus local-body type ("Aarubote VDC"); the type is printed in the label and isn't
    /// derivable from the name.
    pub birth_municipal: Option<String>,
    pub birth_ward: Option<String>,
    pub permanent_district: Option<String>,
    pub permanent_municipal: Option<String>,
    pub permanent_ward: Option<String>,
    pub father_name: Option<String>,
    pub mother_name: Option<String>,
    pub spouse_name: Option<String>,
    pub citizenship_type: Option<String>,
    pub date_of_issue_bs: Option<String>,
}

/// Drops free-text fields with no letters or under 2 chars. Applied after `extract` and
/// [`combine`] (which also takes caller-supplied documents).
pub(super) fn sanitize(mut doc: CitizenshipDocument) -> CitizenshipDocument {
    doc.full_name = clean_text(doc.full_name);
    doc.birth_district = clean_text(doc.birth_district);
    doc.birth_municipal = clean_text(doc.birth_municipal);
    doc.permanent_district = clean_text(doc.permanent_district);
    doc.permanent_municipal = clean_text(doc.permanent_municipal);
    doc.father_name = clean_text(doc.father_name);
    doc.mother_name = clean_text(doc.mother_name);
    doc.spouse_name = clean_text(doc.spouse_name);
    doc.citizenship_type = clean_text(doc.citizenship_type);
    doc
}

pub(super) fn clean_text(value: Option<String>) -> Option<String> {
    value.filter(|text| {
        let trimmed = text.trim();
        // 2+ chars: a lone combining mark like "ः" counts as alphabetic.
        trimmed.chars().count() >= 2 && trimmed.chars().any(char::is_alphabetic)
    })
}

/// Below this many fields a scan counts as failed and triggers the fallbacks in `extraction`.
/// Working scans fill 9+, failed ones 0-1.
pub const LOW_YIELD_FIELDS: usize = 4;

/// Number of non-`None` fields.
pub fn populated_fields(doc: &CitizenshipDocument) -> usize {
    [
        &doc.citizenship_number,
        &doc.full_name,
        &doc.gender,
        &doc.date_of_birth_ad,
        &doc.date_of_birth_bs,
        &doc.birth_district,
        &doc.birth_municipal,
        &doc.birth_ward,
        &doc.permanent_district,
        &doc.permanent_municipal,
        &doc.permanent_ward,
        &doc.father_name,
        &doc.mother_name,
        &doc.spouse_name,
        &doc.citizenship_type,
        &doc.date_of_issue_bs,
    ]
    .iter()
    .filter(|field| field.is_some())
    .count()
}

/// Merges front and back; pass `CitizenshipDocument::default()` for a missing side.
pub fn combine(front: &CitizenshipDocument, back: &CitizenshipDocument) -> CitizenshipDocument {
    sanitize(CitizenshipDocument {
        citizenship_number: back.citizenship_number.clone().or_else(|| front.citizenship_number.clone()),
        full_name: back.full_name.clone().or_else(|| front.full_name.clone()),
        gender: back.gender.clone().or_else(|| front.gender.clone()),
        date_of_birth_ad: back.date_of_birth_ad.clone().or_else(|| front.date_of_birth_ad.clone()),
        date_of_birth_bs: front.date_of_birth_bs.clone().or_else(|| back.date_of_birth_bs.clone()),
        birth_district: back.birth_district.clone().or_else(|| front.birth_district.clone()),
        birth_municipal: back.birth_municipal.clone().or_else(|| front.birth_municipal.clone()),
        birth_ward: back.birth_ward.clone().or_else(|| front.birth_ward.clone()),
        permanent_district: back.permanent_district.clone().or_else(|| front.permanent_district.clone()),
        permanent_municipal: back.permanent_municipal.clone().or_else(|| front.permanent_municipal.clone()),
        permanent_ward: back.permanent_ward.clone().or_else(|| front.permanent_ward.clone()),
        father_name: front.father_name.clone().or_else(|| back.father_name.clone()),
        mother_name: front.mother_name.clone().or_else(|| back.mother_name.clone()),
        spouse_name: front.spouse_name.clone().or_else(|| back.spouse_name.clone()),
        citizenship_type: back.citizenship_type.clone().or_else(|| front.citizenship_type.clone()),
        date_of_issue_bs: back.date_of_issue_bs.clone().or_else(|| front.date_of_issue_bs.clone()),
    })
}
