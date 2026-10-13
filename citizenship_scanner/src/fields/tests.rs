//! Current-layout parser and document tests on hand-transcribed real OCR lines.

use super::current_layout::extract;
use super::document::{CitizenshipDocument, clean_text, combine};
use super::layout::OcrLine;
use super::values::normalize_gender;

fn line(text: &str, confidence: f32, left: i32, top: i32) -> OcrLine {
    line_with_width(text, confidence, left, top, 100)
}

fn line_with_width(text: &str, confidence: f32, left: i32, top: i32, width: i32) -> OcrLine {
    OcrLine {
        text: text.to_owned(),
        confidence,
        polygon: [[left, top], [left + width, top], [left + width, top + 20], [left, top + 20]],
    }
}

/// Real front page, one detected line per row.
fn front_page_lines() -> Vec<OcrLine> {
    vec![
        line_with_width("नेपाल सरकार", 0.98, 400, 10, 200),
        line_with_width("गृह मन्त्रालय", 0.98, 380, 40, 240),
        line_with_width("जिल्ला प्रशासन कार्यालय ........... बर्दिया", 0.97, 300, 70, 400),
        line_with_width("नेपाली नागरिकताको प्रमाणपत्र", 0.97, 350, 100, 350),
        line_with_width("ना.प्र.नं. : ६५-०१-७७-०२८७२", 0.96, 10, 140, 350),
        line_with_width("नाम थर: युबिन अधिकारी", 0.96, 10, 180, 300),
        line("लिङ्ग : पुरुष", 0.96, 700, 180),
        line_with_width("जन्म स्थान: जिल्ला : बर्दिया", 0.95, 10, 220, 350),
        line("गा.पा. : बढैयाताल", 0.95, 10, 250),
        line("वडा नं. : २", 0.95, 700, 250),
        line_with_width("स्थायी बासस्थान: जिल्ला : बर्दिया", 0.95, 10, 290, 350),
        line("गा.पा. : बढैयाताल", 0.95, 10, 320),
        line("वडा नं. : २", 0.95, 700, 320),
        line_with_width("जन्म मिति: साल २०६० महिना १० गते २६", 0.94, 10, 360, 400),
        line_with_width("बाबुको नाम थर : रामजी प्रसाद अधिकारी", 0.94, 10, 400, 400),
        line("ना.प्र.नं.: ८०५", 0.94, 700, 400),
        line_with_width("ठेगाना : बढैयाताल गा.पा.-२, बर्दिया", 0.93, 10, 430, 400),
        line("ना.कि.: वंशज", 0.93, 700, 430),
        line_with_width("आमाको नाम थर: फुलमाया अधिकारी", 0.94, 10, 470, 400),
        line("ना.प्र.नं.: १७७४०", 0.94, 700, 470),
        line_with_width("ठेगाना : बढैयाताल गा.पा.-२, बर्दिया", 0.93, 10, 500, 400),
        line("ना.कि.: वंशज", 0.93, 700, 500),
        line_with_width("पति/पत्नीको नामथर : XXX", 0.9, 10, 540, 300),
        line("ना.प्र.नं.:", 0.7, 700, 540),
    ]
}

/// Real back page.
fn back_page_lines() -> Vec<OcrLine> {
    vec![
        line_with_width(
            "Government of Nepal has issued this Citizenship Certificate with following details.",
            0.97,
            10,
            10,
            800,
        ),
        line_with_width("Citizenship Certificate No.: 65-01-77-02872", 0.98, 10, 60, 400),
        line("Sex: Male", 0.98, 700, 60),
        line_with_width("Full Name.: YUBIN ADHIKARI", 0.98, 10, 100, 350),
        line_with_width("Date of Birth (AD): Year:2004", 0.97, 10, 140, 350),
        line("Month:FEB", 0.97, 500, 140),
        line("Day:09", 0.97, 700, 140),
        line_with_width("Birth Place: District: Bardiya", 0.96, 10, 180, 350),
        line("R. M. : badhaiyatal", 0.96, 500, 210),
        line("Ward No.:2", 0.96, 800, 210),
        line_with_width("Permanent Address: District: Bardiya", 0.96, 10, 250, 400),
        line("R. M. : badhaiyatal", 0.96, 500, 280),
        line("Ward No.:2", 0.96, 800, 280),
        line_with_width(
            "नेपाल नागरिकता ऐन २०६३ बमोजिम यो नागरिकताको प्रमाणपत्र दिइएको छ",
            0.9,
            10,
            320,
            600,
        ),
        line("नागरिकता किसिम: वंशज", 0.93, 10, 360),
        line_with_width("प्रमाण पत्र जारी गर्ने अधिकारीको", 0.9, 700, 400, 300),
        line("दस्तखत :", 0.85, 700, 430),
        line("नाम थर :गणेश विक्रम शाह", 0.93, 700, 460),
        line("दर्जा :प्रशासकीय अधिकृत", 0.93, 700, 490),
        line("जारी मिति : २०७७-०८-०७", 0.93, 700, 520),
    ]
}

#[test]
fn extracts_every_named_field_from_the_front_page() {
    let lines = front_page_lines();
    let doc = extract(&lines);

    assert_eq!(doc.citizenship_number.as_deref(), Some("65-01-77-02872"));
    assert_eq!(doc.full_name.as_deref(), Some("युबिन अधिकारी"));
    assert_eq!(doc.gender.as_deref(), Some("Male"));
    assert_eq!(doc.date_of_birth_bs.as_deref(), Some("2060/10/26"));
    assert_eq!(doc.birth_district.as_deref(), Some("बर्दिया"));
    assert_eq!(doc.birth_municipal.as_deref(), Some("बढैयाताल Rural Municipality"));
    assert_eq!(doc.birth_ward.as_deref(), Some("2"));
    assert_eq!(doc.permanent_district.as_deref(), Some("बर्दिया"));
    assert_eq!(doc.permanent_municipal.as_deref(), Some("बढैयाताल Rural Municipality"));
    assert_eq!(doc.permanent_ward.as_deref(), Some("2"));
    assert_eq!(doc.father_name.as_deref(), Some("रामजी प्रसाद अधिकारी"));
    assert_eq!(doc.mother_name.as_deref(), Some("फुलमाया अधिकारी"));
    assert_eq!(doc.spouse_name, None); // literal "XXX" placeholder must not surface as a name
    assert_eq!(doc.citizenship_type.as_deref(), Some("वंशज"));
}

#[test]
fn extracts_every_named_field_from_the_back_page() {
    let lines = back_page_lines();
    let doc = extract(&lines);

    assert_eq!(doc.citizenship_number.as_deref(), Some("65-01-77-02872"));
    assert_eq!(doc.full_name.as_deref(), Some("YUBIN ADHIKARI"));
    assert_eq!(doc.gender.as_deref(), Some("Male"));
    assert_eq!(doc.date_of_birth_ad.as_deref(), Some("2004/02/09"));
    assert_eq!(doc.birth_district.as_deref(), Some("Bardiya"));
    assert_eq!(doc.birth_municipal.as_deref(), Some("badhaiyatal Rural Municipality"));
    assert_eq!(doc.birth_ward.as_deref(), Some("2"));
    assert_eq!(doc.permanent_district.as_deref(), Some("Bardiya"));
    assert_eq!(doc.permanent_municipal.as_deref(), Some("badhaiyatal Rural Municipality"));
    assert_eq!(doc.permanent_ward.as_deref(), Some("2"));
    assert_eq!(doc.citizenship_type.as_deref(), Some("वंशज"));
    assert_eq!(doc.date_of_issue_bs.as_deref(), Some("2077/08/07"));
}

#[test]
fn permanent_address_falls_back_to_whatever_prints_below_birth_place() {
    // "Permanent Address" unreadable: the second-District fallback must still find the block.
    let mut lines = back_page_lines();
    let permanent_label = lines
        .iter_mut()
        .find(|l| l.text.starts_with("Permanent Address"))
        .unwrap();
    permanent_label.text = "@#$%^&*: District: Bardiya".to_owned();

    let doc = extract(&lines);

    assert_eq!(doc.permanent_district.as_deref(), Some("Bardiya"));
    assert_eq!(doc.permanent_municipal.as_deref(), Some("badhaiyatal Rural Municipality"));
    assert_eq!(doc.permanent_ward.as_deref(), Some("2"));
    assert_eq!(doc.birth_district.as_deref(), Some("Bardiya"));
    assert_eq!(doc.birth_municipal.as_deref(), Some("badhaiyatal Rural Municipality"));
    assert_eq!(doc.birth_ward.as_deref(), Some("2"));
}

#[test]
fn combine_prefers_back_page_english_and_front_page_family_data() {
    let front = extract(&front_page_lines());
    let back = extract(&back_page_lines());
    let combined = combine(&front, &back);

    assert_eq!(combined.citizenship_number.as_deref(), Some("65-01-77-02872"));
    assert_eq!(combined.full_name.as_deref(), Some("YUBIN ADHIKARI"));
    assert_eq!(combined.gender.as_deref(), Some("Male"));
    assert_eq!(combined.date_of_birth_ad.as_deref(), Some("2004/02/09"));
    assert_eq!(combined.date_of_birth_bs.as_deref(), Some("2060/10/26"));
    assert_eq!(combined.father_name.as_deref(), Some("रामजी प्रसाद अधिकारी"));
    assert_eq!(combined.mother_name.as_deref(), Some("फुलमाया अधिकारी"));
}

#[test]
fn combine_handles_a_missing_side() {
    let front = extract(&front_page_lines());
    let combined = combine(&front, &CitizenshipDocument::default());
    assert_eq!(combined.citizenship_number.as_deref(), Some("65-01-77-02872"));
    assert_eq!(combined.full_name.as_deref(), Some("युबिन अधिकारी"));
}

#[test]
fn gender_normalizes_to_one_of_three_literal_values_or_none() {
    assert_eq!(normalize_gender("पुरुष"), Some("Male".to_owned()));
    assert_eq!(normalize_gender("पुरूष"), Some("Male".to_owned()));
    assert_eq!(normalize_gender("Male"), Some("Male".to_owned()));
    assert_eq!(normalize_gender("महिला"), Some("Female".to_owned()));
    assert_eq!(normalize_gender("Female"), Some("Female".to_owned()));
    assert_eq!(normalize_gender("अन्य"), Some("Other".to_owned()));
    // Misreads must not pass as values.
    assert_eq!(normalize_gender("पुरुषभीकी प्ाण"), None);
}

#[test]
fn sanitize_drops_letterless_fragments_but_keeps_real_text() {
    assert_eq!(clean_text(Some("2".to_owned())), None);
    assert_eq!(clean_text(Some("  ".to_owned())), None);
    assert_eq!(clean_text(Some("".to_owned())), None);
    assert_eq!(clean_text(Some("वंशज".to_owned())), Some("वंशज".to_owned()));
    assert_eq!(clean_text(None), None);
}

#[test]
fn combine_sanitizes_a_caller_supplied_document_from_the_api_directly() {
    // /combine also takes caller-built documents.
    let junk_front = CitizenshipDocument {
        father_name: Some("2".to_owned()),
        citizenship_type: Some("  ".to_owned()),
        full_name: Some("युबिन अधिकारी".to_owned()),
        ..CitizenshipDocument::default()
    };
    let combined = combine(&junk_front, &CitizenshipDocument::default());
    assert_eq!(combined.father_name, None);
    assert_eq!(combined.citizenship_type, None);
    assert_eq!(combined.full_name.as_deref(), Some("युबिन अधिकारी"));
}
