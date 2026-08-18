//! Structured field extraction for a Nepali citizenship certificate
//! (नागरिकताको प्रमाणपत्र). Unlike the NID/passport services' generic
//! label/value engine, this returns a fixed-shape JSON object with named
//! keys (`full_name`, `gender`, `date_of_birth_ad`, ...) — chosen so a
//! downstream consumer can read `doc.full_name` directly instead of
//! scanning a `fields: [{label, value}]` array for a label string that
//! might read slightly differently on every scan.
//!
//! The certificate has two physical sides scanned as two separate images:
//! a bilingual (Devanagari + English) front with the certificate number,
//! name, parents, and addresses; and an English-only back with the same
//! identity fields restated plus the AD date of birth (split across
//! `Year:`/`Month:`/`Day:` rows) and the issuing officer. [`extract`] runs
//! on one side's OCR lines and fills whatever that side prints; [`combine`]
//! merges a front-side and back-side result into one document, preferring
//! the back page's plain-English fields where both sides printed the same
//! fact (its "Label: value" rows read more reliably than the front's
//! multi-field packed rows) and the front page for what only it has
//! (parents, addresses' local-language spelling, citizenship type).

use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Clone)]
pub struct OcrLine {
    pub text: String,
    pub confidence: f32,
    pub polygon: [[i32; 2]; 4],
}

/// Internal-only label/value pair for OCR text that didn't map to a named
/// field — never part of the public API response (see module docs); kept
/// only to log for debugging via `OCR_DEBUG_LINES`.
#[derive(Debug, PartialEq, Clone)]
struct Field {
    label: String,
    value: String,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
#[serde(default)]
pub struct CitizenshipDocument {
    pub citizenship_number: Option<String>,
    /// As printed — English (back page) preferred when both sides were
    /// scanned and both printed a name, since it's unambiguous machine
    /// text; falls back to the Devanagari name when only the front page
    /// was scanned.
    pub full_name: Option<String>,
    /// One of `"Male"`, `"Female"`, `"Other"` — normalized from whichever
    /// script/language the source printed (लिङ्ग / Sex), never the raw OCR
    /// text, so a downstream consumer can match against those three
    /// literal values instead of every spelling variant a scan might read.
    pub gender: Option<String>,
    /// ISO 8601 (YYYY-MM-DD), Gregorian — from the back page's `Year:`/
    /// `Month:`/`Day:` rows.
    pub date_of_birth_ad: Option<String>,
    /// `YYYY-MM-DD` in the Bikram Sambat calendar (not converted to AD) —
    /// from the front page's `साल`/`महिना`/`गते` rows.
    pub date_of_birth_bs: Option<String>,
    pub birth_district: Option<String>,
    pub birth_municipality: Option<String>,
    pub birth_ward: Option<String>,
    pub permanent_district: Option<String>,
    pub permanent_municipality: Option<String>,
    pub permanent_ward: Option<String>,
    pub father_name: Option<String>,
    pub mother_name: Option<String>,
    pub spouse_name: Option<String>,
    /// वंशज / जन्म / अंगीकृत / वैवाहिक अंगीकृत, as printed.
    pub citizenship_type: Option<String>,
    pub issuing_district_office: Option<String>,
    pub issuing_officer_name: Option<String>,
    pub issuing_officer_designation: Option<String>,
    /// `YYYY-MM-DD` in Bikram Sambat, as printed.
    pub date_of_issue_bs: Option<String>,
}

const MONTH_ABBREVIATIONS: &[&str] = &[
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
];

/// The form's literal placeholder for "not applicable" (an unmarried
/// holder's spouse name/number) — not a real value.
const NOT_APPLICABLE: &str = "XXX";

pub fn extract(lines: &[OcrLine]) -> CitizenshipDocument {
    let mut doc = CitizenshipDocument::default();

    doc.citizenship_number = value_after_keyword(lines, "ना.प्र.नं", &[])
        .or_else(|| value_after_keyword(lines, "Citizenship Certificate No", &[]))
        .map(|v| devanagari_digits_to_ascii(v.trim()));

    // "नाम थर" (the holder's own name) is also a substring of "बाबुको नाम
    // थर", "आमाको नाम थर", and "...को नामथर" (father/mother/spouse) — if
    // the holder's own line goes undetected (confirmed with a real scan:
    // one detection box spanned "आमाको नाम थर: फुलमाया अधिकारी : ना.प्र.नं:
    // ..." as a single line), the plain keyword search below would find
    // and attribute a *relative's* name instead. Scope the owner-only
    // searches (name, sex) to lines that aren't themselves one of those
    // relatives' rows, rather than trusting reading order alone to put the
    // holder's own row first.
    let owner_lines: Vec<OcrLine> = lines
        .iter()
        .filter(|line| !["बाबुको", "आमाको", "पति"].iter().any(|kw| line.text.contains(kw)))
        .cloned()
        .collect();

    let full_name_english =
        value_after_keyword(&owner_lines, "Full Name", &[]).map(|v| v.trim_end_matches('.').trim().to_owned());
    let full_name_devanagari =
        value_after_keyword(&owner_lines, "नाम थर", &["लिङ्ग", "ना.प्र.नं", "ना.कि", "जन्म"]);
    doc.full_name = full_name_english.or(full_name_devanagari);

    // "Ser" alongside "Sex" is a confirmed real recognizer misread (x -> r)
    // on this exact label — cheap and safe to also try, since it's a
    // 3-letter English word that never legitimately appears elsewhere on
    // this document.
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
        (Some(y), Some(m), Some(d)) => Some(format!("{y}-{m:0>2}-{d:0>2}")),
        _ => None,
    };

    let birth_block_devanagari = block_between(lines, "जन्म स्थान", &["स्थायी बासस्थान", "स्थायी ठेगाना"]);
    let birth_block_english = block_between(lines, "Birth Place", &["Permanent Address"]);
    let birth_block = if birth_block_devanagari.is_empty() { birth_block_english } else { birth_block_devanagari };
    doc.birth_district = value_after_keyword(birth_block, "जिल्ला", &[])
        .or_else(|| value_after_keyword(birth_block, "District", &[]));
    doc.birth_municipality = value_after_keyword(birth_block, "गा.पा", &["वडा"])
        .or_else(|| value_after_keyword(birth_block, "न.पा", &["वडा"]))
        .or_else(|| value_after_keyword(birth_block, "R. M", &["Ward"]))
        .or_else(|| value_after_keyword(birth_block, "R.M", &["Ward"]));
    doc.birth_ward = value_after_keyword(birth_block, "वडा", &[])
        .or_else(|| value_after_keyword(birth_block, "Ward", &[]))
        .map(|v| devanagari_digits_to_ascii(v.trim()))
        .filter(|v| is_plausible_ward_number(v));

    let permanent_block_devanagari = block_between(lines, "स्थायी बासस्थान", &["जन्म मिति", "बाबुको"])
        .none_if_empty()
        .or_else(|| block_between(lines, "स्थायी ठेगाना", &["जन्म मिति", "बाबुको"]).none_if_empty());
    let permanent_block_english = block_between(lines, "Permanent Address", &["नागरिकता", "Citizenship Type"]);
    let permanent_block = permanent_block_devanagari.unwrap_or(&[]);
    let permanent_block = if permanent_block.is_empty() { permanent_block_english } else { permanent_block };
    doc.permanent_district = value_after_keyword(permanent_block, "जिल्ला", &[])
        .or_else(|| value_after_keyword(permanent_block, "District", &[]));
    doc.permanent_municipality = value_after_keyword(permanent_block, "गा.पा", &["वडा"])
        .or_else(|| value_after_keyword(permanent_block, "न.पा", &["वडा"]))
        .or_else(|| value_after_keyword(permanent_block, "R. M", &["Ward"]))
        .or_else(|| value_after_keyword(permanent_block, "R.M", &["Ward"]));
    doc.permanent_ward = value_after_keyword(permanent_block, "वडा", &[])
        .or_else(|| value_after_keyword(permanent_block, "Ward", &[]))
        .map(|v| devanagari_digits_to_ascii(v.trim()))
        .filter(|v| is_plausible_ward_number(v));

    let father_block = block_between(lines, "बाबुको नाम थर", &["आमाको नाम थर"]);
    doc.father_name = value_after_keyword(father_block, "नाम थर", &["ना.प्र.नं", "ना.कि"]);

    let mother_block = block_between(lines, "आमाको नाम थर", &["पति", "पत्नी"]);
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
    doc.issuing_officer_name = value_after_keyword(officer_block, "नाम थर", &["दर्जा"]);
    doc.issuing_officer_designation = value_after_keyword(officer_block, "दर्जा", &["जारी मिति"]);
    doc.date_of_issue_bs =
        value_after_keyword(officer_block, "जारी मिति", &[]).map(|v| devanagari_digits_to_ascii(v.trim()));

    doc.issuing_district_office = value_after_keyword(lines, "जिल्ला प्रशासन कार्यालय", &[]);

    // Never part of the response (see `Field`'s doc comment) — logged only,
    // so a scan that's coming back thinner than expected can still be
    // debugged by re-running with OCR_DEBUG_LINES=1 without the API
    // contract itself carrying an unpredictable, unparseable bag of
    // leftover text.
    if std::env::var("OCR_DEBUG_LINES").is_ok() {
        for field in parse_generic_fields(lines) {
            tracing::debug!(label = %field.label, value = %field.value, "unmatched OCR field");
        }
    }

    sanitize(doc)
}

/// Canonicalizes a recognized sex marker to exactly one of three literal
/// values, in whichever script/language it printed — anything that doesn't
/// cleanly match a known token (including a garbled OCR misread) comes back
/// `None` rather than surfacing noise as if it were a real value.
fn normalize_gender(raw: &str) -> Option<String> {
    match raw.trim() {
        "पुरुष" | "पुरूष" | "Male" | "male" | "M" | "m" => Some("Male".to_owned()),
        "महिला" | "Female" | "female" | "F" | "f" => Some("Female".to_owned()),
        "अन्य" | "Other" | "other" => Some("Other".to_owned()),
        _ => None,
    }
}

/// Final cleanup pass: drops any free-text field that's empty after
/// trimming or contains no letters at all (a lone digit/punctuation
/// fragment that survived because *some* text followed the label keyword,
/// but isn't plausibly the field's real value). Applied both after
/// [`extract`] and after [`combine`], since `/combine` also accepts a
/// caller-supplied document directly — the API should return equally clean
/// output either way.
fn sanitize(mut doc: CitizenshipDocument) -> CitizenshipDocument {
    doc.full_name = clean_text(doc.full_name);
    doc.birth_district = clean_text(doc.birth_district);
    doc.birth_municipality = clean_text(doc.birth_municipality);
    doc.permanent_district = clean_text(doc.permanent_district);
    doc.permanent_municipality = clean_text(doc.permanent_municipality);
    doc.father_name = clean_text(doc.father_name);
    doc.mother_name = clean_text(doc.mother_name);
    doc.spouse_name = clean_text(doc.spouse_name);
    doc.citizenship_type = clean_text(doc.citizenship_type);
    doc.issuing_district_office = clean_text(doc.issuing_district_office);
    doc.issuing_officer_name = clean_text(doc.issuing_officer_name);
    doc.issuing_officer_designation = clean_text(doc.issuing_officer_designation);
    doc
}

fn clean_text(value: Option<String>) -> Option<String> {
    value.filter(|text| {
        let trimmed = text.trim();
        !trimmed.is_empty() && trimmed.chars().any(char::is_alphabetic)
    })
}

/// Merges a front-side and back-side [`extract`] result into one document.
/// Missing sides pass `&CitizenshipDocument::default()`.
pub fn combine(front: &CitizenshipDocument, back: &CitizenshipDocument) -> CitizenshipDocument {
    sanitize(CitizenshipDocument {
        citizenship_number: back.citizenship_number.clone().or_else(|| front.citizenship_number.clone()),
        full_name: back.full_name.clone().or_else(|| front.full_name.clone()),
        gender: back.gender.clone().or_else(|| front.gender.clone()),
        date_of_birth_ad: back.date_of_birth_ad.clone().or_else(|| front.date_of_birth_ad.clone()),
        date_of_birth_bs: front.date_of_birth_bs.clone().or_else(|| back.date_of_birth_bs.clone()),
        birth_district: back.birth_district.clone().or_else(|| front.birth_district.clone()),
        birth_municipality: back.birth_municipality.clone().or_else(|| front.birth_municipality.clone()),
        birth_ward: back.birth_ward.clone().or_else(|| front.birth_ward.clone()),
        permanent_district: back.permanent_district.clone().or_else(|| front.permanent_district.clone()),
        permanent_municipality: back.permanent_municipality.clone().or_else(|| front.permanent_municipality.clone()),
        permanent_ward: back.permanent_ward.clone().or_else(|| front.permanent_ward.clone()),
        father_name: front.father_name.clone().or_else(|| back.father_name.clone()),
        mother_name: front.mother_name.clone().or_else(|| back.mother_name.clone()),
        spouse_name: front.spouse_name.clone().or_else(|| back.spouse_name.clone()),
        citizenship_type: back.citizenship_type.clone().or_else(|| front.citizenship_type.clone()),
        issuing_district_office: front.issuing_district_office.clone().or_else(|| back.issuing_district_office.clone()),
        issuing_officer_name: back.issuing_officer_name.clone().or_else(|| front.issuing_officer_name.clone()),
        issuing_officer_designation: back.issuing_officer_designation.clone().or_else(|| front.issuing_officer_designation.clone()),
        date_of_issue_bs: back.date_of_issue_bs.clone().or_else(|| front.date_of_issue_bs.clone()),
    })
}

fn combine_ad_date(year: Option<String>, month: Option<String>, day: Option<String>) -> Option<String> {
    let year: i32 = year?.trim().parse().ok()?;
    let month_upper = month?.trim().to_uppercase();
    let month_num = MONTH_ABBREVIATIONS.iter().position(|m| *m == month_upper)? as u32 + 1;
    let day: u32 = day?.trim().parse().ok()?;
    if !(1..=31).contains(&day) {
        return None;
    }
    Some(format!("{year:04}-{month_num:02}-{day:02}"))
}

fn devanagari_digits_to_ascii(text: &str) -> String {
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

/// Finds `keyword` and returns the text after *its own* colon — scoped to
/// stop before the first occurrence of any of `stop_keywords`, so a
/// multi-word label like "वडा नं." or "Ward No." (keyword `"वडा"`/`"Ward"`,
/// with trailing abbreviation text of its own before the colon) still
/// resolves to the colon that actually belongs to it, and a single row
/// packing multiple `label : value` pairs (this form's parents'
/// name+citizenship-number and address+citizenship-type rows) doesn't leak
/// the next label's colon into this one's value. Falls back to the text
/// right after the keyword when no colon is found in scope at all.
///
/// Skips a candidate line entirely (keeps searching the remaining lines)
/// if the matched value itself contains the certificate's own repeating
/// watermark phrase — the recognizer sometimes reads a faint watermark
/// line as if it were real content sitting in an otherwise-blank field
/// (seen in practice on the "spouse name" row, which is blank on an
/// unmarried holder's certificate). A real field value never legitimately
/// contains this phrase, so it's an unambiguous noise signal.
///
/// Falls back to the nearest neighboring box (same row to the right, else
/// the row below) when the keyword's own line has no value in it at all —
/// confirmed necessary against real bounding-box output: a *clean*
/// detection routinely puts "Citizenship Certificate No.:" and
/// "65-01-77-02872" in two entirely separate boxes rather than one merged
/// string, which the same-line-only check above can't see at all. Both
/// paths were needed on real scans, not just one.
fn value_after_keyword(lines: &[OcrLine], keyword: &str, stop_keywords: &[&str]) -> Option<String> {
    for (index, line) in lines.iter().enumerate() {
        let text = &line.text;
        let Some(kw_pos) = text.find(keyword) else { continue };
        let after_kw = &text[kw_pos + keyword.len()..];
        let stop_pos = stop_keywords.iter().filter_map(|kw| after_kw.find(kw)).min().unwrap_or(after_kw.len());
        let scope = &after_kw[..stop_pos];
        let value = match scope.find(':') {
            Some(colon_pos) => &scope[colon_pos + 1..],
            None => scope,
        };
        let value = value.trim().trim_matches(|c: char| c == '.' || c == '*' || c.is_whitespace());
        if !value.is_empty() && !looks_like_watermark_noise(value) {
            return Some(value.to_owned());
        }
        if let Some(value) = nearest_value_near(lines, index, stop_keywords) {
            return Some(value);
        }
    }
    None
}

/// Substrings from the certificate's own repeating background watermark
/// and page boilerplate — never legitimately part of a field's *value*
/// (only its labels/headers), so a value containing one is a misread of
/// the watermark bleeding through, not real content.
const WATERMARK_PHRASES: &[&str] = &[
    "नागरिकताको प्रमाण",
    "नागरिकता ऐन",
    "नेपाली नागरिकता",
    "नेपाल सरकार",
    "गृह मन्त्रालय",
    "जिल्ला प्रशासन",
];

fn looks_like_watermark_noise(text: &str) -> bool {
    WATERMARK_PHRASES.iter().any(|phrase| text.contains(phrase))
}

/// A ward number is 1-2 plain digits — nothing else a real value could be.
/// Guards `birth_ward`/`permanent_ward` against a garbled OCR line (words,
/// punctuation, or an unrelated nearby run of digits) surviving just
/// because *some* text followed the "वडा"/"Ward" keyword.
fn is_plausible_ward_number(text: &str) -> bool {
    !text.is_empty() && text.len() <= 2 && text.chars().all(|c| c.is_ascii_digit())
}

/// Same as [`value_after_keyword`], but the caller only wants the leading
/// run of ASCII digits from the result (the back page's `Year:`/`Day:`
/// rows).
fn number_after_keyword(lines: &[OcrLine], keyword: &str) -> Option<String> {
    let value = value_after_keyword(lines, keyword, &[])?;
    let digits: String = value.chars().filter(char::is_ascii_digit).collect();
    (!digits.is_empty()).then_some(digits)
}

/// Finds `keyword` and returns the run of digits (Devanagari or ASCII,
/// converted to ASCII) immediately following it, skipping any separator
/// punctuation in between — the front page's `साल २०६०` / `महिना १०` /
/// `गते २६` triplet, which prints its numbers in Devanagari digits.
fn devanagari_number_after(lines: &[OcrLine], keyword: &str) -> Option<String> {
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

/// The contiguous run of `lines` from the line containing `start_keyword`
/// (inclusive) up to the line containing any of `end_keywords` (exclusive),
/// or to the end of `lines` if none of them appear. Used to scope a
/// keyword search (e.g. "ना.प्र.नं") to one section of the document when
/// the same label text is printed more than once (father's vs. mother's
/// citizenship number).
fn block_between<'a>(lines: &'a [OcrLine], start_keyword: &str, end_keywords: &[&str]) -> &'a [OcrLine] {
    let Some(start) = lines.iter().position(|line| line.text.contains(start_keyword)) else {
        return &[];
    };
    let end = lines[start + 1..]
        .iter()
        .position(|line| end_keywords.iter().any(|kw| line.text.contains(kw)))
        .map(|offset| start + 1 + offset)
        .unwrap_or(lines.len());
    &lines[start..end]
}

trait NoneIfEmpty<'a> {
    fn none_if_empty(self) -> Option<&'a [OcrLine]>;
}
impl<'a> NoneIfEmpty<'a> for &'a [OcrLine] {
    fn none_if_empty(self) -> Option<&'a [OcrLine]> {
        (!self.is_empty()).then_some(self)
    }
}

// ---- generic label/value fallback (debug logging only, see `extract`) ----

const FIELD_LABEL_MIN_CONFIDENCE: f32 = 0.90;
const FIELD_VALUE_MIN_CONFIDENCE: f32 = 0.75;
const MAX_FIELD_ALIGNMENT_DRIFT: i32 = 120;

const LABEL_WORDS: &[&str] = &[
    "GOVERNMENT", "NEPAL", "CITIZENSHIP", "CERTIFICATE", "NO", "FULL", "NAME", "SEX", "DATE", "OF",
    "BIRTH", "PLACE", "PERMANENT", "ADDRESS", "DISTRICT", "WARD", "YEAR", "MONTH", "DAY", "RM",
    "TYPE", "ISSUED", "ISSUING", "OFFICER", "AUTHORITY", "SIGNATURE", "HOLDER",
];

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

/// The nearest line that plausibly holds `lines[label_index]`'s value:
/// prefers another box on the same visual row to the right (the common
/// case — "Label:" then its value beside it), falling back to the row
/// below (the front page's stacked "label above, value below" rows) only
/// if nothing qualifies to the right. Skips anything that itself looks
/// like a label (a real label sitting to the right/below would mean this
/// field's value is actually still missing, not that label), and anything
/// matching `stop_keywords` or the watermark, same as the caller's
/// same-line check.
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
        // Any vertical overlap at all is too weak a "same row" test — a
        // full-width header row above the real row can graze a label's
        // top edge by a few pixels and still count, and its left edge
        // landing near the page margin (same as the label's) then made it
        // win on distance over the real value further right. Confirmed on
        // a real scan: the header line and the true value both technically
        // overlapped "Citizenship Certificate No.:" vertically, but the
        // header's overlap was 17px of the label's 60px height (28%)
        // against the real value's 50px (83%) — require the overlap to be
        // most of the label's own height, not just nonzero.
        let overlap = bounds.bottom.min(label_bounds.bottom) - bounds.top.max(label_bounds.top);
        let same_row = overlap > label_bounds.height() / 2;
        if trace {
            eprintln!(
                "  candidate={:?} bounds=(l{},t{},r{},b{}) same_row={}",
                line.text, bounds.left, bounds.top, bounds.right, bounds.bottom, same_row
            );
        }
        // A candidate box's left edge only has to clear the label's own
        // left edge, not its right edge — detection boxes routinely
        // overlap horizontally by a few pixels (the corner-expansion/
        // unclip padding both boxes get), so requiring strictly "starts
        // after the label ends" missed a real match on a real scan where
        // "Full Name:" and "YUBIN ADHIKARI" were adjacent, correctly
        // detected, separate boxes.
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
    (!value.is_empty() && !looks_like_watermark_noise(value)).then(|| value.to_owned())
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
            polygon: [[left, top], [left + width, top], [left + width, top + 20], [left, top + 20]],
        }
    }

    /// Real front-page transcription (the sample this service was tuned
    /// against): certificate number, name/sex, birth place, permanent
    /// address, BS date of birth, and father/mother/spouse blocks — each on
    /// its own detected line the way PP-OCRv5 actually segments this form.
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

    /// Real back-page transcription.
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
        assert_eq!(doc.date_of_birth_bs.as_deref(), Some("2060-10-26"));
        assert_eq!(doc.birth_district.as_deref(), Some("बर्दिया"));
        assert_eq!(doc.birth_municipality.as_deref(), Some("बढैयाताल"));
        assert_eq!(doc.birth_ward.as_deref(), Some("2"));
        assert_eq!(doc.permanent_district.as_deref(), Some("बर्दिया"));
        assert_eq!(doc.permanent_municipality.as_deref(), Some("बढैयाताल"));
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
        assert_eq!(doc.date_of_birth_ad.as_deref(), Some("2004-02-09"));
        assert_eq!(doc.birth_district.as_deref(), Some("Bardiya"));
        assert_eq!(doc.birth_municipality.as_deref(), Some("badhaiyatal"));
        assert_eq!(doc.birth_ward.as_deref(), Some("2"));
        assert_eq!(doc.permanent_district.as_deref(), Some("Bardiya"));
        assert_eq!(doc.permanent_municipality.as_deref(), Some("badhaiyatal"));
        assert_eq!(doc.permanent_ward.as_deref(), Some("2"));
        assert_eq!(doc.citizenship_type.as_deref(), Some("वंशज"));
        assert_eq!(doc.issuing_officer_name.as_deref(), Some("गणेश विक्रम शाह"));
        assert_eq!(doc.issuing_officer_designation.as_deref(), Some("प्रशासकीय अधिकृत"));
        assert_eq!(doc.date_of_issue_bs.as_deref(), Some("2077-08-07"));
    }

    #[test]
    fn combine_prefers_back_page_english_and_front_page_family_data() {
        let front = extract(&front_page_lines());
        let back = extract(&back_page_lines());
        let combined = combine(&front, &back);

        assert_eq!(combined.citizenship_number.as_deref(), Some("65-01-77-02872"));
        assert_eq!(combined.full_name.as_deref(), Some("YUBIN ADHIKARI"));
        assert_eq!(combined.gender.as_deref(), Some("Male"));
        assert_eq!(combined.date_of_birth_ad.as_deref(), Some("2004-02-09"));
        assert_eq!(combined.date_of_birth_bs.as_deref(), Some("2060-10-26"));
        assert_eq!(combined.father_name.as_deref(), Some("रामजी प्रसाद अधिकारी"));
        assert_eq!(combined.mother_name.as_deref(), Some("फुलमाया अधिकारी"));
        assert_eq!(combined.issuing_officer_name.as_deref(), Some("गणेश विक्रम शाह"));
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
        // A garbled recognizer misread must not surface as if it were a
        // real value — sanitizer rejects, doesn't guess.
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
        // /combine also accepts a caller-built document, not just one this
        // service's own `extract` produced — its output must be equally
        // clean either way.
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
}
