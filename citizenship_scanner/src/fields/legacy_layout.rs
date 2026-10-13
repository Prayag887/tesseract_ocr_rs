//! Old (yellow, pre-2073) card layout. Fallback only: runs [`extract`] first and fills its gaps;
//! called just for scans `extraction` already judged low-yield.
//!
//! Old-layout differences: front number has no label (back: "ना.प्र.प.नं"); names are "नाम,थर";
//! values and block headers sit in their own boxes and often sort before their labels; one
//! combined "उ.म.न.पा/गा.वि.स" local-body label; back uses "बाहकको नाम" and a bare "मिति".

use super::current_layout::extract;
use super::document::{CitizenshipDocument, sanitize};
use super::keywords::{contains_keyword, find_keyword, looks_like_noise};
use super::layout::{Bounds, OcrLine};
use super::values::{devanagari_digits_to_ascii, normalize_gender, parse_bs_date, ward_number};

/// A value equal to one of these means the search ran into the next row's label.
const LABEL_ONLY_VALUES: &[&str] = &["ठेगाना", "नागरिकता", "नाम थर", "नाम,थर", "जिल्ला", "वडा नं"];

/// [`extract`] + old-layout rules for the fields it left empty + punctuation tidy-up.
pub fn extract_fallback(lines: &[OcrLine]) -> CitizenshipDocument {
    let mut doc = extract(lines);

    // Position-based rules (n-th name/address row) are only safe on the old front, identified by
    // its combined local-body label; on a newer card they mis-assigned the mother as father.
    let old_front = lines.iter().any(|line| is_combined_local_body_label(&line.text));

    // With the combined label, the generic local-body value is just the label's tail — replace it.
    let municipal = nth_values(lines, |text| is_combined_local_body_label(text), combined_local_body_value);
    if let Some(value) = municipal.first() {
        doc.birth_municipal = value.clone();
    }
    if let Some(value) = municipal.get(1) {
        doc.permanent_municipal = value.clone();
    }

    if old_front {
        fill_old_front_addresses(&mut doc, lines);
    }

    // Old front prints exactly three name rows: holder, father, mother.
    if old_front {
        let mut name_labels: Vec<usize> =
            (0..lines.len()).filter(|&index| is_name_label(&lines[index].text)).collect();
        name_labels.sort_by_key(|&index| Bounds::from_line(&lines[index]).top);
        let names: Vec<Option<String>> = name_labels.iter().map(|&index| row_value(lines, index)).collect();
        fill(&mut doc.full_name, names.first().cloned().flatten());
        fill(&mut doc.father_name, names.get(1).cloned().flatten());
        fill(&mut doc.mother_name, names.get(2).cloned().flatten());
    }
    fill(
        &mut doc.full_name,
        lines.iter().find_map(|line| value_after_fuzzy(&line.text, "बाहकको नाम")),
    );

    fill(&mut doc.gender, gender_from_row(lines));
    fill(&mut doc.citizenship_number, citizenship_number(lines));
    fill(
        &mut doc.date_of_issue_bs,
        lines
            .iter()
            .filter(|line| line.text.contains("मिति") && !line.text.contains("जन्म"))
            .find_map(|line| value_after(&line.text, "मिति").and_then(|value| parse_bs_date(&value))),
    );

    tidy(doc)
}

/// First district/ward row is birth place, second is permanent address.
fn fill_old_front_addresses(doc: &mut CitizenshipDocument, lines: &[OcrLine]) {
    let districts = nth_values(
        lines,
        |text| contains_keyword(text, "जिल्ला") && !text.contains("प्रशासन") && !looks_like_noise(text),
        |text| value_after(text, "जिल्ला"),
    );
    fill(&mut doc.birth_district, districts.first().cloned().flatten());
    fill(&mut doc.permanent_district, districts.get(1).cloned().flatten());

    let wards = nth_values(lines, |text| text.contains("वडा"), |text| {
        value_after(text, "वडा").and_then(|value| ward_number(&value))
    });
    fill(&mut doc.birth_ward, wards.first().cloned().flatten());
    fill(&mut doc.permanent_ward, wards.get(1).cloned().flatten());
}

fn fill(field: &mut Option<String>, value: Option<String>) {
    if field.is_none() {
        *field = value;
    }
}

/// `value_of` for each `is_label` line, top to bottom.
fn nth_values(
    lines: &[OcrLine],
    is_label: impl Fn(&str) -> bool,
    value_of: impl Fn(&str) -> Option<String>,
) -> Vec<Option<String>> {
    let mut matches: Vec<&OcrLine> = lines.iter().filter(|line| is_label(&line.text)).collect();
    matches.sort_by_key(|line| Bounds::from_line(line).top);
    matches.into_iter().map(|line| value_of(&line.text)).collect()
}

/// Text after `keyword`'s colon (or after the keyword), label punctuation stripped.
fn value_after(text: &str, keyword: &str) -> Option<String> {
    let start = text.find(keyword)? + keyword.len();
    after_colon(&text[start..])
}

fn value_after_fuzzy(text: &str, keyword: &str) -> Option<String> {
    let (_, end) = find_keyword(text, keyword, 1)?;
    after_colon(&text[end..])
}

fn after_colon(rest: &str) -> Option<String> {
    let value = match rest.find([':', 'ः']) {
        Some(pos) => &rest[pos + rest[pos..].chars().next().map_or(1, char::len_utf8)..],
        None => rest,
    };
    let value = trim_label_punctuation(value);
    (!value.is_empty()).then(|| value.to_owned())
}

fn trim_label_punctuation(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || matches!(c, '-' | ':' | 'ः' | '.' | ',' | '/' | '*'))
}

/// "उ.म.न.पा/गा.वि.स": the slash plus the VDC abbreviation (OCR often drops the dots).
fn is_combined_local_body_label(text: &str) -> bool {
    text.contains('/') && (contains_keyword(text, "गा.वि.स") || contains_keyword(text, "गाविस"))
}

/// Place after the combined label, as printed; type unknown, so none appended.
fn combined_local_body_value(text: &str) -> Option<String> {
    let label_end = ["गा.वि.स", "गाविस"]
        .iter()
        .find_map(|keyword| find_keyword(text, keyword, 1).map(|(_, end)| end))?;
    let mut rest = &text[label_end..];
    if let Some(stop) = rest.find("वडा") {
        rest = &rest[..stop];
    }
    let value = trim_label_punctuation(rest);
    (value.chars().count() >= 2).then(|| value.to_owned())
}

/// "नाम थर" / "नाम,थर", with or without a relative's prefix.
fn is_name_label(text: &str) -> bool {
    let compact: String = text.chars().filter(|c| !c.is_whitespace() && *c != ',').collect();
    compact.contains("नामथर") && !looks_like_noise(text)
}

/// Name label's own tail plus every box to its right on the same row, up to the next label.
fn row_value(lines: &[OcrLine], label_index: usize) -> Option<String> {
    let label = Bounds::from_line(&lines[label_index]);
    let own = {
        let text = &lines[label_index].text;
        let compact_end = text.find("थर").map(|pos| pos + "थर".len()).unwrap_or(text.len());
        after_colon(&text[compact_end..]).unwrap_or_default()
    };

    let mut row: Vec<(i32, &str)> = lines
        .iter()
        .enumerate()
        .filter(|&(index, _)| index != label_index)
        .filter_map(|(_, line)| {
            let bounds = Bounds::from_line(line);
            let overlap = bounds.bottom.min(label.bottom) - bounds.top.max(label.top);
            (overlap > label.height() / 2 && bounds.left > label.left).then_some((bounds.left, line.text.as_str()))
        })
        .collect();
    row.sort_by_key(|&(left, _)| left);

    let mut parts = vec![own];
    for (_, text) in row {
        if is_row_stop(text) {
            break;
        }
        parts.push(trim_label_punctuation(text).to_owned());
    }
    let value = parts.into_iter().filter(|part| !part.is_empty()).collect::<Vec<_>>().join(" ");
    (!value.is_empty()).then_some(value)
}

/// Box that starts the next label on the row.
fn is_row_stop(text: &str) -> bool {
    text.starts_with("लि")
        || ["लिङ्ग", "ठेगाना", "नागरिकता", "ना.प्र"].iter().any(|keyword| contains_keyword(text, keyword))
        || text.contains(':')
        || text.contains('ः')
}

/// Sex box: the label OCRs too mangled ("लिडङ्ः"), so anchor on a box starting "लि" + gender word.
fn gender_from_row(lines: &[OcrLine]) -> Option<String> {
    lines.iter().filter(|line| line.text.starts_with("लि")).find_map(|line| {
        line.text
            .split(|c: char| c.is_whitespace() || c == ':' || c == 'ः' || c == '-')
            .find_map(normalize_gender)
    })
}

/// Labelled back number or unlabeled front number; only clean digit reads (else none).
fn citizenship_number(lines: &[OcrLine]) -> Option<String> {
    let labeled = lines
        .iter()
        .find_map(|line| value_after_fuzzy(&line.text, "ना.प्र.प.नं").and_then(|value| clean_number(&value)));
    if labeled.is_some() {
        return labeled;
    }
    let first_name_top = lines
        .iter()
        .filter(|line| is_name_label(&line.text))
        .map(|line| Bounds::from_line(line).top)
        .min()?;
    lines
        .iter()
        .filter(|line| Bounds::from_line(line).bottom <= first_name_top + Bounds::from_line(line).height() / 2)
        .find_map(|line| clean_number(&line.text))
}

fn clean_number(text: &str) -> Option<String> {
    let converted = devanagari_digits_to_ascii(trim_label_punctuation(text));
    let digits = converted.chars().filter(char::is_ascii_digit).count();
    let clean = converted.chars().all(|c| c.is_ascii_digit() || matches!(c, '/' | '-' | ' '));
    (clean && digits >= 6).then(|| converted.split_whitespace().collect::<Vec<_>>().join(""))
}

/// First line with 8+ digits that isn't a date — the number line to re-read. No label required
/// (the old label never OCRs reliably; the old front has none).
pub fn number_line(lines: &[OcrLine]) -> Option<usize> {
    lines.iter().position(|line| {
        let text = devanagari_digits_to_ascii(&line.text);
        let digit_runs: Vec<&str> =
            text.split(|c: char| !c.is_ascii_digit()).filter(|run| !run.is_empty()).collect();
        let digits: usize = digit_runs.iter().map(|run| run.len()).sum();
        let is_date = ["मिति", "साल", "ऐन", "जारी"].iter().any(|kw| line.text.contains(kw))
            || matches!(digit_runs.as_slice(), [y, m, d] if y.len() == 4 && m.len() <= 2 && d.len() <= 2);
        digits >= 8 && !is_date
    })
}

/// Majority vote over many reads of the number line: misreads scatter, the true digits repeat.
/// Needs 3+ votes and a strict lead; separator taken from the agreeing reads.
pub fn vote_number(reads: &[(String, f32)]) -> Option<String> {
    let mut tally: Vec<(String, usize, f32, Vec<String>)> = Vec::new();
    for (text, confidence) in reads {
        let Some(formatted) = format_number_read(text) else { continue };
        let key: String = formatted.chars().filter(char::is_ascii_digit).collect();
        match tally.iter_mut().find(|(existing, ..)| *existing == key) {
            Some((_, count, total, forms)) => {
                *count += 1;
                *total += confidence;
                forms.push(formatted);
            }
            None => tally.push((key, 1, *confidence, vec![formatted])),
        }
    }
    tally.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.total_cmp(&a.2)));
    let (key, count, _, forms) = tally.first()?;
    let runner_up = tally.get(1).map_or(0, |entry| entry.1);
    if *count < 3 || *count <= runner_up {
        return None;
    }
    let mut separated: Vec<&String> = forms.iter().filter(|form| form.len() > key.len()).collect();
    separated.sort();
    let best_form = separated
        .chunk_by(|a, b| a == b)
        .max_by_key(|group| group.len())
        .map(|group| group[0].clone());
    Some(best_form.unwrap_or_else(|| key.clone()))
}

/// Digits of one read; each non-digit run between them -> one separator ("-" if present, else "/").
/// `None` below 8 digits.
fn format_number_read(text: &str) -> Option<String> {
    let converted = devanagari_digits_to_ascii(text);
    let start = converted.find(|c: char| c.is_ascii_digit())?;
    let end = converted.rfind(|c: char| c.is_ascii_digit())? + 1;
    let mut out = String::new();
    let mut gap = String::new();
    for c in converted[start..end].chars() {
        if c.is_ascii_digit() {
            if !gap.is_empty() {
                if gap.contains('-') {
                    out.push('-');
                } else if !gap.trim().is_empty() {
                    out.push('/');
                }
                gap.clear();
            }
            out.push(c);
        } else {
            gap.push(c);
        }
    }
    (out.chars().filter(char::is_ascii_digit).count() >= 8).then_some(out)
}

/// Strips leaked label punctuation; drops values that are just a label.
fn tidy(mut doc: CitizenshipDocument) -> CitizenshipDocument {
    for field in [
        &mut doc.full_name,
        &mut doc.birth_district,
        &mut doc.birth_municipal,
        &mut doc.permanent_district,
        &mut doc.permanent_municipal,
        &mut doc.father_name,
        &mut doc.mother_name,
        &mut doc.spouse_name,
        &mut doc.citizenship_type,
    ] {
        *field = field.take().and_then(|value| {
            let value = trim_label_punctuation(&value);
            let is_label = LABEL_ONLY_VALUES.iter().any(|label| value == *label);
            (!value.is_empty() && !is_label).then(|| value.to_owned())
        });
    }
    sanitize(doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, left: i32, top: i32, width: i32) -> OcrLine {
        OcrLine {
            text: text.to_owned(),
            confidence: 0.9,
            polygon: [[left, top], [left + width, top], [left + width, top + 40], [left, top + 40]],
        }
    }

    /// Real OCR output of the old front (raw, upright), post-sort order.
    fn old_front_lines() -> Vec<OcrLine> {
        vec![
            line("नेपाल सरकार", 700, 80, 300),
            line("नेपाली नागरिकताको प्रमाण-पत्र", 580, 290, 760),
            line("२७१०६०/९९८६", 460, 400, 360),
            line("अविनाश थापा मगर", 800, 458, 420),
            line("लिङ्गः पुरुष", 1360, 460, 200),
            line("नाम थर:-", 585, 462, 180),
            line("जिल्ला:- काठमाण्डौ", 810, 540, 380),
            line("जन्मस्थान-", 585, 545, 210),
            line("वडा नं:९", 1300, 610, 180),
            line("उ.म.न.पा/ गा.वि.सः का.म.पा.", 585, 615, 560),
            line("जिल्ला:- काठमाण्डौ", 930, 680, 380),
            line("स्थायी बासस्थान-", 585, 685, 330),
            line("वडा नं:९", 1300, 750, 180),
            line("उ.म.न.पा/ गा.वि.सः का.म.पा.", 585, 755, 560),
            line("जन्ममिति- सालः- २०४८", 585, 830, 470),
            line("महिना: ०२ गतेः ३१", 1160, 832, 350),
            line("वाबके", 585, 885, 110),
            line("नाम,थर:-राम", 700, 888, 240),
            line("थापा मगर", 1170, 886, 220),
            line("बहादुर", 1000, 890, 140),
            line("आमाको", 585, 1025, 130),
            line("नाम,थरः-", 720, 1028, 190),
        ]
    }

    #[test]
    fn primary_extraction_misses_the_old_layout() {
        // If the primary parser ever reads this layout, revisit the fallback.
        let doc = extract(&old_front_lines());
        assert!(doc.full_name.is_none() || doc.full_name.as_deref() != Some("अविनाश थापा मगर"));
    }

    #[test]
    fn extracts_the_old_front_page() {
        let doc = extract_fallback(&old_front_lines());
        assert_eq!(doc.citizenship_number.as_deref(), Some("271060/9986"));
        assert_eq!(doc.full_name.as_deref(), Some("अविनाश थापा मगर"));
        assert_eq!(doc.gender.as_deref(), Some("Male"));
        assert_eq!(doc.date_of_birth_bs.as_deref(), Some("2048/02/31"));
        assert_eq!(doc.birth_district.as_deref(), Some("काठमाण्डौ"));
        assert_eq!(doc.permanent_district.as_deref(), Some("काठमाण्डौ"));
        assert_eq!(doc.birth_municipal.as_deref(), Some("का.म.पा"));
        assert_eq!(doc.permanent_municipal.as_deref(), Some("का.म.पा"));
        assert_eq!(doc.birth_ward.as_deref(), Some("9"));
        assert_eq!(doc.permanent_ward.as_deref(), Some("9"));
        assert_eq!(doc.father_name.as_deref(), Some("राम बहादुर थापा मगर"));
        assert_eq!(doc.mother_name, None);
    }

    #[test]
    fn extracts_the_old_back_page() {
        let lines = vec![
            line("पति/ पत्नीको: नाम थर:", 250, 40, 520),
            line("ठेगाना:", 250, 110, 170),
            line("नेपाल नागरिकता ऐन २०६३ बमोजिम यो नागरिकताको प्रमाण-पत्र दिइएको छ।", 250, 190, 1700),
            line("नागरिकताको किसिम: वंशज", 250, 360, 680),
            line("प्रमाणपत्र बाहकको नाम:-अबिनाश थापा मगर", 250, 290, 1050),
            line("ना.प्र.प.नं:- २७१०६०/९९८६", 1230, 650, 700),
            line("मितिः-२०६४/०५/२६", 1230, 1140, 560),
        ];
        let doc = extract_fallback(&lines);
        assert_eq!(doc.full_name.as_deref(), Some("अबिनाश थापा मगर"));
        assert_eq!(doc.citizenship_type.as_deref(), Some("वंशज"));
        assert_eq!(doc.citizenship_number.as_deref(), Some("271060/9986"));
        assert_eq!(doc.date_of_issue_bs.as_deref(), Some("2064/05/26"));
        assert_eq!(doc.spouse_name, None);
    }

    #[test]
    fn positional_name_rows_are_left_alone_on_the_newer_layout() {
        // Newer-layout front: positional names must not apply.
        let lines = vec![
            line("गा.पा: बढैयाताल", 100, 300, 400),
            line("आमाको नाम थर: फुलमाया अधिकारी", 100, 600, 700),
            line("पति/पत्नीको नाम थर: XXX", 100, 700, 700),
        ];
        let doc = extract_fallback(&lines);
        assert_eq!(doc.father_name, None);
        assert_ne!(doc.full_name.as_deref(), Some("फुलमाया अधिकारी"));
    }

    #[test]
    fn number_vote_recovers_the_printed_number_from_noisy_reads() {
        // Real reads of the old back's number.
        let reads: Vec<(String, f32)> = [
            "ना.प्र प न: २७१०६०९९८६",
            "न २७१०६०×९९८६",
            "न:- २७१०६०\\९९८६",
            "न:- २७१०६०^९९८६",
            "न:- २७१०६०/९९८६",
            "२७१०६०४९९८६",
            "२७१०८०९९5६",
            "-२७१०६०/९९५६",
        ]
        .iter()
        .map(|text| (text.to_string(), 0.8))
        .collect();
        assert_eq!(vote_number(&reads).as_deref(), Some("271060/9986"));
    }

    #[test]
    fn number_vote_needs_a_clear_majority() {
        let reads: Vec<(String, f32)> =
            ["२७१०६०/९९८६", "२७९०६०/९९८६", "२७१०६०/९९५६"].iter().map(|t| (t.to_string(), 0.8)).collect();
        assert_eq!(vote_number(&reads), None);
    }

    #[test]
    fn number_line_skips_dates_and_finds_unlabeled_numbers() {
        let lines = vec![
            line("नेपाल नागरिकता ऐन २०६३ बमोजिम", 0, 0, 900),
            line("मितिः २०६४/०५/२६", 0, 100, 400),
            line("R७9०६0/९९५६", 0, 200, 400),
        ];
        assert_eq!(number_line(&lines), Some(2));
    }

    #[test]
    fn rejects_a_partly_misread_number() {
        assert_eq!(clean_number("R७9०६0/९९५६"), None);
        assert_eq!(clean_number("२७१००*९९५९"), None);
        assert_eq!(clean_number("२७१०६०/९९८६").as_deref(), Some("271060/9986"));
    }
}
