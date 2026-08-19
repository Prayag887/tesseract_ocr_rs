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

/// Sorts detected lines into reading order (top-to-bottom rows, left-to-
/// right within a row) in place. Every label/value pairing in this module
/// — `block_between`'s "label starts the block", `nearest_value_near`'s
/// same-row-then-below search — assumes the lines arrive in this order;
/// nothing upstream (detection, recognition) actually guarantees it, they
/// just hand back boxes in whatever order the model happened to propose
/// them. Confirmed on a real scan where this bit: "Birth Place:" and its
/// same-row value "District: Panchthar" came back with the *value*
/// listed first, because that box's detected top edge sat a few pixels
/// above the label box's — enough to invert a naive sort-by-top, even
/// though the two are visually on the same line. `block_between` then
/// started the birth-place block one line late, after the row it needed,
/// so the real birth district fell outside it and a *different* row's
/// district (the permanent address's) fell inside it instead.
///
/// Plain sort-by-top can't fix this — it's the same bug. Rows are found
/// by clustering: two boxes belong to the same row if their vertical
/// spans overlap at all, not by comparing single top coordinates.
///
/// Each row is anchored to the *first* (topmost) box assigned to it, and
/// every later candidate is tested against that one anchor — not against
/// the row's current envelope. Tested against the envelope first and it
/// chains: box B overlaps anchor A, so B joins and the envelope grows to
/// cover both; box C only overlaps the *new, taller* envelope, not A
/// itself, so C joins too; repeat. On a real scan with five stacked
/// label rows spaced tightly enough that each row's box just brushed the
/// next, every label in the form's left column chained into a single
/// "row" that swallowed the whole column, so all five labels sorted
/// before any of their values — exactly the label/value inversion this
/// function exists to prevent, just at a larger scale.
pub fn sort_reading_order(lines: &mut Vec<OcrLine>) {
    let mut indexed: Vec<(usize, Bounds)> =
        lines.iter().enumerate().map(|(index, line)| (index, Bounds::from_line(line))).collect();
    indexed.sort_by_key(|(_, bounds)| bounds.top);

    let mut rows: Vec<(Bounds, Vec<(usize, Bounds)>)> = Vec::new();
    for entry in indexed {
        let (_, bounds) = entry;
        match rows.last_mut() {
            Some((anchor, row)) if bounds.top < anchor.bottom && anchor.top < bounds.bottom => {
                row.push(entry);
            }
            _ => rows.push((bounds, vec![entry])),
        }
    }
    let rows: Vec<Vec<(usize, Bounds)>> = rows.into_iter().map(|(_, row)| row).collect();

    let originals: Vec<OcrLine> = lines.drain(..).collect();
    for mut row in rows {
        row.sort_by_key(|(_, bounds)| bounds.left);
        lines.extend(merge_adjacent(&originals, &row));
    }
}

/// Rejoins boxes the detector split mid-phrase, so one printed line comes
/// back as one `OcrLine`. Two boxes merge when the horizontal gap between
/// them is under `WORD_GAP_RATIO` of the row's text height — ordinary word
/// spacing — and stay separate beyond that, which is where this form's
/// label and value columns sit.
///
/// Needed because detection granularity isn't stable across preprocessing:
/// on an upscaled crop the detector starts emitting one box *per word*
/// ("प्रयाग" and "ढकाल" separately, "नाम"/"थर" separately) where the same
/// card at native resolution gave one box per line. Every label in this
/// module is matched as a contiguous string, so a label split across two
/// boxes ("नाम थर" becoming "नाम" + "थर") stops matching entirely — and
/// the value that belongs to it lands in a third box with nothing left to
/// tie it to. Confirmed on a real scan: the whole front page extracted
/// zero fields with word-split detection despite the recognizer reading
/// every character on it correctly.
fn merge_adjacent(originals: &[OcrLine], row: &[(usize, Bounds)]) -> Vec<OcrLine> {
    /// Fraction of text height a gap may span and still count as a space
    /// between words of one phrase rather than a column break. Tuned down
    /// from 1.0 against real scans: at 1.0 this form's label and value
    /// columns merged into one line, which let a field's search run past
    /// its own column and take the *next* one's value ("Citizenship
    /// Certificate No." merged with the "Sex: Male" beside it and returned
    /// `citizenship_number: "Male"`). Ordinary word spacing sits well
    /// under half the text height, so 0.5 keeps phrases joined while
    /// leaving columns apart.
    const WORD_GAP_RATIO: f32 = 0.5;

    /// How far two boxes' vertical centres may sit apart, as a fraction of
    /// the smaller box's height, and still count as the same printed line.
    /// The row grouping that feeds this function deliberately accepts *any*
    /// vertical overlap, which is right for ordering but far too loose for
    /// merging: it put the back page's preamble sentence in the same group
    /// as the label row beneath it (their boxes overlap by a few pixels),
    /// and merging on gap alone then produced "Citizenship Certificate
    /// No.: Government of Nepal has issued this... 65-01-77-02872" as a
    /// single line. Comparing centres instead of overlap keeps stacked
    /// rows apart no matter how much their boxes bleed into each other.
    const BASELINE_TOLERANCE: f32 = 0.4;

    let same_line = |a: Bounds, b: Bounds| {
        let centre_gap = ((a.top + a.bottom) - (b.top + b.bottom)).abs() as f32 / 2.0;
        centre_gap <= a.height().min(b.height()) as f32 * BASELINE_TOLERANCE
    };

    let mut merged: Vec<OcrLine> = Vec::new();
    let mut pending: Option<(String, f32, Bounds)> = None;

    for &(index, bounds) in row {
        let line = &originals[index];
        match pending.take() {
            Some((text, confidence, group))
                if same_line(group, bounds)
                    && (bounds.left - group.right) as f32
                        <= group.height().min(bounds.height()) as f32 * WORD_GAP_RATIO =>
            {
                pending = Some((
                    format!("{text} {}", line.text),
                    confidence.min(line.confidence),
                    group.union(bounds),
                ));
            }
            Some((text, confidence, group)) => {
                merged.push(OcrLine { text, confidence, polygon: group.to_polygon() });
                pending = Some((line.text.clone(), line.confidence, bounds));
            }
            None => pending = Some((line.text.clone(), line.confidence, bounds)),
        }
    }
    if let Some((text, confidence, group)) = pending {
        merged.push(OcrLine { text, confidence, polygon: group.to_polygon() });
    }
    merged
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
    /// `YYYY/MM/DD`, Gregorian — from the back page's `Year:`/`Month:`/
    /// `Day:` rows. Not ISO 8601 (`-`-separated); `/`-separated per the
    /// same convention as every other date field in this schema.
    pub date_of_birth_ad: Option<String>,
    /// `YYYY/MM/DD` in the Bikram Sambat calendar (not converted to AD) —
    /// from the front page's `साल`/`महिना`/`गते` rows.
    pub date_of_birth_bs: Option<String>,
    pub birth_district: Option<String>,
    /// The local body plus its *type*, as one string — "Aarubote VDC",
    /// "Urlabari Municipality", "badhaiyatal Rural Municipality". The type
    /// lives in the printed label rather than the value ("VDC : Aarubote"),
    /// so a bare "Aarubote" would silently lose which kind of local body it
    /// is — and cards issued before and after Nepal's 2017 local
    /// restructuring use different types for the same address, so it isn't
    /// derivable from the name alone.
    pub birth_municipal: Option<String>,
    pub birth_ward: Option<String>,
    pub permanent_district: Option<String>,
    /// See [`CitizenshipDocument::birth_municipal`].
    pub permanent_municipal: Option<String>,
    pub permanent_ward: Option<String>,
    pub father_name: Option<String>,
    pub mother_name: Option<String>,
    pub spouse_name: Option<String>,
    /// वंशज / जन्म / अंगीकृत / वैवाहिक अंगीकृत, as printed.
    pub citizenship_type: Option<String>,
    /// `YYYY/MM/DD` in Bikram Sambat, as printed.
    pub date_of_issue_bs: Option<String>,
}

const MONTH_ABBREVIATIONS: &[&str] = &[
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
];

/// Every local-body label this form is known to print, paired with the
/// canonical English type name to append to the value — see
/// [`CitizenshipDocument::birth_municipal`] for why the type has to be
/// carried into the value at all.
///
/// Order is load-bearing, longest/most-specific first: several of these
/// are substrings of each other ("न.पा" inside "म.न.पा" and "उ.म.न.पा";
/// "Municipality" inside "Rural Municipality"; "Metropolitan" inside
/// "Sub-Metropolitan"), so a shorter spelling tried first would match a
/// longer label's tail and mislabel a metropolitan city as a plain
/// municipality — while still extracting the right *name*, which makes it
/// exactly the kind of wrong that reads as correct.
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

/// Finds whichever local-body label `block` prints and returns its value
/// with the body type appended ("Aarubote" under a "VDC :" label becomes
/// `"Aarubote VDC"`).
fn local_body_in(block: &[OcrLine]) -> Option<String> {
    for (keyword, type_name) in LOCAL_BODY_LABELS {
        let Some(value) = value_after_keyword(block, keyword, &["वडा", "Ward"]) else { continue };
        let value = value.trim();
        // The label matched but its value didn't survive extraction — keep
        // looking rather than returning a bare type name with no place
        // attached to it. Checked on the *name alone*, before the type is
        // appended: a one-character misread ("ल") becomes a comfortably
        // long "ल Municipality" once suffixed, which then sails through
        // every downstream length check that would otherwise have caught
        // it. No real place name here is a single character.
        if value.chars().count() < 2 {
            continue;
        }
        return Some(format!("{value} {type_name}"));
    }
    None
}

/// The form's literal placeholder for "not applicable" (an unmarried
/// holder's spouse name/number) — not a real value.
const NOT_APPLICABLE: &str = "XXX";

pub fn extract(lines: &[OcrLine]) -> CitizenshipDocument {
    let mut doc = CitizenshipDocument::default();

    // "नाम थर" (the holder's own name) is also a substring of "बाबुको नाम
    // थर", "आमाको नाम थर", and "...को नामथर" (father/mother/spouse) — if
    // the holder's own line goes undetected (confirmed with a real scan:
    // one detection box spanned "आमाको नाम थर: फुलमाया अधिकारी : ना.प्र.नं:
    // ..." as a single line), the plain keyword search below would find
    // and attribute a *relative's* name instead. "ना.प्र.नं" (citizenship
    // number) has the identical problem — it's the same label reused for
    // the holder's own number and both parents' — so scope every
    // owner-only search (number, name, sex) to lines that aren't
    // themselves one of those relatives' rows, rather than trusting
    // reading order alone to put the holder's own row first.
    //
    // Fuzzy, not exact — this is a defensive exclusion, so a false exclude
    // just means the search tries another line (safe), while a false
    // *include* is the actual bug it exists to prevent. Confirmed on a real
    // scan: "बाबुको" (father's-name marker) misread as "बाबको" (missing a
    // vowel sign) slipped past an exact `.contains` check, leaving that
    // line in `owner_lines`; the fuzzy "नाम थर" search then matched inside
    // it and attributed the father's name as the holder's own.
    //
    // Truncates at the first relative marker rather than filtering out
    // only the lines that literally contain one — confirmed on the same
    // scan that a relative's *own* "ना.प्र.नं" (citizenship number) label
    // sits on its own detection box, separate from the "बाबुको नाम थर" box
    // that names whose section it's in. A per-line content filter can't
    // catch that box at all; it has no relative marker in its own text.
    // The form's layout always puts the holder's own fields first, then
    // father's, then mother's, then spouse's — so everything from the
    // first relative marker onward is fair game to drop, not just lines
    // that happen to mention one.
    //
    let owner_end = lines
        .iter()
        .position(|line| ["बाबुको", "आमाको", "पति"].iter().any(|kw| contains_keyword(&line.text, kw)))
        .unwrap_or(lines.len());
    let owner_lines: Vec<OcrLine> = lines[..owner_end].to_vec();

    // "ना.प्र.न" (no final anusvara) is tried after the full spelling as a
    // second chance, not instead of it: the recognizer drops or doubles a
    // character in this label often enough that the full form lands at
    // distance 2 and stops matching (a real scan read it as "नान.प्र.न.",
    // one inserted "न" *and* a "ं"->"." swap). The shorter form absorbs one
    // of those two errors into its own missing character, bringing the
    // same line back within distance 1.
    // Stop keywords are the labels printed immediately to the right of the
    // number on each layout — without them, a row that merged across the
    // column gap carries the neighbour's "Sex: Male" into this field.
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
        (Some(y), Some(m), Some(d)) => bs_date(&y, &m, &d),
        // The three parts print on one row ("साल: २०५६ महिना: ०७ गते: २३"),
        // so when only the day's label is the one that got mangled — "गते"
        // read as "वत्ते" on a real scan, too far off for fuzzy matching —
        // the digits are all still there in reading order. Falling back to
        // "year, month, day are the three numbers on the साल row" recovers
        // the whole date instead of dropping it for one bad label.
        _ => bs_date_from_row(lines),
    };

    // The short "स्थान"/"बासस्थान" spellings are fallbacks for when the
    // recognizer mangles the leading word of a block header badly enough
    // that even fuzzy matching misses it — a real scan read "जन्म स्थान"
    // as a bare "स्थान" (the "जन्म" landed in its own box) and "स्थायी
    // बासस्थान" as "्यायी बासस्थान". Losing a *block* header is much more
    // damaging than losing one label: every field inside the block goes
    // null at once, which is exactly what happened (district, local body
    // and ward all empty on a page where the recognizer had read each of
    // them perfectly).
    //
    // Safe despite "स्थान" also being a substring of "बासस्थान": the form
    // always prints birth place above permanent address, `block_between`
    // starts at the *first* match, and the permanent header is listed as
    // this block's end keyword — so the birth block still stops where the
    // permanent one begins.
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

    // "बासस्थान" alone as a last resort — same reasoning as the birth
    // block's "स्थान" fallback above.
    let permanent_block_devanagari = block_between(lines, "स्थायी बासस्थान", &["जन्म मिति", "बाबुको"])
        .none_if_empty()
        .or_else(|| block_between(lines, "स्थायी ठेगाना", &["जन्म मिति", "बाबुको"]).none_if_empty())
        .or_else(|| block_between(lines, "बासस्थान", &["जन्म मिति", "बाबुको"]).none_if_empty());
    let permanent_block_english = block_between(lines, "Permanent Address", &["नागरिकता", "Citizenship Type"]);
    let permanent_block = permanent_block_devanagari.unwrap_or(&[]);
    let permanent_block = if permanent_block.is_empty() { permanent_block_english } else { permanent_block };
    // Last resort: the form's layout is fixed — "District"/"जिल्ला" is
    // printed exactly twice on the page, once for birth place and once,
    // always immediately below it, for permanent address. So when none of
    // "Permanent Address"'s own spellings (checked above) matched
    // anything, don't give up on the fields entirely — the *second*
    // District cluster on the page is the permanent-address block no
    // matter what its own header read as.
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

    // The bare "बाबुको"/"आमाको" fallbacks cover the same split-header case
    // as the address blocks above: a real scan put "बाबुको" in its own
    // detection box with "नाम थर" in the next one, so the full header
    // never appeared on any single line and the father block came back
    // empty even though his name was read perfectly two boxes later.
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
    // "जारी मिति" (issue date) only ever labels this one field on the whole
    // certificate, unlike "नाम थर"/"दर्जा" above which need the officer
    // block's scope to avoid matching the holder's own name — so it's safe
    // to search the full page, not just the block. That block starts at
    // "जारी गर्ने अधिकारी", a label the recognizer frequently misreads
    // entirely on the low-contrast bottom-left region; when it does, the
    // block is empty and this field would otherwise go null even though the
    // "जारी मिति : ..." line itself is read fine.
    doc.date_of_issue_bs = value_after_keyword(officer_block, "जारी मिति", &[])
        .or_else(|| value_after_keyword(lines, "जारी मिति", &[]))
        .and_then(|v| parse_bs_date(v.trim()));

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
    doc.birth_municipal = clean_text(doc.birth_municipal);
    doc.permanent_district = clean_text(doc.permanent_district);
    doc.permanent_municipal = clean_text(doc.permanent_municipal);
    doc.father_name = clean_text(doc.father_name);
    doc.mother_name = clean_text(doc.mother_name);
    doc.spouse_name = clean_text(doc.spouse_name);
    doc.citizenship_type = clean_text(doc.citizenship_type);
    doc
}

fn clean_text(value: Option<String>) -> Option<String> {
    value.filter(|text| {
        let trimmed = text.trim();
        // Rust's `is_alphabetic` follows Unicode's derived Alphabetic
        // property, which — unlike a plain "is this a letter" check —
        // includes Devanagari's combining vowel signs and visarga, since
        // they're needed to form a complete syllable together with a base
        // consonant. That's correct for the property's own purpose, but
        // means a single stray combining mark like "ः" (visarga alone,
        // detached from any base character — confirmed on a real scan
        // where it survived as `mother_name`) passes an any-alphabetic
        // check on its own. Requiring at least 2 characters is a cheap
        // way to reject that without depending on a full Unicode
        // General-Category table this project doesn't otherwise need —
        // no real name/place/type value in this schema is ever one
        // character long.
        trimmed.chars().count() >= 2 && trimmed.chars().any(char::is_alphabetic)
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

fn combine_ad_date(year: Option<String>, month: Option<String>, day: Option<String>) -> Option<String> {
    let year: i32 = year?.trim().parse().ok()?;
    let month_upper = month?.trim().to_uppercase();
    let month_num = MONTH_ABBREVIATIONS.iter().position(|m| *m == month_upper)? as u32 + 1;
    let day: u32 = day?.trim().parse().ok()?;
    if !(1..=31).contains(&day) {
        return None;
    }
    Some(format!("{year:04}/{month_num:02}/{day:02}"))
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

/// Recovers a BS birth date from the row that carries it, using position
/// instead of labels: the "साल"/"महिना"/"गते" row prints exactly three
/// numbers in year-month-day order, so when one of the three labels is too
/// mangled to match, the digits themselves are still unambiguous. Requires
/// exactly three numbers on the row and a 4-digit year, so a row that
/// picked up an extra number from a neighbouring column is rejected rather
/// than silently reordered into a wrong date.
fn bs_date_from_row(lines: &[OcrLine]) -> Option<String> {
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

/// Formats a Bikram Sambat date, rejecting component values the calendar
/// cannot produce. A misread digit otherwise yields a confidently wrong
/// but well-formed date — a real scan read the month of "साल: २०६० महिना:
/// १०" as `90`, giving "2060/90/26", which is worse than returning
/// nothing because it looks parseable to whatever consumes it. BS months
/// run 1-12 and days 1-32 (the longest BS month has 32 days).
fn bs_date(year: &str, month: &str, day: &str) -> Option<String> {
    let month_num: u32 = month.parse().ok()?;
    let day_num: u32 = day.parse().ok()?;
    if !(1..=12).contains(&month_num) || !(1..=32).contains(&day_num) {
        return None;
    }
    Some(format!("{year}/{month_num:02}/{day_num:02}"))
}

/// Parses a Bikram Sambat date's digits out of raw OCR text and reformats
/// as `YYYY/MM/DD` — deliberately not just digit-converting and passing
/// the source punctuation through, since the recognizer doesn't read the
/// separator reliably: "जारी मिति : २०७३-०१-०५" has come back with a
/// period substituted for one of the dashes ("2073.01-05"), which a
/// caller then has no consistent way to parse. Extracting exactly the 8
/// digits (4 for year, 2 each for month/day) and discarding whatever
/// separator surrounded them sidesteps that misread entirely.
fn parse_bs_date(text: &str) -> Option<String> {
    let digits: String = devanagari_digits_to_ascii(text).chars().filter(char::is_ascii_digit).collect();
    if digits.len() != 8 {
        return None;
    }
    Some(format!("{}/{}/{}", &digits[0..4], &digits[4..6], &digits[6..8]))
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
/// Standard Levenshtein distance (character-level, not byte-level — matters
/// for Devanagari, where one visible glyph is often 3 UTF-8 bytes).
fn edit_distance(a: &[char], b: &[char]) -> usize {
    let (n, m) = (a.len(), b.len());
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// Extra chars of match-window slack reserved purely for spurious
/// whitespace the recognizer inserts mid-word — since those cost nothing
/// against `max_distance` (see `find_keyword`), the window has to be able
/// to grow past `keyword.len() + max_distance` to actually capture them.
/// Sized generously: a real scan produced "Citize ns hip" for
/// "Citizenship", three extra spaces inside one 12-character word.
const WHITESPACE_SLACK: usize = 6;

/// Finds `keyword` in `text` allowing up to `max_distance` character-level
/// errors, *not counting whitespace*. Returns the byte range `[start, end)`
/// of the best match, so the caller can slice `text` after it same as an
/// exact match would.
///
/// Whitespace is stripped from both sides before the edit distance is
/// computed — real scans routinely split a label across stray spaces
/// ("Citize nship", "Permane nt", "Dis trict") often more than one per
/// word, and charging those against the same one-error budget as a genuine
/// wrong letter meant most of them were never going to fit. A dropped or
/// inserted space changes nothing about which word was printed, so it
/// isn't a meaningful error to tolerate for — it's closer to noise than to
/// a typo.
fn find_keyword(text: &str, keyword: &str, max_distance: usize) -> Option<(usize, usize)> {
    let keyword_chars: Vec<char> = keyword.chars().filter(|c| !c.is_whitespace()).collect();
    let text_chars: Vec<char> = text.chars().collect();
    let klen = keyword_chars.len();
    if klen == 0 {
        return None;
    }
    let min_window = klen.saturating_sub(max_distance).max(1);
    let max_window = klen + max_distance + WHITESPACE_SLACK;

    // Real single-character typos are overwhelmingly substitutions (same
    // length as the keyword), not insertions/deletions — confirmed against
    // a real scan: "R.M" (the municipality-label keyword) against a line
    // OCR'd as "R.N." found *two* equally-scored distance-1 candidates,
    // "R." (a 2-char window, keyword with "M" deleted) and "R.N" (a 3-char
    // window, "M"->"N" substituted). Preferring whichever was found first
    // picked the deletion match, leaving "N." dangling off the end of the
    // keyword instead of the label, and "N." then got read as the field's
    // *value*. Break ties toward the window closest to the keyword's own
    // length so the substitution reading wins. Ties are compared against
    // `klen` (the whitespace-stripped length) for the same reason the
    // window itself now needs the slack above: a window padded out with
    // free whitespace is not actually "further" from the keyword.
    let mut best: Option<(usize, usize, usize)> = None;
    for start in 0..text_chars.len() {
        for window in min_window..=max_window {
            let end = start + window;
            if end > text_chars.len() {
                break;
            }
            let stripped: Vec<char> =
                text_chars[start..end].iter().filter(|c| !c.is_whitespace()).copied().collect();
            let distance = edit_distance(&stripped, &keyword_chars);
            if distance > max_distance {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, _, best_distance)) if distance < best_distance => true,
                Some((best_start, best_end, best_distance)) if distance == best_distance => {
                    let best_window = best_end - best_start;
                    window.abs_diff(klen) < best_window.abs_diff(klen)
                }
                _ => false,
            };
            if better {
                best = Some((start, end, distance));
            }
        }
    }

    best.map(|(start, end, _)| {
        let byte_start: usize = text_chars[..start].iter().map(|c| c.len_utf8()).sum();
        let byte_end: usize = text_chars[..end].iter().map(|c| c.len_utf8()).sum();
        (byte_start, byte_end)
    })
}

/// `find_keyword`, but at a distance chosen by the keyword's own length
/// rather than passed in by the caller — under 4 characters stays exact
/// (distance 0), everything else allows one error (distance 1). Below 4
/// characters a single edit is too cheap relative to the keyword's own
/// length to mean anything: confirmed on a real scan that "पति" (3 chars,
/// the spouse-name marker) fuzzy-matches "पत" inside "प्रमाणपत्र"
/// ("...Certificate", part of the certificate's own printed title,
/// present on every scan) at distance 1. Shared by every caller that
/// tests "does this line contain roughly this keyword" rather than
/// extracting a value after it — `block_between`'s start/end markers and
/// the owner/relative-row split both need the identical safety margin
/// `value_after_keyword` gets from its own length-independent tolerance,
/// and re-deriving the threshold separately at each call site is how it
/// would drift out of sync.
fn contains_keyword(text: &str, keyword: &str) -> bool {
    let max_distance = if keyword.chars().count() < 4 { 0 } else { 1 };
    find_keyword(text, keyword, max_distance).is_some()
}

/// `value_after_keyword`'s search, run once per `max_distance` — see that
/// function's doc comment for why exact matches always win over fuzzy ones
/// across the *whole* document, not just within one line.
fn value_after_keyword_pass(
    lines: &[OcrLine],
    keyword: &str,
    stop_keywords: &[&str],
    max_distance: usize,
) -> Option<String> {
    for (index, line) in lines.iter().enumerate() {
        let text = &line.text;
        // A line that's itself the certificate's own boilerplate can never
        // be a real field label — skip it as a match candidate entirely
        // rather than relying on the extracted value happening to fail
        // the same noise check afterward. Matters once fuzzy tolerance is
        // loose enough that a boilerplate sentence can score close to a
        // real (but badly misread) label — see NOISE_PHRASES.
        if looks_like_noise(text) {
            continue;
        }
        let Some((_start, kw_end)) = find_keyword(text, keyword, max_distance) else { continue };
        let after_kw = &text[kw_end..];
        let stop_pos = stop_keywords.iter().filter_map(|kw| after_kw.find(kw)).min().unwrap_or(after_kw.len());
        let scope = &after_kw[..stop_pos];
        let value = match scope.find(':') {
            Some(colon_pos) => &scope[colon_pos + 1..],
            None => scope,
        };
        let value = value.trim().trim_matches(|c: char| c == '.' || c == '*' || c.is_whitespace());
        if !value.is_empty() && !looks_like_noise(value) {
            return Some(value.to_owned());
        }
        if let Some(value) = nearest_value_near(lines, index, stop_keywords) {
            return Some(value);
        }
    }
    None
}

/// Finds `keyword`'s value the same way the rest of this module always
/// has (exact substring, same-line or nearest-neighbor) — but if that
/// finds nothing *anywhere in the document*, retries with progressively
/// more character-level error tolerance in the keyword match (1, then 2).
/// The recognizer routinely mangles an otherwise-correct label —
/// confirmed against real scans: "Citizenship" read as "Citize nship" (one
/// inserted space) and, worse, as "Citize ns hip Certificate No." (two
/// inserted spaces); "Year" as "Ycar"/"Yenr"; "District" as
/// "Distriet"/missing its final "t" — and an exact-substring match treats
/// each of those as if the label were never printed at all, even though
/// the value sitting right next to it was read correctly.
///
/// Each tolerance level is tried across *every* line before the next
/// (looser) level is tried on *any* line — not per-line — because a short
/// keyword's fuzzy match can land on an unrelated word at the same edit
/// distance as a real typo: searching for "Day" fuzzy-matched "Dat" in
/// "Date of Birth (AD): ..." (distance 1, identical to "Year"~"Ycar"'s
/// real fix) on a line that happened to come before the genuine "Day:09"
/// line, stealing the field. Requiring no distance-1 match to exist
/// *anywhere* before distance-2 is allowed *anywhere* (and so on) means
/// the real, closer match always wins that race. `value_after_keyword_pass`
/// also refuses to match a line that's the certificate's own boilerplate
/// (see `looks_like_noise`) — needed once tolerance reaches 2, since by
/// then a long keyword can score close against a long *wrong* sentence
/// that happens to share most of its words.
fn value_after_keyword(lines: &[OcrLine], keyword: &str, stop_keywords: &[&str]) -> Option<String> {
    // Tried a third tier at distance 2 (to catch e.g. "Citize ns hip
    // Certificate No.", two inserted spaces) — reverted. Tested against
    // five real scans and it introduced more new wrong-value regressions
    // (permanent_district/birth_municipal matching the wrong block on
    // scans that were correct at distance-1) than the one field it fixed.
    // Distance 1 stays the ceiling until a fix can raise it without that
    // trade — the noise-line pre-filter and the distance-1 fuzzy pass
    // below still apply either way.
    (0..=1).find_map(|max_distance| value_after_keyword_pass(lines, keyword, stop_keywords, max_distance))
}

/// Substrings from the certificate's own repeating background watermark
/// and page boilerplate — never legitimately part of a field's *value*
/// (only its labels/headers), so a value containing one is a misread of
/// the watermark bleeding through, not real content.
// Boilerplate/watermark substrings that are never legitimately part of a
// field's *value* — only its labels, headers, or standard printed notices.
// A value containing one is a misread bleeding in from elsewhere on the
// page, not real content. Note these are checked against the *extracted
// value*, i.e. whatever's left after the matched keyword itself is
// stripped off — so a phrase here needs to survive that stripping no
// matter which keyword-length prefix of it a search happened to consume.
// The submission-notice entries below are a real example: a keyword search
// for "जिल्ला प्रशासन कार्यालय" (issuing office) or even just "जिल्ला"
// (birth/permanent district) can both land on the standard "submit to the
// district administration or police office" footer instead of a real
// office/district — the "जिल्ला प्रशासन" phrase in the *source line*
// doesn't survive being the matched keyword, so only the boilerplate's
// unconsumed remainder ("...office or police office, kindly submit") is
// listed here.
const NOISE_PHRASES: &[&str] = &[
    "नागरिकताको प्रमाण",
    "नागरिकता ऐन",
    "नेपाली नागरिकता",
    "नेपाल सरकार",
    "गृह मन्त्रालय",
    "जिल्ला प्रशासन",
    "कार्यालयमा वा प्रहरी कार्यालयमा",
    "बुझाईदिनुहोला",
    // The back page's fixed English preamble sentence — never a field
    // label or value, but "...issued this Citizenship Certificate with
    // following details" shares enough of "Citizenship Certificate No"
    // that it's a real fuzzy-match risk for that keyword specifically.
    "issued this",
    "following details",
];

fn looks_like_noise(text: &str) -> bool {
    NOISE_PHRASES.iter().any(|phrase| text.contains(phrase))
}

/// A ward number is 1-2 plain digits — nothing else a real value could be.
/// Guards `birth_ward`/`permanent_ward` against a garbled OCR line (words,
/// punctuation, or an unrelated nearby run of digits) surviving just
/// because *some* text followed the "वडा"/"Ward" keyword.
fn is_plausible_ward_number(text: &str) -> bool {
    !text.is_empty() && text.len() <= 2 && text.chars().all(|c| c.is_ascii_digit())
}

/// Pulls the ward digits out of whatever followed the "वडा"/"Ward"
/// keyword. The label's own abbreviation often has no colon after it
/// ("वडा नं. २", "Ward No.2"), so the remainder still carries "नं."/"No."
/// ahead of the number and fails a plain digits-only check — while the
/// digits themselves are perfectly readable. Keeping only the digits and
/// then applying [`is_plausible_ward_number`] recovers those without
/// loosening what counts as a valid ward.
fn ward_number(value: &str) -> Option<String> {
    // The label's own abbreviation has to come off *before* the
    // lookalike mapping below, not after: "No" ends in an "o", which that
    // mapping turns into a zero, so "Ward No.2" came out as "02" — a
    // plausible-looking two-digit ward that is simply wrong.
    let value = devanagari_digits_to_ascii(value);
    let value = value.replace("No", "").replace("no", "").replace("नं", "");

    // A ward value is a single short token once the label and its
    // punctuation are gone. Requiring that *before* the lookalike mapping
    // below is what keeps the mapping honest: it only ever runs on
    // something already shaped like a ward number, never on leftover
    // prose. Without this the fuzzy label search — which matched "Ward"
    // against "Bard" inside "District: Bardiya", one edit away — handed
    // "iya" to the mapping, whose "i"->1 rule turned a district name into
    // a confident, completely wrong ward of "1".
    let core: String = value.chars().filter(|c| c.is_alphanumeric()).collect();
    if core.is_empty() || core.chars().count() > 2 {
        return None;
    }

    // Letters that are visually the same glyph as a digit in this form's
    // print, mapped back. Confirmed on real scans: "Ward No.1" read as
    // "Ward No.l" and "Ward No.5" as "Ward No.S", both of which otherwise
    // leave the field empty despite the number being perfectly legible.
    // Only safe *because* a ward is digits-only — the same substitution
    // applied to a name or place would corrupt real letters, so it stays
    // scoped to this one field rather than living in a shared cleanup.
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
/// Fuzzy for the same reason `value_after_keyword` is — confirmed on a
/// real scan that an exact-only version of this function missed
/// "Permanent Address" entirely because it read as "Permane nt Address:"
/// (one inserted space), silently emptying the whole permanent-address
/// block and every field sourced from it, not just one misread value.
fn block_between<'a>(lines: &'a [OcrLine], start_keyword: &str, end_keywords: &[&str]) -> &'a [OcrLine] {
    let Some(start) = lines.iter().position(|line| contains_keyword(&line.text, start_keyword)) else {
        return &[];
    };
    let end = lines[start + 1..]
        .iter()
        .position(|line| end_keywords.iter().any(|kw| contains_keyword(&line.text, kw)))
        .map(|offset| start + 1 + offset)
        .unwrap_or(lines.len());
    &lines[start..end]
}

/// Positional counterpart to `block_between`, for when a block's own header
/// can't be matched under any spelling at all. Rather than depend on any
/// particular label text, this finds the *second* line matching any of
/// `keywords` and returns the block starting there — since on this
/// certificate's fixed layout, "District"/"जिल्ला" (and every other field
/// this is used for) prints exactly twice: once inside birth place, then
/// again, always positioned right after it, inside permanent address.
/// Ends at whichever `end_keywords` line comes first after that, or the end
/// of `lines`.
fn second_occurrence_block<'a>(lines: &'a [OcrLine], keywords: &[&str], end_keywords: &[&str]) -> &'a [OcrLine] {
    let mut seen = 0;
    for (start, line) in lines.iter().enumerate() {
        if !keywords.iter().any(|kw| contains_keyword(&line.text, kw)) {
            continue;
        }
        seen += 1;
        if seen < 2 {
            continue;
        }
        let end = lines[start + 1..]
            .iter()
            .position(|line| end_keywords.iter().any(|kw| contains_keyword(&line.text, kw)))
            .map(|offset| start + 1 + offset)
            .unwrap_or(lines.len());
        return &lines[start..end];
    }
    &[]
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
    (!value.is_empty() && !looks_like_noise(value)).then(|| value.to_owned())
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

    /// Smallest box covering both — the merged bounds of two boxes joined
    /// into one line (see `merge_adjacent`).
    fn union(self, other: Self) -> Self {
        Self {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }

    /// Back to an `OcrLine` polygon, corners in the same clockwise-from-
    /// top-left order the detector emits.
    fn to_polygon(self) -> [[i32; 2]; 4] {
        [
            [self.left, self.top],
            [self.right, self.top],
            [self.right, self.bottom],
            [self.left, self.bottom],
        ]
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
        // "Permanent Address" recognized as complete noise (no keyword,
        // fuzzy or otherwise, can match it) — the label box in the image
        // this is modeled on. District/R.M./Ward still read fine right
        // below Birth Place, so the positional fallback in `block_after`
        // should still recover them instead of leaving all three null.
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
        // Birth block must stay untouched by the fallback.
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
