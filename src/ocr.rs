use std::cmp::Ordering;
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::error::AppError;

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8080/ocr";
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;
const DEFAULT_QUEUE_TIMEOUT_MILLIS: u64 = 2_000;
const DEFAULT_MAX_CONCURRENCY: usize = 1;
const DEFAULT_MIN_CONFIDENCE: f32 = 0.45;
const FIELD_LABEL_MIN_CONFIDENCE: f32 = 0.90;
const FIELD_VALUE_MIN_CONFIDENCE: f32 = 0.75;
const MAX_FIELD_ALIGNMENT_DRIFT: i32 = 120;

// Printed ID card values (names, nationalities) are routinely all-caps too,
// so "all uppercase ASCII" can't distinguish a label from a value — e.g. the
// printed name "ROSHAN JOSHI" is indistinguishable by case alone from the
// label "FULL NAME". Gate on a known label vocabulary instead.
const LABEL_WORDS: &[&str] = &[
    "NATIONAL",
    "IDENTITY",
    "CARD",
    "NATIONALITY",
    "FULL",
    "NAME",
    "DATE",
    "OF",
    "ISSUE",
    "BIRTH",
    "SEX",
    "PERMANENT",
    "ADDRESS",
    "CITIZENSHIP",
    "TYPE",
    "NUMBER",
    "CC",
    "NICIN",
    "ID",
    "NO",
];

#[derive(Debug, PartialEq)]
pub(crate) struct OcrLine {
    pub(crate) text: String,
    pub(crate) confidence: f32,
    pub(crate) polygon: [[i32; 2]; 4],
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct Field {
    pub label: String,
    pub value: String,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct OcrDocument {
    pub fields: Vec<Field>,
}

pub struct PaddleOcrClient {
    http: Client,
    endpoint: Url,
    api_key: Option<String>,
    permits: Semaphore,
    queue_timeout: Duration,
    min_confidence: f32,
}

impl PaddleOcrClient {
    pub fn from_env() -> Result<Self, AppError> {
        let endpoint = std::env::var("PADDLE_OCR_URL")
            .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_owned())
            .parse::<Url>()
            .map_err(|error| AppError::OcrConfig(format!("invalid PADDLE_OCR_URL: {error}")))?;
        let request_timeout = Duration::from_secs(env_number(
            "PADDLE_OCR_TIMEOUT_SECS",
            DEFAULT_REQUEST_TIMEOUT_SECS,
        )?);
        let queue_timeout = Duration::from_millis(env_number(
            "PADDLE_OCR_QUEUE_TIMEOUT_MS",
            DEFAULT_QUEUE_TIMEOUT_MILLIS,
        )?);
        let max_concurrency = env_number("PADDLE_OCR_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)?;
        if max_concurrency == 0 {
            return Err(AppError::OcrConfig(
                "PADDLE_OCR_MAX_CONCURRENCY must be greater than zero".to_owned(),
            ));
        }
        let min_confidence = env_number("PADDLE_OCR_MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE)?;
        if !(0.0..=1.0).contains(&min_confidence) {
            return Err(AppError::OcrConfig(
                "PADDLE_OCR_MIN_CONFIDENCE must be between 0 and 1".to_owned(),
            ));
        }

        Self::new(
            endpoint,
            std::env::var("PADDLE_OCR_API_KEY").ok(),
            max_concurrency,
            request_timeout,
            queue_timeout,
            min_confidence,
        )
    }

    fn new(
        endpoint: Url,
        api_key: Option<String>,
        max_concurrency: usize,
        request_timeout: Duration,
        queue_timeout: Duration,
        min_confidence: f32,
    ) -> Result<Self, AppError> {
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .timeout(request_timeout)
            .pool_max_idle_per_host(max_concurrency)
            .build()
            .map_err(AppError::OcrTransport)?;

        Ok(Self {
            http,
            endpoint,
            api_key,
            permits: Semaphore::new(max_concurrency),
            queue_timeout,
            min_confidence,
        })
    }

    pub async fn extract(&self, image: &[u8]) -> Result<OcrDocument, AppError> {
        let _permit = tokio::time::timeout(self.queue_timeout, self.permits.acquire())
            .await
            .map_err(|_| AppError::OcrQueueTimeout)?
            .map_err(|_| AppError::OcrUnavailable)?;

        // Paddle's service contract requires Base64 in JSON. This is the only
        // full-image encoding allocation; the source image and encoded string
        // are borrowed everywhere else in the request path.
        let encoded_len = image
            .len()
            .checked_add(2)
            .and_then(|len| len.checked_div(3))
            .and_then(|len| len.checked_mul(4))
            .ok_or(AppError::OcrInputTooLarge)?;
        let mut encoded = String::with_capacity(encoded_len);
        BASE64.encode_string(image, &mut encoded);

        let payload = PaddleRequest {
            file: &encoded,
            file_type: 1,
            use_doc_orientation_classify: false,
            use_doc_unwarping: false,
            use_textline_orientation: true,
            text_rec_score_thresh: self.min_confidence,
            visualize: false,
        };

        let mut request = self.http.post(self.endpoint.as_str()).json(&payload);
        if let Some(api_key) = self.api_key.as_deref() {
            request = request.bearer_auth(api_key);
        }

        let response = request.send().await.map_err(AppError::OcrTransport)?;
        if !response.status().is_success() {
            return Err(AppError::OcrRejected(response.status().as_u16()));
        }

        let response = response
            .json::<PaddleResponse>()
            .await
            .map_err(AppError::OcrTransport)?;
        if response.error_code != 0 {
            return Err(AppError::OcrRejected(response.error_code));
        }
        let result = response.result.ok_or(AppError::OcrProtocol(
            "PP-OCRv5 response did not contain a result",
        ))?;

        let mut lines = Vec::new();
        for page in result.ocr_results {
            let PaddlePrunedResult {
                rec_texts,
                rec_scores,
                rec_polys,
            } = page.pruned_result;
            if rec_texts.len() != rec_scores.len() || rec_texts.len() != rec_polys.len() {
                return Err(AppError::OcrProtocol(
                    "PP-OCRv5 returned mismatched text, score, and polygon counts",
                ));
            }
            lines.reserve(rec_texts.len());
            for ((text, confidence), polygon) in
                rec_texts.into_iter().zip(rec_scores).zip(rec_polys)
            {
                let text = text.trim();
                if !text.is_empty() && confidence >= self.min_confidence {
                    lines.push(OcrLine {
                        text: text.to_owned(),
                        confidence,
                        polygon,
                    });
                }
            }
        }

        lines.sort_by(reading_order);
        let fields = parse_fields(&lines);

        Ok(OcrDocument { fields })
    }
}

fn env_number<T>(name: &'static str, default: T) -> Result<T, AppError>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(value) => value
            .parse::<T>()
            .map_err(|error| AppError::OcrConfig(format!("invalid {name}: {error}"))),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(AppError::OcrConfig(format!("invalid {name}: {error}"))),
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PaddleRequest<'a> {
    file: &'a str,
    file_type: u8,
    use_doc_orientation_classify: bool,
    use_doc_unwarping: bool,
    use_textline_orientation: bool,
    text_rec_score_thresh: f32,
    visualize: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PaddleResponse {
    error_code: u16,
    result: Option<PaddleResult>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PaddleResult {
    #[serde(default)]
    ocr_results: Vec<PaddlePage>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PaddlePage {
    pruned_result: PaddlePrunedResult,
}

#[derive(Deserialize)]
struct PaddlePrunedResult {
    #[serde(default)]
    rec_texts: Vec<String>,
    #[serde(default)]
    rec_scores: Vec<f32>,
    #[serde(default)]
    rec_polys: Vec<[[i32; 2]; 4]>,
}

pub(crate) fn reading_order(a: &OcrLine, b: &OcrLine) -> Ordering {
    let a_top = a.polygon.iter().map(|point| point[1]).min().unwrap_or(0);
    let b_top = b.polygon.iter().map(|point| point[1]).min().unwrap_or(0);
    let a_left = a.polygon.iter().map(|point| point[0]).min().unwrap_or(0);
    let b_left = b.polygon.iter().map(|point| point[0]).min().unwrap_or(0);
    a_top.cmp(&b_top).then_with(|| a_left.cmp(&b_left))
}

/// What shape a field's value is expected to have — lets [`extract_fields`]
/// pick the right line(s) out of a block or a whole-document scan without a
/// bespoke function per field.
#[derive(Clone, Copy)]
enum ValueShape {
    /// Non-numeric Devanagari text (a name, written in script).
    Devanagari,
    /// Contains at least one lowercase ASCII letter — distinguishes real
    /// English prose (an address) from an ALL-CAPS label on the same card.
    LowercaseText,
    /// A dash/dot/slash/space-separated digit run shaped like an ID number
    /// (`digit_group_lengths` 3-5 groups) — see [`is_identity_number`].
    IdentityNumber,
    /// A `YYYY-MM-DD`-shaped digit run — see [`is_date`].
    Date,
    /// Exact (trimmed) match against one of a fixed set of literal tokens,
    /// e.g. Nepali sex markers (पुरुष/महिला/...) that have no shared prefix
    /// with any label and so can't be found via keyword search.
    Enum(&'static [&'static str]),
}

fn matches_shape(text: &str, shape: ValueShape) -> bool {
    match shape {
        ValueShape::Devanagari => {
            contains_devanagari(text) && !text.chars().any(|character| character.is_numeric())
        }
        ValueShape::LowercaseText => text.chars().any(|character| character.is_ascii_lowercase()),
        ValueShape::IdentityNumber => is_identity_number(text),
        ValueShape::Date => is_date(text),
        ValueShape::Enum(options) => options.contains(&text.trim()),
    }
}

/// An optional multi-line scan: every line between the first line
/// containing `label_keyword` and the next line containing `stop_keyword`
/// that matches `shape`, joined in left-to-right order. Generalizes what
/// used to be two near-identical hand-written loops (the Devanagari name
/// block between "नाम" and "FULL NAME", and the ID-number block between
/// "परिचय" and "नाम").
/// How a block's matched lines combine into one value: `Concat` treats them
/// as pieces of one longer value (a given name + a surname, space-joined in
/// reading order); `Unique` treats them as alternate script/format
/// representations of the *same* value (a Devanagari ID number line and its
/// Latin-digit twin — deduped and " / "-joined, same as `join_unique`).
#[derive(Clone, Copy, PartialEq)]
enum BlockJoin {
    Concat,
    Unique,
}

struct FieldBlock {
    label_keyword: &'static str,
    stop_keyword: &'static str,
    shape: ValueShape,
    join: BlockJoin,
}

/// Declares how to find one output field's value. A document type is just a
/// list of these — adding a new document type is adding data, not a new
/// function.
struct FieldSignature {
    output_label: &'static str,
    /// Case-insensitive label keywords tried against the generic
    /// label/value parser first, then against a looser keyword-anchored
    /// nearest-line-below search if the generic parser missed it.
    english_keywords: &'static [&'static str],
    block: Option<FieldBlock>,
    /// A whole-document fallback/merge scan for this shape — used both as
    /// a last resort when no label anchor was found at all (ID number) and
    /// as the *only* source for values with no fixed label of their own
    /// (a birth date is just "some other date-shaped line that isn't the
    /// issue date").
    global_shape: Option<ValueShape>,
}

struct DocumentType {
    /// Keywords scored for document-type detection (case-insensitive
    /// substring match, Latin or Devanagari). Scored rather than
    /// all-required so one garbled/missing anchor line doesn't demote the
    /// whole document to the generic parser and lose every field below.
    detect_keywords: &'static [&'static str],
    min_detect_matches: usize,
    fields: &'static [FieldSignature],
}

const NATIONAL_IDENTITY_CARD: DocumentType = DocumentType {
    detect_keywords: &["IDENTITY", "NATIONALITY", "FULL NAME", "परिचय", "राष्ट्रिय"],
    min_detect_matches: 2,
    fields: &[
        FieldSignature {
            output_label: "NATIONAL ID NUMBER",
            english_keywords: &[],
            block: Some(FieldBlock {
                label_keyword: "परिचय",
                stop_keyword: "नाम",
                shape: ValueShape::IdentityNumber,
                join: BlockJoin::Unique,
            }),
            global_shape: Some(ValueShape::IdentityNumber),
        },
        FieldSignature {
            output_label: "NATIONALITY",
            english_keywords: &["NATIONALITY"],
            block: None,
            global_shape: None,
        },
        FieldSignature {
            output_label: "FULL NAME",
            english_keywords: &["FULL NAME"],
            block: Some(FieldBlock {
                label_keyword: "नाम",
                stop_keyword: "FULL NAME",
                shape: ValueShape::Devanagari,
                join: BlockJoin::Concat,
            }),
            global_shape: None,
        },
        FieldSignature {
            output_label: "DATE OF ISSUE",
            english_keywords: &["DATE OF ISSUE"],
            block: None,
            global_shape: None,
        },
        FieldSignature {
            output_label: "DATE OF BIRTH",
            english_keywords: &[],
            block: None,
            global_shape: Some(ValueShape::Date),
        },
        FieldSignature {
            output_label: "SEX",
            english_keywords: &["SEX"],
            block: None,
            global_shape: Some(ValueShape::Enum(&["पुरुष", "पुरूष", "महिला", "अन्य"])),
        },
    ],
};

const CITIZENSHIP_CARD: DocumentType = DocumentType {
    // Real card says "CC NUMBER" ("नागरिकता नं. | CC NUMBER"), not "ICC
    // NUMBER" — a wrong assumption baked into the original test fixture
    // that silently demoted every real citizenship card to generic parsing
    // (never met the 2-keyword detection threshold). Keep "ICC NUMBER" too
    // in case some other card genuinely uses it.
    detect_keywords: &["CITIZENSHIP TYPE", "CC NUMBER", "ICC NUMBER"],
    min_detect_matches: 2,
    fields: &[
        FieldSignature {
            output_label: "PERMANENT ADDRESS",
            english_keywords: &["PERMANENT ADDRESS"],
            block: Some(FieldBlock {
                label_keyword: "PERMANENT ADDRESS",
                stop_keyword: "CITIZENSHIP TYPE",
                shape: ValueShape::LowercaseText,
                join: BlockJoin::Concat,
            }),
            global_shape: None,
        },
        FieldSignature {
            output_label: "NICIN",
            english_keywords: &["NICIN"],
            block: None,
            global_shape: None,
        },
        FieldSignature {
            output_label: "CITIZENSHIP TYPE",
            english_keywords: &["CITIZENSHIP TYPE"],
            block: None,
            global_shape: None,
        },
        FieldSignature {
            output_label: "CITIZENSHIP NUMBER",
            english_keywords: &["CC NUMBER", "ICC NUMBER"],
            block: None,
            global_shape: None,
        },
    ],
};

const DOCUMENT_TYPES: &[DocumentType] = &[NATIONAL_IDENTITY_CARD, CITIZENSHIP_CARD];

pub(crate) fn parse_fields(lines: &[OcrLine]) -> Vec<Field> {
    let generic = parse_generic_fields(lines);
    for document_type in DOCUMENT_TYPES {
        let matches = document_type
            .detect_keywords
            .iter()
            .filter(|keyword| has_line_ci(lines, keyword))
            .count();
        if matches >= document_type.min_detect_matches {
            return extract_fields(lines, &generic, document_type.fields);
        }
    }
    generic
}

/// Runs every [`FieldSignature`] in `signatures` against `lines`, in order.
/// A raw line already consumed by an earlier match (a block, or an English
/// keyword lookup) is excluded from later shape scans for the *same*
/// field — so a whole-document fallback scan (ID number) doesn't re-find
/// and duplicate lines its own labeled block already picked up, and so an
/// unrelated field's whole-document scan (birth date) can't re-claim a line
/// another field already claimed (the issue date).
fn extract_fields(lines: &[OcrLine], generic: &[Field], signatures: &[FieldSignature]) -> Vec<Field> {
    let mut fields = Vec::with_capacity(signatures.len());
    let mut used_values: Vec<String> = Vec::new();

    for signature in signatures {
        let mut parts: Vec<String> = Vec::new();

        if let Some(block) = &signature.block
            && let Some((text, consumed)) = collect_block(lines, block)
        {
            used_values.extend(consumed);
            parts.push(text);
        }

        for keyword in signature.english_keywords {
            if let Some(value) = field_value(generic, keyword)
                .map(str::to_owned)
                .or_else(|| relaxed_field_value(lines, &[keyword]))
            {
                used_values.push(value.clone());
                parts.push(value);
            }
        }

        if let Some(shape) = signature.global_shape
            && let Some(text) = join_unique(
                lines
                    .iter()
                    .map(|line| line.text.as_str())
                    .filter(|text| matches_shape(text, shape))
                    .filter(|text| !used_values.iter().any(|used| used == text)),
            )
        {
            parts.push(text);
        }

        // Devanagari-script parts consistently come first, English second
        // (matches how these cards read: script label/value above, Latin
        // transliteration below) — not per-field special-casing, just a
        // property of the final string regardless of which mechanism
        // (block vs. keyword vs. shape scan) happened to find each part.
        parts.sort_by_key(|part| !contains_devanagari(part));

        if let Some(value) = join_unique(parts.iter().map(String::as_str)) {
            push_owned_field(&mut fields, signature.output_label, Some(value));
        }
    }

    fields
}

/// Every line between the first line containing `block.label_keyword` and
/// the next line containing `block.stop_keyword` that matches
/// `block.shape`, combined per `block.join`. Returns the combined text
/// alongside the individual raw line texts it consumed, so the caller can
/// exclude them from any later fallback scan for the same field.
fn collect_block(lines: &[OcrLine], block: &FieldBlock) -> Option<(String, Vec<String>)> {
    let label = lines
        .iter()
        .position(|line| line.text.contains(block.label_keyword))?;
    let mut indexes = lines[label + 1..]
        .iter()
        .take_while(|line| !line.text.contains(block.stop_keyword))
        .enumerate()
        .filter(|(_, line)| matches_shape(&line.text, block.shape))
        .map(|(offset, _)| label + 1 + offset)
        .collect::<Vec<_>>();
    if block.join == BlockJoin::Concat {
        // Left-to-right order matters for pieces of one value spread
        // across the same row (a given name and surname side by side).
        // Alternate-script/format lines (BlockJoin::Unique) are typically
        // stacked vertically instead, where their discovery order (top to
        // bottom, already how `indexes` was built) is the meaningful one —
        // re-sorting by x there just introduces order flakiness from
        // near-identical x-coordinates.
        indexes.sort_unstable_by_key(|&index| line_left(&lines[index]));
    }

    let consumed: Vec<String> = indexes.iter().map(|&index| lines[index].text.clone()).collect();
    let text = match block.join {
        BlockJoin::Concat => join_line_text(lines, &indexes),
        BlockJoin::Unique => join_unique(consumed.iter().map(String::as_str)).unwrap_or_default(),
    };
    (!text.is_empty()).then_some((text, consumed))
}

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
                (
                    vertical_gap_from(bounds, candidate),
                    left_distance(bounds.left, candidate),
                )
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

fn has_line_ci(lines: &[OcrLine], needle: &str) -> bool {
    let needle = needle.to_uppercase();
    lines
        .iter()
        .any(|line| line.text.to_uppercase().contains(&needle))
}

/// Locates a field's value even when the strict spatial grouping in
/// `parse_generic_fields` dropped it (label confidence below
/// `FIELD_LABEL_MIN_CONFIDENCE`, or alignment drift past
/// `MAX_FIELD_ALIGNMENT_DRIFT`). Used only for known NID field keywords, so
/// it can afford to search by keyword + nearest-line-below without the
/// generic parser's anti-noise confidence gate.
fn relaxed_field_value(lines: &[OcrLine], keywords: &[&str]) -> Option<String> {
    let label_index = lines.iter().position(|line| {
        let upper = line.text.to_uppercase();
        keywords.iter().any(|keyword| upper.contains(keyword))
    })?;
    let label_line = &lines[label_index];

    if let Some((_, value)) = label_line.text.split_once(':') {
        let value = value.trim();
        if !value.is_empty() {
            return Some(value.to_owned());
        }
    }

    let bounds = Bounds::from_line(label_line);
    lines
        .iter()
        .enumerate()
        .filter(|(index, line)| {
            *index != label_index
                && is_below(bounds, line)
                && !looks_like_label(&line.text)
                && left_distance(bounds.left, line) <= MAX_FIELD_ALIGNMENT_DRIFT * 2
        })
        .min_by_key(|(_, line)| vertical_gap_from(bounds, line))
        .map(|(_, line)| line.text.trim().to_owned())
}

fn field_value<'a>(fields: &'a [Field], label: &str) -> Option<&'a str> {
    fields
        .iter()
        .find(|field| field.label.contains(label))
        .map(|field| field.value.as_str())
}

fn push_borrowed_field(fields: &mut Vec<Field>, label: &str, value: Option<&str>) {
    if let Some(value) = value
        && let Some(field) = make_field(label, value)
    {
        fields.push(field);
    }
}

fn push_owned_field(fields: &mut Vec<Field>, label: &str, value: Option<String>) {
    if let Some(value) = value {
        push_borrowed_field(fields, label, Some(&value));
    }
}

fn join_unique<'a>(values: impl Iterator<Item = &'a str>) -> Option<String> {
    let mut unique = Vec::new();
    for value in values.map(str::trim).filter(|value| !value.is_empty()) {
        if !unique.contains(&value) {
            unique.push(value);
        }
    }
    if unique.is_empty() {
        return None;
    }
    let capacity = unique
        .iter()
        .map(|value| value.len())
        .sum::<usize>()
        .saturating_add(unique.len().saturating_sub(1).saturating_mul(3));
    let mut joined = String::with_capacity(capacity);
    for value in unique {
        if !joined.is_empty() {
            joined.push_str(" / ");
        }
        joined.push_str(value);
    }
    Some(joined)
}

/// Nepal's real National ID Number is 4 dash-separated groups
/// (`336-286-062-0`, i.e. 3-3-3-1 digits) — not the 3-group `[3,3,4]` shape
/// a synthetic test fixture originally assumed. Confirmed against a real
/// government-issued card. Broadened to any 3-5 group digit run that isn't
/// itself date-shaped, rather than one fixed group-count pattern, since a
/// second real ID could plausibly use yet another grouping.
fn is_identity_number(text: &str) -> bool {
    let groups = digit_group_lengths(text);
    let group_count = (3..=5).contains(&groups.len());
    let group_sizes = groups.iter().all(|&length| (1..=4).contains(&length));
    let total_digits = groups.iter().sum::<usize>() >= 6;
    group_count && group_sizes && total_digits && !is_date(text)
}

fn is_date(text: &str) -> bool {
    digit_group_lengths(text).as_slice() == [4, 2, 2]
}

fn digit_group_lengths(text: &str) -> Vec<usize> {
    // OCR noise regularly swaps '-' for '.', '/', or a stray space in
    // dashed ID/date runs; treat any of them as a group separator instead
    // of requiring an exact '-' match.
    let mut lengths = Vec::new();
    for group in text.trim().split(['-', '.', '/', ' ']) {
        if group.is_empty() {
            continue;
        }
        if !group.chars().all(|character| character.is_numeric()) {
            return Vec::new();
        }
        lengths.push(group.chars().count());
    }
    lengths
}

fn contains_devanagari(text: &str) -> bool {
    text.chars()
        .any(|character| ('\u{0900}'..='\u{097f}').contains(&character))
}

fn is_field_label(line: &OcrLine) -> bool {
    line.confidence >= FIELD_LABEL_MIN_CONFIDENCE && looks_like_label(&line.text)
}

fn looks_like_label(text: &str) -> bool {
    let mut saw_label_word = false;
    for word in text.split_whitespace() {
        let cleaned: String = word.chars().filter(char::is_ascii_alphabetic).collect();
        if cleaned.is_empty() {
            // Non-ASCII (Devanagari) or punctuation-only token: skip, don't
            // let it disqualify the line either way.
            continue;
        }
        if cleaned.chars().any(|character| character.is_ascii_lowercase()) {
            return false;
        }
        if cleaned.len() == 1 {
            // Stray single-letter OCR artifact (commonly a misread '/'
            // bilingual separator rendered as "I"); not conclusive.
            continue;
        }
        if !LABEL_WORDS.contains(&cleaned.as_str()) {
            return false;
        }
        saw_label_word = true;
    }
    saw_label_word
}

fn connected_line_groups(
    lines: &[OcrLine],
    include: impl Fn(&OcrLine) -> bool + Copy,
) -> Vec<Vec<usize>> {
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

fn value_group_from(
    lines: &[OcrLine],
    seed: usize,
    include: impl Fn(&OcrLine) -> bool,
) -> Vec<usize> {
    let mut group = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            (index != seed && include(line) && lines_share_value_row(&lines[seed], line))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    group.push(seed);
    group
}

fn connected_group(lines: &[OcrLine], seed: usize, remaining: &mut Vec<usize>) -> Vec<usize> {
    let mut group = vec![seed];
    while let Some(position) = remaining.iter().position(|candidate| {
        group
            .iter()
            .any(|member| lines_share_row(&lines[*member], &lines[*candidate]))
    }) {
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
    overlaps_vertically
        && horizontal_gap <= a_bounds.height().max(b_bounds.height()).saturating_mul(2)
}

fn lines_share_value_row(a: &OcrLine, b: &OcrLine) -> bool {
    let a_bounds = Bounds::from_line(a);
    let b_bounds = Bounds::from_line(b);
    let center_distance = (a_bounds.top + a_bounds.bottom)
        .abs_diff(b_bounds.top + b_bounds.bottom)
        .saturating_div(2);
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
        Bounds {
            left: i32::MAX,
            top: i32::MAX,
            right: i32::MIN,
            bottom: i32::MIN,
        },
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
    let value_top = value
        .polygon
        .iter()
        .map(|point| point[1])
        .min()
        .unwrap_or(0);
    value_top.saturating_sub(label.bottom)
}

fn is_below(label: Bounds, value: &OcrLine) -> bool {
    let value_top = value
        .polygon
        .iter()
        .map(|point| point[1])
        .min()
        .unwrap_or(0);
    value_top
        >= label
            .bottom
            .saturating_sub(label.height().saturating_div(2))
}

fn left_distance(label_left: i32, value: &OcrLine) -> i32 {
    label_left
        .abs_diff(line_left(value))
        .try_into()
        .unwrap_or(i32::MAX)
}

fn join_line_text(lines: &[OcrLine], indexes: &[usize]) -> String {
    let capacity = indexes
        .iter()
        .map(|&index| lines[index].text.len())
        .sum::<usize>()
        .saturating_add(indexes.len().saturating_sub(1));
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
    (!label.is_empty() && !value.is_empty()).then(|| Field {
        label: label.to_owned(),
        value: value.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use axum::Json;
    use axum::Router;
    use axum::routing::post;
    use serde_json::{Value, json};

    use super::*;

    fn line(text: &str, confidence: f32, left: i32, top: i32) -> OcrLine {
        line_with_width(text, confidence, left, top, 100)
    }

    fn line_with_width(text: &str, confidence: f32, left: i32, top: i32, width: i32) -> OcrLine {
        OcrLine {
            text: text.to_owned(),
            confidence,
            polygon: [
                [left, top],
                [left + width, top],
                [left + width, top + 20],
                [left, top + 20],
            ],
        }
    }

    #[test]
    fn extracts_inline_and_spatial_fields() {
        let lines = vec![
            line("NAME", 0.99, 10, 10),
            line("Prayag", 0.98, 12, 40),
            line("NATIONALITY: Nepali", 0.97, 10, 80),
        ];

        assert_eq!(
            parse_fields(&lines),
            vec![
                Field {
                    label: "NAME".to_owned(),
                    value: "Prayag".to_owned(),
                },
                Field {
                    label: "NATIONALITY".to_owned(),
                    value: "Nepali".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn treats_devanagari_text_as_a_value_without_an_english_label() {
        let lines = vec![
            line("नागरिकताको किसिम I CITIZENSHIP TYPE", 0.99, 10, 10),
            line("जारी गर्ने अधिकारी ISSUING OFFICER", 0.99, 1_000, 10),
            line("वंशज", 0.98, 12, 40),
        ];

        assert_eq!(
            parse_fields(&lines),
            vec![Field {
                label: "नागरिकताको किसिम I CITIZENSHIP TYPE".to_owned(),
                value: "वंशज".to_owned(),
            }]
        );
    }

    #[test]
    fn joins_split_labels_and_values_on_the_same_row() {
        let lines = vec![
            line("DATE", 0.99, 10, 10),
            line("OF", 0.99, 112, 10),
            line("ISSUE", 0.99, 214, 10),
            line("2024-02-15", 0.99, 12, 40),
            line("FULL NAME", 0.99, 10, 80),
            line_with_width("Prayag", 0.99, 12, 110, 100),
            line_with_width("Dhakal", 0.99, 120, 110, 100),
        ];

        assert_eq!(
            parse_fields(&lines),
            vec![
                Field {
                    label: "DATE OF ISSUE".to_owned(),
                    value: "2024-02-15".to_owned(),
                },
                Field {
                    label: "FULL NAME".to_owned(),
                    value: "Prayag Dhakal".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn omits_low_confidence_and_unaligned_field_guesses() {
        let lines = vec![
            line("GOVERNMENTIOFI", 0.88, 500, 10),
            line("noise", 0.90, 500, 40),
            line("NATIONAL IDENTITY CARD", 0.99, 500, 80),
            line("712-322-1775", 0.99, 200, 110),
            line("SEX", 0.99, 10, 150),
            line("M", 0.99, 12, 180),
        ];

        assert_eq!(
            parse_fields(&lines),
            vec![Field {
                label: "SEX".to_owned(),
                value: "M".to_owned(),
            }]
        );
    }

    #[test]
    fn returns_all_person_fields_from_a_national_identity_card() {
        let lines = vec![
            line_with_width("NATIONAL IDENTITY CARD", 0.99, 500, 10, 300),
            line("राष्ट्रिय परिचय नम्बर", 0.95, 500, 50),
            line("NATIONALITY", 0.99, 10, 50),
            line("Nepali", 0.99, 12, 80),
            line_with_width("७१२-३२२-१७७४", 0.96, 500, 80, 180),
            line_with_width("712-322-1775", 0.99, 500, 110, 180),
            line("नाम थर", 0.99, 500, 150),
            line("प्रयाग", 0.99, 500, 180),
            line("ढकाल", 0.99, 608, 180),
            line("FULL NAME", 0.99, 500, 220),
            line("Prayag", 0.99, 500, 250),
            line("Dhakal", 0.99, 608, 250),
            line("DATE", 0.99, 10, 290),
            line("OF", 0.99, 112, 290),
            line("ISSUE", 0.99, 214, 290),
            line("2024-02-15", 0.99, 12, 320),
            line("जन्म मिति", 0.99, 10, 360),
            line("DATE", 0.99, 500, 360),
            line("OF", 0.99, 602, 360),
            line("BIRTH", 0.99, 704, 360),
            line("२०५६-०७-२३", 0.99, 10, 390),
            line("1999-11-09", 0.99, 500, 390),
            line("लिङ्ग", 0.99, 10, 430),
            line("SEX", 0.99, 500, 430),
            line("पुरूष", 0.99, 10, 460),
            line("M", 0.99, 500, 460),
        ];

        assert_eq!(
            parse_fields(&lines),
            vec![
                Field {
                    label: "NATIONAL ID NUMBER".to_owned(),
                    value: "७१२-३२२-१७७४ / 712-322-1775".to_owned(),
                },
                Field {
                    label: "NATIONALITY".to_owned(),
                    value: "Nepali".to_owned(),
                },
                Field {
                    label: "FULL NAME".to_owned(),
                    value: "प्रयाग ढकाल / Prayag Dhakal".to_owned(),
                },
                Field {
                    label: "DATE OF ISSUE".to_owned(),
                    value: "2024-02-15".to_owned(),
                },
                Field {
                    label: "DATE OF BIRTH".to_owned(),
                    value: "२०५६-०७-२३ / 1999-11-09".to_owned(),
                },
                Field {
                    label: "SEX".to_owned(),
                    value: "पुरूष / M".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn recognizes_the_real_four_group_national_id_number_format() {
        // "712-322-1775" (3 groups) in the fixture above was a made-up
        // shape. A real scanned Nepal NID card's number is 4 dash-separated
        // groups ("336-286-062-0", i.e. 3-3-3-1 digits) — is_identity_number
        // only matched the 3-group shape and silently dropped this field
        // even though OCR recognized the text perfectly.
        let lines = vec![
            line_with_width("NATIONAL IDENTITY CARD", 0.99, 500, 10, 300),
            line("राष्ट्रिय परिचय नम्बर", 0.95, 500, 50),
            line("NATIONALITY", 0.99, 10, 50),
            line("Nepali", 0.99, 12, 80),
            line_with_width("३३६-२८६-०६२-०", 0.997, 500, 80, 180),
            line_with_width("336-286-062-0", 0.999, 500, 110, 180),
            line("नाम थर", 0.99, 500, 150),
            line("FULL NAME", 0.99, 500, 220),
            line("Roshan Joshi", 0.99, 500, 250),
        ];

        let fields = parse_fields(&lines);
        let get = |label: &str| {
            fields
                .iter()
                .find(|field| field.label == label)
                .map(|field| field.value.as_str())
        };

        assert_eq!(
            get("NATIONAL ID NUMBER"),
            Some("३३६-२८६-०६२-० / 336-286-062-0")
        );
    }

    #[test]
    fn detects_and_extracts_a_document_type_defined_purely_as_table_data() {
        // Proves the engine generalizes: a brand-new document type (not
        // NID, not citizenship) is just a DocumentType constant — zero new
        // parsing functions, using the exact same detect_keywords scoring
        // and extract_fields engine every other document type goes
        // through.
        const LICENSE_FIELDS: &[FieldSignature] = &[
            FieldSignature {
                output_label: "LICENSE NUMBER",
                english_keywords: &["LICENSE NO"],
                block: None,
                global_shape: None,
            },
            FieldSignature {
                output_label: "EXPIRY",
                english_keywords: &["EXPIRY DATE"],
                block: None,
                global_shape: None,
            },
        ];
        const DRIVER_LICENSE: DocumentType = DocumentType {
            detect_keywords: &["DRIVER LICENSE", "LICENSE NO"],
            min_detect_matches: 2,
            fields: LICENSE_FIELDS,
        };

        let lines = vec![
            line_with_width("DRIVER LICENSE", 0.99, 10, 10, 200),
            line("LICENSE NO", 0.99, 10, 50),
            line("DL-99887", 0.99, 12, 80),
            line("EXPIRY DATE", 0.99, 10, 120),
            line("2030-01-01", 0.99, 12, 150),
        ];

        let matches = DRIVER_LICENSE
            .detect_keywords
            .iter()
            .filter(|keyword| has_line_ci(&lines, keyword))
            .count();
        assert!(matches >= DRIVER_LICENSE.min_detect_matches);

        let generic = parse_generic_fields(&lines);
        let fields = extract_fields(&lines, &generic, DRIVER_LICENSE.fields);

        assert_eq!(
            fields,
            vec![
                Field {
                    label: "LICENSE NUMBER".to_owned(),
                    value: "DL-99887".to_owned(),
                },
                Field {
                    label: "EXPIRY".to_owned(),
                    value: "2030-01-01".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn still_extracts_all_fields_when_one_header_anchor_is_missing() {
        // Real-world scan: the "NATIONALITY" header word never cleared the
        // service's confidence floor, so it never became an OcrLine at all.
        // The card must still be recognized as a national identity card
        // (2-of-N anchor scoring) and every remaining field still found.
        let lines = vec![
            line_with_width("NATIONAL IDENTITY CARD", 0.99, 500, 10, 300),
            line("राष्ट्रिय परिचय नम्बर", 0.95, 500, 50),
            line("Nepali", 0.99, 12, 80),
            line_with_width("712-322-1775", 0.99, 500, 110, 180),
            line("नाम थर", 0.99, 500, 150),
            line("प्रयाग", 0.99, 500, 180),
            line("ढकाल", 0.99, 608, 180),
            line("FULL NAME", 0.99, 500, 220),
            line("Prayag", 0.99, 500, 250),
            line("Dhakal", 0.99, 608, 250),
            line("DATE", 0.99, 10, 290),
            line("OF", 0.99, 112, 290),
            line("ISSUE", 0.99, 214, 290),
            line("2024-02-15", 0.99, 12, 320),
            line("1999-11-09", 0.99, 500, 390),
            line("SEX", 0.99, 500, 430),
            line("M", 0.99, 500, 460),
        ];

        let fields = parse_fields(&lines);
        let get = |label: &str| {
            fields
                .iter()
                .find(|field| field.label == label)
                .map(|field| field.value.as_str())
        };

        assert_eq!(get("NATIONAL ID NUMBER"), Some("712-322-1775"));
        assert_eq!(get("FULL NAME"), Some("प्रयाग ढकाल / Prayag Dhakal"));
        assert_eq!(get("DATE OF ISSUE"), Some("2024-02-15"));
        assert_eq!(get("DATE OF BIRTH"), Some("1999-11-09"));
        assert_eq!(get("SEX"), Some("M"));
    }

    #[test]
    fn tolerates_alternate_separators_in_dates_and_id_numbers() {
        // OCR frequently confuses '-' with '.' or '/' in dashed digit runs.
        let lines = vec![
            line_with_width("NATIONAL IDENTITY CARD", 0.99, 500, 10, 300),
            line("NATIONALITY", 0.99, 10, 50),
            line("Nepali", 0.99, 12, 80),
            line("परिचय", 0.95, 500, 50),
            line_with_width("712.322/1775", 0.99, 500, 80, 180),
            line("FULL NAME", 0.99, 500, 220),
            line("Prayag Dhakal", 0.99, 500, 250),
            line("DATE", 0.99, 10, 290),
            line("OF", 0.99, 112, 290),
            line("ISSUE", 0.99, 214, 290),
            line("2024.02.15", 0.99, 12, 320),
            line("1999/11 09", 0.99, 500, 390),
            line("SEX", 0.99, 500, 430),
            line("M", 0.99, 500, 460),
        ];

        let fields = parse_fields(&lines);
        let get = |label: &str| {
            fields
                .iter()
                .find(|field| field.label == label)
                .map(|field| field.value.as_str())
        };

        assert_eq!(get("NATIONAL ID NUMBER"), Some("712.322/1775"));
        assert_eq!(get("DATE OF ISSUE"), Some("2024.02.15"));
        assert_eq!(get("DATE OF BIRTH"), Some("1999/11 09"));
    }

    #[test]
    fn does_not_mistake_an_all_caps_printed_name_for_a_label() {
        // Reproduces a real scan: FULL NAME's value line "ROSHAN JOSHI" is
        // itself all-caps, so the old "2+ uppercase ASCII letters = label"
        // heuristic classified it as another label and the anchor search
        // skipped past it, latching onto "जन्म मिति" (Nepali for "date of
        // birth") below it instead.
        let lines = vec![
            line_with_width("NATIONAL IDENTITY CARD", 0.99, 500, 10, 300),
            line("NATIONALITY", 0.99, 10, 50),
            line("Nepali", 0.99, 12, 80),
            line("परिचय", 0.95, 500, 50),
            line_with_width("336-286-062-0", 0.99, 500, 80, 180),
            line("FULL NAME", 0.99, 448, 948),
            line_with_width("ROSHAN JOSHI", 0.95, 445, 1011, 300),
            line("जन्म मिति", 0.90, 443, 1126),
            line("DATE OF BIRTH", 0.99, 805, 1148),
            line("२०५९-९०-९०", 0.94, 446, 1191),
            line("2003-01-24", 1.00, 800, 1197),
            line("SEX", 0.99, 802, 1292),
            line("M", 1.00, 802, 1354),
        ];

        let fields = parse_fields(&lines);
        let get = |label: &str| {
            fields
                .iter()
                .find(|field| field.label == label)
                .map(|field| field.value.as_str())
        };

        assert_eq!(get("FULL NAME"), Some("ROSHAN JOSHI"));
        assert_eq!(get("DATE OF BIRTH"), Some("२०५९-९०-९० / 2003-01-24"));
    }

    #[test]
    fn returns_clean_bilingual_citizenship_fields() {
        let lines = vec![
            line("स्थायी ठेगाना I PERMANENT ADDRESS", 0.99, 10, 10),
            line("NICIN", 0.99, 500, 10),
            line("सूर्यविनायक नगरपालिका-४, भक्तपुर", 0.99, 12, 40),
            line("01", 0.99, 502, 40),
            line("Suryabinayak Municipality-4, Bhaktapur", 0.99, 12, 70),
            line("नागरिकताको किसिम I CITIZENSHIP TYPE", 0.99, 10, 110),
            line("वंशज", 0.99, 12, 140),
            line("नागरिकता नं. I CC NUMBER", 0.99, 10, 180),
            line("०४-०२-७३-०००१९", 0.99, 12, 210),
        ];

        assert_eq!(
            parse_fields(&lines),
            vec![
                Field {
                    label: "PERMANENT ADDRESS".to_owned(),
                    value: "सूर्यविनायक नगरपालिका-४, भक्तपुर / Suryabinayak Municipality-4, Bhaktapur"
                        .to_owned(),
                },
                Field {
                    label: "NICIN".to_owned(),
                    value: "01".to_owned(),
                },
                Field {
                    label: "CITIZENSHIP TYPE".to_owned(),
                    value: "वंशज".to_owned(),
                },
                Field {
                    label: "CITIZENSHIP NUMBER".to_owned(),
                    value: "०४-०२-७३-०००१९".to_owned(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn calls_the_pp_ocr_v5_service_contract() {
        async fn mock(Json(request): Json<Value>) -> Json<Value> {
            assert_eq!(request["file"], "dGVzdC1pbWFnZQ==");
            assert_eq!(request["fileType"], 1);
            assert_eq!(request["useDocOrientationClassify"], false);
            assert_eq!(request["useTextlineOrientation"], true);
            assert_eq!(request["textRecScoreThresh"], 0.45);
            assert_eq!(request["visualize"], false);
            Json(json!({
                "logId": "test",
                "errorCode": 0,
                "errorMsg": "Success",
                "result": {
                    "ocrResults": [{
                        "prunedResult": {
                            "rec_texts": ["NAME", "Prayag", "noise"],
                            "rec_scores": [0.99, 0.98, 0.1],
                            "rec_polys": [
                                [[10, 10], [110, 10], [110, 30], [10, 30]],
                                [[10, 40], [110, 40], [110, 60], [10, 60]],
                                [[10, 70], [110, 70], [110, 90], [10, 90]]
                            ]
                        },
                        "ocrImage": null
                    }],
                    "dataInfo": {}
                }
            }))
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock PP-OCRv5 service");
        let endpoint = format!(
            "http://{}/ocr",
            listener.local_addr().expect("server address")
        )
        .parse()
        .expect("valid mock URL");
        let server = tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/ocr", post(mock)))
                .await
                .expect("mock server failed");
        });
        let client = PaddleOcrClient::new(
            endpoint,
            None,
            1,
            Duration::from_secs(2),
            Duration::from_secs(1),
            0.45,
        )
        .expect("create PP-OCRv5 client");

        let document = client
            .extract(b"test-image")
            .await
            .expect("extract mocked document");
        server.abort();

        assert_eq!(
            document.fields,
            vec![Field {
                label: "NAME".to_owned(),
                value: "Prayag".to_owned(),
            }]
        );
        assert_eq!(
            serde_json::to_value(document).expect("serialize OCR response"),
            json!({
                "fields": [{
                    "label": "NAME",
                    "value": "Prayag"
                }]
            })
        );
    }
}
