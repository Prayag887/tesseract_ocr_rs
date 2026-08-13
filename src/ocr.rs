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

#[derive(Debug, PartialEq)]
struct OcrLine {
    text: String,
    confidence: f32,
    polygon: [[i32; 2]; 4],
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

fn reading_order(a: &OcrLine, b: &OcrLine) -> Ordering {
    let a_top = a.polygon.iter().map(|point| point[1]).min().unwrap_or(0);
    let b_top = b.polygon.iter().map(|point| point[1]).min().unwrap_or(0);
    let a_left = a.polygon.iter().map(|point| point[0]).min().unwrap_or(0);
    let b_left = b.polygon.iter().map(|point| point[0]).min().unwrap_or(0);
    a_top.cmp(&b_top).then_with(|| a_left.cmp(&b_left))
}

fn parse_fields(lines: &[OcrLine]) -> Vec<Field> {
    let generic = parse_generic_fields(lines);
    if is_national_identity_card(lines) {
        return parse_national_identity_fields(lines, &generic);
    }
    if is_citizenship_card(lines) {
        return parse_citizenship_fields(lines, &generic);
    }
    generic
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

fn is_national_identity_card(lines: &[OcrLine]) -> bool {
    has_line(lines, "IDENTITY") && has_line(lines, "NATIONALITY") && has_line(lines, "FULL NAME")
}

fn is_citizenship_card(lines: &[OcrLine]) -> bool {
    has_line(lines, "CITIZENSHIP TYPE") && has_line(lines, "ICC NUMBER")
}

fn has_line(lines: &[OcrLine], needle: &str) -> bool {
    lines.iter().any(|line| line.text.contains(needle))
}

fn parse_national_identity_fields(lines: &[OcrLine], generic: &[Field]) -> Vec<Field> {
    let mut fields = Vec::with_capacity(6);

    let id_values = lines
        .iter()
        .position(|line| line.text.contains("परिचय"))
        .map(|label| {
            lines[label + 1..]
                .iter()
                .take_while(|line| !line.text.contains("नाम"))
                .filter(|line| is_identity_number(&line.text))
                .map(|line| line.text.as_str())
        })
        .and_then(join_unique);
    push_owned_field(&mut fields, "NATIONAL ID NUMBER", id_values);

    push_borrowed_field(
        &mut fields,
        "NATIONALITY",
        field_value(generic, "NATIONALITY"),
    );

    let nepali_name = lines
        .iter()
        .position(|line| line.text.contains("नाम") && line.text.contains("थर"))
        .map(|label| {
            let mut indexes = lines[label + 1..]
                .iter()
                .take_while(|line| !line.text.contains("FULL NAME"))
                .enumerate()
                .filter_map(|(offset, line)| {
                    (line.confidence >= FIELD_VALUE_MIN_CONFIDENCE
                        && contains_devanagari(&line.text)
                        && !line.text.chars().any(|character| character.is_numeric()))
                    .then_some(label + 1 + offset)
                })
                .collect::<Vec<_>>();
            indexes.sort_unstable_by_key(|&index| line_left(&lines[index]));
            join_line_text(lines, &indexes)
        })
        .filter(|name| !name.is_empty());
    let full_name = join_unique(
        [nepali_name.as_deref(), field_value(generic, "FULL NAME")]
            .into_iter()
            .flatten(),
    );
    push_owned_field(&mut fields, "FULL NAME", full_name);

    let issue_date = field_value(generic, "DATE OF ISSUE");
    push_borrowed_field(&mut fields, "DATE OF ISSUE", issue_date);

    let birth_dates = join_unique(
        lines
            .iter()
            .map(|line| line.text.as_str())
            .filter(|text| is_date(text))
            .filter(|text| Some(*text) != issue_date),
    );
    push_owned_field(&mut fields, "DATE OF BIRTH", birth_dates);

    let nepali_sex = lines
        .iter()
        .map(|line| line.text.trim())
        .find(|text| matches!(*text, "पुरुष" | "पुरूष" | "महिला" | "अन्य"));
    let sex = join_unique(
        [nepali_sex, field_value(generic, "SEX")]
            .into_iter()
            .flatten(),
    );
    push_owned_field(&mut fields, "SEX", sex);

    fields
}

fn parse_citizenship_fields(lines: &[OcrLine], generic: &[Field]) -> Vec<Field> {
    let mut fields = Vec::with_capacity(4);
    let english_address = lines
        .iter()
        .position(|line| line.text.contains("PERMANENT ADDRESS"))
        .and_then(|label| {
            lines[label + 1..]
                .iter()
                .take_while(|line| !line.text.contains("CITIZENSHIP TYPE"))
                .find(|line| {
                    line.text
                        .chars()
                        .any(|character| character.is_ascii_lowercase())
                })
        })
        .map(|line| line.text.as_str());
    let address = join_unique(
        [field_value(generic, "PERMANENT ADDRESS"), english_address]
            .into_iter()
            .flatten(),
    );
    push_owned_field(&mut fields, "PERMANENT ADDRESS", address);
    push_borrowed_field(&mut fields, "NICIN", field_value(generic, "NICIN"));
    push_borrowed_field(
        &mut fields,
        "CITIZENSHIP TYPE",
        field_value(generic, "CITIZENSHIP TYPE"),
    );
    push_borrowed_field(
        &mut fields,
        "CITIZENSHIP NUMBER",
        field_value(generic, "ICC NUMBER"),
    );
    fields
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

fn is_identity_number(text: &str) -> bool {
    digit_group_lengths(text) == [3, 3, 4]
}

fn is_date(text: &str) -> bool {
    digit_group_lengths(text) == [4, 2, 2]
}

fn digit_group_lengths(text: &str) -> [usize; 3] {
    let mut lengths = [0; 3];
    let mut groups = text.trim().split('-');
    for length in &mut lengths {
        let Some(group) = groups.next() else {
            return [0; 3];
        };
        if !group.chars().all(|character| character.is_numeric()) {
            return [0; 3];
        }
        *length = group.chars().count();
    }
    if groups.next().is_some() {
        return [0; 3];
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
    let mut uppercase_ascii = 0;
    for character in text.chars() {
        if character.is_ascii_lowercase() {
            return false;
        }
        uppercase_ascii += usize::from(character.is_ascii_uppercase());
    }
    uppercase_ascii >= 2
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
    fn returns_clean_bilingual_citizenship_fields() {
        let lines = vec![
            line("स्थायी ठेगाना I PERMANENT ADDRESS", 0.99, 10, 10),
            line("NICIN", 0.99, 500, 10),
            line("सूर्यविनायक नगरपालिका-४, भक्तपुर", 0.99, 12, 40),
            line("01", 0.99, 502, 40),
            line("Suryabinayak Municipality-4, Bhaktapur", 0.99, 12, 70),
            line("नागरिकताको किसिम I CITIZENSHIP TYPE", 0.99, 10, 110),
            line("वंशज", 0.99, 12, 140),
            line("नागरिकता न. ICC NUMBER", 0.99, 10, 180),
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
