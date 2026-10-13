//! Field extraction for a Nepali citizenship certificate into a fixed-shape
//! [`CitizenshipDocument`]. Front (Devanagari) and back (English) are separate scans:
//! [`extract`] reads one side, [`combine`] merges them (back preferred where both print a field).
//!
//! - `current_layout` — [`extract`], primary parser
//! - `legacy_layout` — [`extract_fallback`], old yellow cards (fallback only)
//! - `document` — [`CitizenshipDocument`], [`combine`], [`populated_fields`]
//! - `layout` — [`OcrLine`], reading order, box geometry, page blocks
//! - `keywords` — fuzzy label search, label -> value lookup
//! - `values` — gender/ward/date/digit normalization
//! - `generic` — debug-only label/value dump

mod current_layout;
mod document;
mod generic;
mod keywords;
mod layout;
mod legacy_layout;
mod values;
#[cfg(test)]
mod tests;

pub use current_layout::extract;
pub use document::{CitizenshipDocument, LOW_YIELD_FIELDS, combine, populated_fields};
pub use layout::{OcrLine, sort_reading_order};
pub use legacy_layout::{extract_fallback, number_line, vote_number};
