//! Text geometry: [`OcrLine`], reading order, [`Bounds`], and label-delimited page blocks.

use super::keywords::contains_keyword;

#[derive(Debug, PartialEq, Clone)]
pub struct OcrLine {
    pub text: String,
    pub confidence: f32,
    pub polygon: [[i32; 2]; 4],
}

/// Sorts lines into reading order in place; all label/value pairing depends on it.
/// Rows are clusters of vertically overlapping boxes (not sort-by-top: a value box sitting a few
/// px higher would precede its label). Each candidate is tested against the row's first box, not
/// the growing envelope, which would chain tightly stacked rows into one.
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

/// Rejoins boxes split mid-phrase (word-level detection breaks contiguous labels like "नाम थर").
fn merge_adjacent(originals: &[OcrLine], row: &[(usize, Bounds)]) -> Vec<OcrLine> {
    /// Gap below this fraction of text height = word space; above = column break (1.0 merged columns).
    const WORD_GAP_RATIO: f32 = 0.5;

    /// Max centre offset (fraction of the smaller height) to count as one printed line; overlap alone
    /// merged the back page's preamble into the row below.
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

/// Lines from the first `start_keyword` match (inclusive) to the next `end_keywords` match
/// (exclusive), fuzzy-matched. Scopes searches for labels printed more than once.
pub(super) fn block_between<'a>(lines: &'a [OcrLine], start_keyword: &str, end_keywords: &[&str]) -> &'a [OcrLine] {
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

/// Block starting at the second line matching `keywords` — for when a block header is unreadable
/// ("District" prints exactly twice: birth place, then permanent address).
pub(super) fn second_occurrence_block<'a>(lines: &'a [OcrLine], keywords: &[&str], end_keywords: &[&str]) -> &'a [OcrLine] {
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

pub(super) trait NoneIfEmpty<'a> {
    fn none_if_empty(self) -> Option<&'a [OcrLine]>;
}

impl<'a> NoneIfEmpty<'a> for &'a [OcrLine] {
    fn none_if_empty(self) -> Option<&'a [OcrLine]> {
        (!self.is_empty()).then_some(self)
    }
}

#[derive(Clone, Copy)]
pub(super) struct Bounds {
    pub(super) left: i32,
    pub(super) top: i32,
    pub(super) right: i32,
    pub(super) bottom: i32,
}

impl Bounds {
    pub(super) fn from_line(line: &OcrLine) -> Self {
        Self {
            left: line_left(line),
            top: line.polygon.iter().map(|point| point[1]).min().unwrap_or(0),
            right: line.polygon.iter().map(|point| point[0]).max().unwrap_or(0),
            bottom: line.polygon.iter().map(|point| point[1]).max().unwrap_or(0),
        }
    }

    pub(super) fn height(self) -> i32 {
        self.bottom.saturating_sub(self.top).max(1)
    }

    /// Smallest box covering both.
    pub(super) fn union(self, other: Self) -> Self {
        Self {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }

    /// Back to a polygon, clockwise from top-left.
    pub(super) fn to_polygon(self) -> [[i32; 2]; 4] {
        [
            [self.left, self.top],
            [self.right, self.top],
            [self.right, self.bottom],
            [self.left, self.bottom],
        ]
    }
}

pub(super) fn line_left(line: &OcrLine) -> i32 {
    line.polygon.iter().map(|point| point[0]).min().unwrap_or(0)
}
