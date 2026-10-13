//! Debug overlay: kept boxes drawn over the OCR input (`scanned_document/bboxed_{side}.jpg`).

use opencv::core::{Mat, Point, Scalar, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use crate::error::AppError;
use crate::fields::OcrLine;

/// Box outlines with index and confidence (OpenCV fonts can't draw Devanagari; match indexes
/// to the `OCR_DEBUG_LINES` log).
pub(crate) fn draw_debug_bboxes(image: &Mat, lines: &[OcrLine]) -> Result<Mat, AppError> {
    let mut annotated = image.try_clone()?;
    let box_color = Scalar::new(0.0, 220.0, 0.0, 0.0);
    let label_color = Scalar::new(0.0, 0.0, 255.0, 0.0);
    for (index, line) in lines.iter().enumerate() {
        let points: Vector<Point> =
            line.polygon.iter().map(|p| Point::new(p[0], p[1])).collect();
        let mut polygon = Vector::<Vector<Point>>::new();
        polygon.push(points);
        imgproc::polylines(&mut annotated, &polygon, true, box_color, 2, imgproc::LINE_8, 0)?;

        let label = format!("{index}:{:.2}", line.confidence);
        let origin = Point::new(line.polygon[0][0], (line.polygon[0][1] - 4).max(10));
        imgproc::put_text_def(
            &mut annotated,
            &label,
            origin,
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.4,
            label_color,
        )?;
    }
    Ok(annotated)
}
