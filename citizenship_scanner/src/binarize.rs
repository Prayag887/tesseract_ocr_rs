//! Second pass of the upscale pipeline (see `upscale.rs`): after the crop
//! is upscaled, this collapses it to pure black/white with dark ink pixels
//! turned *white* and everything else (paper, watermark tint, colored
//! stamps/signatures) turned *black* — inverted from the conventional
//! "black text on white page" convention, matching what the debug output
//! is named for (`_debug_inverted_binary_{side}.jpg`).
//!
//! Deliberately simple: a global Otsu threshold, not the local/adaptive
//! approach in `preprocess.rs` (that one exists to fight an uneven-lighting
//! photo of watermarked paper; this one runs on an already-upscaled,
//! already-sharp crop where a single global cutoff between "ink" and
//! "everything else" is enough). A stamp or signature landing on top of
//! real text can come out fused into noise here and that's accepted, not a
//! bug to chase — the goal is that *clear* text always survives, not that
//! every overlap is disentangled.

use opencv::core::Mat;
use opencv::imgproc;

use crate::error::AppError;

pub fn invert_binarize(bgr: &Mat) -> Result<Mat, AppError> {
    let mut gray = Mat::default();
    imgproc::cvt_color_def(bgr, &mut gray, imgproc::COLOR_BGR2GRAY)?;

    let mut binary = Mat::default();
    imgproc::threshold(
        &gray,
        &mut binary,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )?;

    // Both downstream models expect a 3-channel input (see preprocess.rs)
    // even though the content is now pure black/white.
    let mut bgr_out = Mat::default();
    imgproc::cvt_color_def(&binary, &mut bgr_out, imgproc::COLOR_GRAY2BGR)?;
    Ok(bgr_out)
}
