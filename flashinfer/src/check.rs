// SPDX-License-Identifier: Apache-2.0

//! Error measures for comparing a kernel's output with a reference.
//!
//! `f32::max` and `f64::max` return the other operand when one is NaN, so a maximum folded
//! with them drops a NaN and an output of NaN counts as no error. The measures here give NaN
//! instead, and a comparison such as `err < tolerance` then fails.

/// The larger of two errors; NaN if either is.
pub(crate) fn nan_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) }
}

/// [`nan_max`] for `f64`.
pub(crate) fn nan_max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
}

/// The largest `|a - b|`; NaN if any element of either is NaN.
pub fn max_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, nan_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nan_output_is_an_error() {
        assert_eq!(max_error(&[1.0, 2.0], &[1.0, 2.5]), 0.5);
        assert!(max_error(&[1.0, f32::NAN], &[1.0, 2.0]).is_nan());
        assert!(max_error(&[f32::NAN, 2.0], &[1.0, 2.0]).is_nan());
        assert!(nan_max_f64(f64::NAN, 1.0).is_nan() && nan_max_f64(1.0, f64::NAN).is_nan());
        assert_eq!(nan_max_f64(1.0, 3.0), 3.0);
    }
}
