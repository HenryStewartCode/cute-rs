//! The simulator's `printf`: C's format language as CUDA's `vprintf`
//! interprets it, so that simulated output matches the GPU's byte for byte.
//!
//! A conversion is `%[flags][width][.precision][length]conversion` with the
//! flags `- + space 0 #`, a decimal width and precision, the length
//! modifiers `hh h l ll L j z t`, and the conversions `d i u o x X c e E f
//! F g G p %`. As with `vprintf`, `%d` and the other integer conversions read
//! an `int` (32 bits) unless `l` or `ll` asks for 64, and `hh` and `h`
//! narrow further; the floating conversions read a double.

use crate::ir::DType;

/// The largest width or precision honoured (larger ones are clamped, so a
/// stray long digit string cannot exhaust memory).
const MAX_FIELD: usize = 4096;

/// One parsed conversion specification.
#[derive(Default)]
struct Spec {
    left: bool,
    plus: bool,
    space: bool,
    zero: bool,
    alt: bool,
    width: usize,
    precision: Option<usize>,
    /// Bits an integer conversion reads: 8, 16, 32 or 64.
    int_bits: u32,
}

/// Expand `format` with `args`, each a value's type and raw bits.
pub(super) fn format_printf(format: &str, args: &[(DType, u64)], to_f64: fn(DType, u64) -> f64) -> String {
    let mut out = String::new();
    let mut args = args.iter();
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let mut spec = Spec { int_bits: 32, ..Spec::default() };
        // The specification's text, written back as it is when it is not a
        // conversion (an unknown letter, or a `%` at the end).
        let mut text = String::from('%');
        while let Some(&f) = chars.peek() {
            match f {
                '-' => spec.left = true,
                '+' => spec.plus = true,
                ' ' => spec.space = true,
                '0' => spec.zero = true,
                '#' => spec.alt = true,
                _ => break,
            }
            text.push(f);
            chars.next();
        }
        while let Some(d) = chars.peek().and_then(|c| c.to_digit(10)) {
            spec.width = (spec.width.saturating_mul(10).saturating_add(d as usize)).min(MAX_FIELD);
            text.push(chars.next().expect("a digit"));
        }
        if chars.peek() == Some(&'.') {
            text.push(chars.next().expect("a point"));
            let mut p: usize = 0;
            while let Some(d) = chars.peek().and_then(|c| c.to_digit(10)) {
                p = (p.saturating_mul(10).saturating_add(d as usize)).min(MAX_FIELD);
                text.push(chars.next().expect("a digit"));
            }
            spec.precision = Some(p);
        }
        while let Some(&m) = chars.peek() {
            match m {
                'h' => spec.int_bits = if spec.int_bits == 16 { 8 } else { 16 },
                'l' | 'L' | 'j' | 'z' | 't' => spec.int_bits = 64,
                _ => break,
            }
            text.push(m);
            chars.next();
        }
        let Some(conv) = chars.next() else {
            out.push_str(&text);
            break;
        };
        if conv == '%' {
            out.push('%');
            continue;
        }
        if !matches!(conv, 'd' | 'i' | 'u' | 'o' | 'x' | 'X' | 'c' | 'e' | 'E' | 'f' | 'F' | 'g' | 'G' | 'p') {
            // Not a conversion this formatter knows: no argument is read.
            text.push(conv);
            out.push_str(&text);
            continue;
        }
        let Some(&(ty, bits)) = args.next() else {
            out.push_str("<missing>");
            continue;
        };
        let raw = if ty.is_float() { bits } else { bits & mask(ty.bits().min(64)) };
        let (sign, body, numeric) = match conv {
            'd' | 'i' => {
                let v = signed_arg(ty, raw, spec.int_bits);
                let digits = int_digits(v.unsigned_abs() as u128, 10, false, spec.precision);
                (sign_of(v < 0, &spec), digits, true)
            }
            'u' | 'o' | 'x' | 'X' => {
                let v = raw & mask(spec.int_bits as usize);
                let radix = match conv {
                    'o' => 8,
                    'u' => 10,
                    _ => 16,
                };
                let mut digits = int_digits(v as u128, radix, conv == 'X', spec.precision);
                if spec.alt && conv == 'o' && !digits.starts_with('0') {
                    digits.insert(0, '0');
                }
                let prefix = match conv {
                    'x' if spec.alt && v != 0 => "0x",
                    'X' if spec.alt && v != 0 => "0X",
                    _ => "",
                };
                (prefix.to_string(), digits, true)
            }
            'c' => (String::new(), ((raw & 0xff) as u8 as char).to_string(), false),
            // The driver formats `vprintf`'s buffer on the host, with the C
            // library: a pointer is `0x` and lowercase hexadecimal digits,
            // and the null pointer `(nil)` (checked on Linux, on SM89).
            'p' if raw == 0 => (String::new(), "(nil)".to_string(), false),
            'p' => (String::new(), format!("{raw:#x}"), false),
            'e' | 'E' | 'f' | 'F' | 'g' | 'G' => {
                let x = if ty.is_float() { to_f64(ty, bits) } else { f64::from_bits(bits) };
                let upper = conv.is_ascii_uppercase();
                let body = if x.is_finite() {
                    let p = spec.precision.unwrap_or(6);
                    match conv.to_ascii_lowercase() {
                        'f' => fixed(x.abs(), p, spec.alt),
                        'e' => exponential(x.abs(), p, spec.alt, upper),
                        _ => general(x.abs(), p, spec.alt, upper),
                    }
                } else {
                    let s = if x.is_nan() { "nan" } else { "inf" };
                    if upper { s.to_uppercase() } else { s.to_string() }
                };
                (sign_of(x.is_sign_negative() && !x.is_nan(), &spec), body, x.is_finite())
            }
            _ => unreachable!("a known conversion"),
        };
        let len = sign.len() + body.len();
        let pad = spec.width.saturating_sub(len);
        // `0` pads after the sign, except when left-justified or (for
        // integers) when a precision is given.
        let int_conv = matches!(conv, 'd' | 'i' | 'u' | 'o' | 'x' | 'X');
        let zero_pad = spec.zero && !spec.left && numeric && !(int_conv && spec.precision.is_some());
        if spec.left {
            out.push_str(&sign);
            out.push_str(&body);
            out.extend(std::iter::repeat_n(' ', pad));
        } else if zero_pad {
            out.push_str(&sign);
            out.extend(std::iter::repeat_n('0', pad));
            out.push_str(&body);
        } else {
            out.extend(std::iter::repeat_n(' ', pad));
            out.push_str(&sign);
            out.push_str(&body);
        }
    }
    out
}

fn mask(bits: usize) -> u64 {
    if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }
}

/// The argument as a signed integer of `bits` bits, as `vprintf` reads it.
fn signed_arg(ty: DType, raw: u64, bits: u32) -> i64 {
    // Wider than the argument: sign-extend a signed argument.
    let full = if ty.is_signed() && ty.bits() < 64 {
        let w = ty.bits() as u32;
        ((raw << (64 - w)) as i64) >> (64 - w)
    } else {
        raw as i64
    };
    let shift = 64 - bits;
    (full << shift) >> shift
}

fn sign_of(negative: bool, spec: &Spec) -> String {
    if negative {
        "-".into()
    } else if spec.plus {
        "+".into()
    } else if spec.space {
        " ".into()
    } else {
        String::new()
    }
}

/// The digits of `v`, at least `precision` of them (a zero precision and
/// a zero value give none, as C specifies).
fn int_digits(v: u128, radix: u32, upper: bool, precision: Option<usize>) -> String {
    if v == 0 && precision == Some(0) {
        return String::new();
    }
    let mut s = match radix {
        8 => format!("{v:o}"),
        16 if upper => format!("{v:X}"),
        16 => format!("{v:x}"),
        _ => v.to_string(),
    };
    if let Some(p) = precision
        && s.len() < p
    {
        s.insert_str(0, &"0".repeat(p - s.len()));
    }
    s
}

fn fixed(x: f64, p: usize, alt: bool) -> String {
    let mut s = format!("{x:.p$}");
    if alt && p == 0 {
        s.push('.');
    }
    s
}

/// `d.ddde+XX`: C's exponent has a sign and at least two digits.
fn exponential(x: f64, p: usize, alt: bool, upper: bool) -> String {
    let s = format!("{x:.p$e}");
    let (mantissa, exp) = s.split_once('e').expect("an exponent");
    let exp: i32 = exp.parse().expect("an integer exponent");
    let mut m = mantissa.to_string();
    if alt && p == 0 {
        m.push('.');
    }
    let e = if upper { 'E' } else { 'e' };
    format!("{m}{e}{}{:02}", if exp < 0 { '-' } else { '+' }, exp.unsigned_abs())
}

/// `%g`: `%e` or `%f` by the exponent, with `P` significant digits and
/// trailing zeros removed (unless `#`).
fn general(x: f64, p: usize, alt: bool, upper: bool) -> String {
    let p = p.max(1);
    let exp = if x == 0.0 {
        0
    } else {
        let s = format!("{x:.*e}", p - 1);
        s.split_once('e').expect("an exponent").1.parse::<i32>().expect("an integer exponent")
    };
    let mut s = if exp < -4 || exp >= p as i32 {
        exponential(x, p - 1, alt, upper)
    } else {
        fixed(x, (p as i32 - 1 - exp) as usize, alt)
    };
    if !alt {
        let (num, tail) = match s.find(['e', 'E']) {
            Some(i) => (s[..i].to_string(), s[i..].to_string()),
            None => (s.clone(), String::new()),
        };
        let num = if num.contains('.') { num.trim_end_matches('0').trim_end_matches('.').to_string() } else { num };
        s = num + &tail;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(format: &str, args: &[(DType, u64)]) -> String {
        format_printf(format, args, |ty, b| match ty {
            DType::F32 => f32::from_bits(b as u32) as f64,
            _ => f64::from_bits(b),
        })
    }

    fn d(x: f64) -> (DType, u64) {
        (DType::F64, x.to_bits())
    }

    fn i(x: i32) -> (DType, u64) {
        (DType::I32, x as u32 as u64)
    }

    /// Expected strings from C's printf.
    #[test]
    fn matches_c() {
        assert_eq!(f("%5d|%-5d|%05d|%+d|% d", &[i(42), i(42), i(-42), i(7), i(7)]), "   42|42   |-0042|+7| 7");
        assert_eq!(f("%.3d|%8.3d|%-+6d|%0+5d", &[i(5), i(-5), i(3), i(3)]), "005|    -005|+3    |+0003");
        assert_eq!(
            f("%x|%#x|%#X|%08x|%o|%#o", &[i(255), i(255), i(255), i(255), i(8), i(8)]),
            "ff|0xff|0XFF|000000ff|10|010"
        );
        assert_eq!(f("%u|%hhd|%hu|%c", &[i(-1), i(200), i(-1), i(65)]), "4294967295|-56|65535|A");
        let big = (DType::I64, (-5_000_000_000i64) as u64);
        assert_eq!(
            f("%lld|%llu|%d", &[big, (DType::U64, u64::MAX), big]),
            "-5000000000|18446744073709551615|-705032704"
        );
        assert_eq!(
            f("%f|%.2f|%8.3f|%-8.1f|%+.0f|%#.0f", &[d(1.5), d(-2.345), d(1.23456), d(2.25), d(2.5), d(3.0)]),
            "1.500000|-2.35|   1.235|2.2     |+2|3."
        );
        assert_eq!(f("%e|%.2E|%10.1e", &[d(12345.678), d(-0.000123), d(0.0)]), "1.234568e+04|-1.23E-04|   0.0e+00");
        assert_eq!(
            f("%g|%g|%g|%.3g|%#g|%G", &[d(100000.0), d(1000000.0), d(0.0001), d(1.23456), d(1.5), d(1e-5)]),
            "100000|1e+06|0.0001|1.23|1.50000|1E-05"
        );
        assert_eq!(
            f("%f|%5.1f|%F|%07.2f", &[d(f64::INFINITY), d(f64::NAN), d(f64::NEG_INFINITY), d(-1.5)]),
            "inf|  nan|-INF|-001.50"
        );
        assert_eq!(f("%7.4f, %7.4f", &[(DType::F32, (-0.4326f32).to_bits() as u64), d(0.1238)]), "-0.4326,  0.1238");
        assert_eq!(f("100%%", &[]), "100%");
        let p = |x: u64| (DType::I64, x);
        assert_eq!(f("%p|%p|%18p", &[p(0x7cf5_b5e0_0000), p(0), p(8)]), "0x7cf5b5e00000|(nil)|               0x8");
    }

    /// Odd format strings neither panic nor read arguments they do not
    /// convert.
    #[test]
    fn odd_formats() {
        assert_eq!(f("50%", &[]), "50%");
        assert_eq!(f("x %-08.3l", &[i(1)]), "x %-08.3l");
        assert_eq!(f("%d %d", &[i(1)]), "1 <missing>");
        assert_eq!(f("%k|%d", &[i(4)]), "%k|4");
        assert_eq!(f("%s|%d", &[i(4)]), "%s|4");
        let long = f("%99999999999999999999999d", &[i(1)]);
        assert_eq!(long.len(), MAX_FIELD);
        let long = f("%.99999999999999999999999f", &[d(1.0)]);
        assert_eq!(long.len(), MAX_FIELD + 2);
    }
}
