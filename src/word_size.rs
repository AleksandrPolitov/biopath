//! `--word-size`: the integer type every graph and engine structure is instantiated with, `u32`
//! (the default) or `u64`.

/// Parse a `--word-size` value: 32 or 64, anything else is refused.
pub(crate) fn parse_word_size(value: &str) -> Result<u8, String> {
    match value {
        "32" => Ok(32),
        "64" => Ok(64),
        _ => Err("the word size is 32 (the default) or 64".to_string()),
    }
}

/// Call `run::<W>(args…)` with the unsigned integer `W` of `bits` bits (32 or 64):
/// `with_word_size!(cli.word_size, run_with(cli))`. Any other value bails with the `--word-size`
/// error, or evaluates the given fallback arm: `with_word_size!(bits, f(x), other => …)`.
macro_rules! with_word_size {
    ($bits:expr, $($run:ident)::+ ($($arg:expr),* $(,)?)) => {
        $crate::word_size::with_word_size!(
            $bits,
            $($run)::+($($arg),*),
            other => anyhow::bail!("Unsupported word size: {other}. Supported are 32 and 64.")
        )
    };
    ($bits:expr, $($run:ident)::+ ($($arg:expr),* $(,)?), $other:ident => $fallback:expr) => {
        match $bits {
            32 => $($run)::+::<u32>($($arg),*),
            64 => $($run)::+::<u64>($($arg),*),
            $other => $fallback,
        }
    };
}

pub(crate) use with_word_size;
