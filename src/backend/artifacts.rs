//! Compilation artifacts and where they are kept (CuTe DSL's `CUTE_DSL_KEEP` and
//! `CUTE_DSL_DUMP_DIR`).

use std::fmt;
use std::path::{Path, PathBuf};

use super::options::OptionError;

/// One thing a compilation produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Artifact {
    /// [`Kernel::dump`](crate::Kernel::dump): the readable listing of the recorded IR.
    Ir,
    /// The emitted MLIR, with source locations.
    Mlir,
    /// The module after the first lowering stage (LLVM dialect, before it becomes PTX).
    Llvm,
    Ptx,
    /// The cubin: from `ptxas`, or from the driver's linker on a GPU machine.
    Cubin,
    /// The SASS: `nvdisasm` or `cuobjdump` on the cubin.
    Sass,
}

impl Artifact {
    pub const ALL: [Artifact; 6] =
        [Artifact::Ir, Artifact::Mlir, Artifact::Llvm, Artifact::Ptx, Artifact::Cubin, Artifact::Sass];

    /// The name `CUTE_RS_KEEP` uses.
    pub fn name(self) -> &'static str {
        match self {
            Artifact::Ir => "ir",
            Artifact::Mlir => "mlir",
            Artifact::Llvm => "llvm",
            Artifact::Ptx => "ptx",
            Artifact::Cubin => "cubin",
            Artifact::Sass => "sass",
        }
    }

    /// The file extension, with its dot.
    pub fn extension(self) -> &'static str {
        match self {
            Artifact::Ir => ".ir.txt",
            Artifact::Mlir => ".mlir",
            Artifact::Llvm => ".llvm.mlir",
            Artifact::Ptx => ".ptx",
            Artifact::Cubin => ".cubin",
            Artifact::Sass => ".sass",
        }
    }

    fn index(self) -> usize {
        Artifact::ALL.iter().position(|a| *a == self).expect("listed")
    }
}

impl fmt::Display for Artifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Which artifacts to keep: each one on, off or not said (so that a later source, such as the
/// environment, can decide).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Keep([Option<bool>; 6]);

impl Keep {
    pub(crate) fn set(&mut self, a: Artifact, on: bool) {
        self.0[a.index()] = Some(on);
    }

    pub(crate) fn get(&self, a: Artifact) -> Option<bool> {
        self.0[a.index()]
    }

    pub(crate) fn wants(&self, a: Artifact) -> bool {
        self.0[a.index()] == Some(true)
    }

    /// Fill what `self` leaves unsaid from `defaults`.
    pub(crate) fn or(mut self, defaults: Keep) -> Keep {
        for (a, d) in self.0.iter_mut().zip(defaults.0) {
            *a = a.or(d);
        }
        self
    }

    /// Parse `CUTE_RS_KEEP`: comma- or space-separated names (`ir`, `mlir`, `llvm`, `ptx`,
    /// `cubin`, `sass`, `all`; CuTe DSL's `ir-debug` is `mlir`, which has locations).
    pub(crate) fn parse(text: &str) -> Result<Keep, OptionError> {
        let mut keep = Keep::default();
        for token in text.split([',', ' ']).map(str::trim).filter(|t| !t.is_empty()) {
            match token.to_ascii_lowercase().as_str() {
                "all" => Artifact::ALL.iter().for_each(|a| keep.set(*a, true)),
                "ir-debug" => keep.set(Artifact::Mlir, true),
                other => match Artifact::ALL.iter().find(|a| a.name() == other) {
                    Some(a) => keep.set(*a, true),
                    None => {
                        return Err(OptionError::new(format!(
                            "`{token}` is not something to keep: use ir, mlir, llvm, ptx, cubin, sass or all"
                        )));
                    }
                },
            }
        }
        Ok(keep)
    }
}

/// What happened to one requested artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactNote {
    pub artifact: Artifact,
    /// Where it was written, if it was.
    pub path: Option<PathBuf>,
    /// Why it could not be produced or written, if it could not.
    pub unavailable: Option<String>,
}

/// A short stable hash of `text` (FNV-1a, 64 bits) as 8 hex digits: file names carry it so
/// that variants of one kernel do not overwrite each other.
pub(crate) fn short_hash(text: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{:08x}", (h ^ (h >> 32)) as u32)
}

/// The path `<dir>/<kernel>_<chip>_<hash>.<ext>`.
pub(crate) fn artifact_path(dir: &Path, kernel: &str, chip: &str, hash: &str, a: Artifact) -> PathBuf {
    dir.join(format!("{kernel}_{chip}_{hash}{}", a.extension()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_tokens() {
        let k = Keep::parse("ptx,cubin sass").unwrap();
        assert!(k.wants(Artifact::Ptx) && k.wants(Artifact::Cubin) && k.wants(Artifact::Sass));
        assert!(!k.wants(Artifact::Mlir) && k.get(Artifact::Mlir).is_none());
        assert!(Keep::parse("all").unwrap().wants(Artifact::Llvm));
        assert!(Keep::parse("ir-debug").unwrap().wants(Artifact::Mlir));
        assert!(Keep::parse("IR").unwrap().wants(Artifact::Ir));
        let e = Keep::parse("ptx,asm").unwrap_err();
        assert!(e.message().contains("`asm`"), "{e}");
        assert!(!Keep::default().wants(Artifact::Ptx));
    }

    #[test]
    fn names_carry_a_hash() {
        let a = short_hash(b"one");
        assert_eq!(a.len(), 8);
        assert_ne!(a, short_hash(b"two"));
        assert_eq!(a, short_hash(b"one"));
        let p = artifact_path(Path::new("/d"), "k", "sm_80", &a, Artifact::Ptx);
        assert_eq!(p, Path::new("/d").join(format!("k_sm_80_{a}.ptx")));
    }
}
