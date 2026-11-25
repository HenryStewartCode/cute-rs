// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runtime fields of MMA atoms: CuTe DSL's
//! `Atom.set(field, value)`, `Atom.get(field)` and `Atom.with_(...)` on
//! [`MmaAtom`] and [`TiledMma`].
//!
//! Some MMA instructions take runtime state besides their operands: whether
//! to accumulate into C or overwrite it (`warpgroup.Field.ACCUMULATE`,
//! `tcgen05.Field.ACCUMULATE`), operand negation, block-scale factors, sparse
//! metadata. CuTe DSL keeps it in the atom as an SSA value, which a kernel
//! sets before or inside its K loop (`tiled_mma.set(ACCUMULATE, False)`
//! before it, `True` after the first k-block). cute-rs does the same: the
//! atom holds a [`Boolean`] per field it takes, which the instructions that
//! take them read in [`gemm`](crate::gemm) ([`MmaAtom::field_values`]), and
//! a [`TiledMma`] (or [`MmaAtom`]) is
//! [`Carry`]: threaded through [`for_range_carry`](crate::for_range_carry),
//! a field set inside the loop is carried to the next iteration and out of
//! it, as CuTe DSL's loop-carried atom is. Only the field values are
//! carried; every field the atom takes is (an unset one as `false`).
//!
//! Which fields an atom takes comes from its operation. The `mma.sync`
//! atoms of cute-rs take none; the WGMMA atoms
//! take [`MmaField::ACCUMULATE`], and the tcgen05 atoms (m10) the rest.

use std::fmt;

use crate::atom::mma::{MmaAtom, TiledMma};
use crate::control::Carry;
use crate::ir::ValueId;
use crate::value::{Boolean, IntoValue, Value};

/// A runtime field of an MMA atom: the members of CuTe DSL's
/// `warpgroup.Field` (`ACCUMULATE`), `tcgen05.Field` (`ACCUMULATE`,
/// `NEGATE_A`, `NEGATE_B`, `SFA`, `SFB`, `METADATA`,
/// `DISABLE_OUTPUT_LANE`) and `warp.Field` (`ACCUMULATE`, `SFA`, `SFB`), in
/// one enum. `ACCUMULATE`, `NEGATE_A` and `NEGATE_B` are booleans; the
/// others take pointers and tensors, which arrive with the tcgen05 and SM120
/// block-scaled atoms (m10).
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MmaField {
    ACCUMULATE,
    NEGATE_A,
    NEGATE_B,
    SFA,
    SFB,
    METADATA,
    DISABLE_OUTPUT_LANE,
}

impl MmaField {
    pub const ALL: [MmaField; 7] = [
        MmaField::ACCUMULATE,
        MmaField::NEGATE_A,
        MmaField::NEGATE_B,
        MmaField::SFA,
        MmaField::SFB,
        MmaField::METADATA,
        MmaField::DISABLE_OUTPUT_LANE,
    ];

    /// Whether the field holds a boolean (`ACCUMULATE`, `NEGATE_A`,
    /// `NEGATE_B`).
    pub fn is_boolean(self) -> bool {
        matches!(self, MmaField::ACCUMULATE | MmaField::NEGATE_A | MmaField::NEGATE_B)
    }

    fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for MmaField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Field.{self:?}")
    }
}

/// The fields an atom takes and the values set so far.
#[derive(Clone, Debug, Default)]
pub(crate) struct MmaFields {
    admissible: &'static [MmaField],
    values: [Option<Boolean>; 7],
    /// The tensor fields SFA and SFB of a block-scaled warp MMA
    /// ([`MmaAtom::set_tensor`]).
    pub(crate) tensors: [Option<crate::atom::mma::ScaleOperand>; 2],
    /// The `{disable-output-lane}` words (Int32) of a
    /// tcgen05 MMA ([`MmaAtom::set_disable_output_lane`]).
    pub(crate) disable_output_lane: Option<Vec<ValueId>>,
}

impl MmaFields {
    /// The fields of an atom taking `admissible`, none set.
    pub(crate) fn taking(admissible: &'static [MmaField]) -> MmaFields {
        MmaFields { admissible, values: [None; 7], tensors: [None, None], disable_output_lane: None }
    }
}

/// Fields carried through a loop only when set before it
/// (so atoms that never set them carry what they did before).
fn carried_when_set(f: MmaField) -> bool {
    matches!(f, MmaField::NEGATE_A | MmaField::NEGATE_B | MmaField::DISABLE_OUTPUT_LANE)
}

impl MmaAtom {
    /// The runtime fields this atom takes (none for the `mma.sync` atoms).
    pub fn fields(&self) -> &'static [MmaField] {
        self.fields.admissible
    }

    #[track_caller]
    fn admit(&self, field: MmaField, what: &str) {
        if !self.fields.admissible.contains(&field) {
            let takes = if self.fields.admissible.is_empty() {
                "it has no runtime fields".to_string()
            } else {
                let names: Vec<String> = self.fields.admissible.iter().map(|f| f.to_string()).collect();
                format!("its fields are {}", names.join(", "))
            };
            panic!("cute: {what}({field}) on the {:?} MMA atom: {field} is not one of its fields; {takes}", self.op());
        }
    }

    /// Set runtime field `field` (CuTe DSL's `atom.set(field, value)`), a
    /// boolean one: [`MmaField::ACCUMULATE`], `NEGATE_A` or `NEGATE_B`.
    /// Panics if the atom does not take it.
    #[track_caller]
    pub fn set(&mut self, field: MmaField, value: impl IntoValue<bool>) {
        self.admit(field, "set");
        assert!(
            field != MmaField::DISABLE_OUTPUT_LANE,
            "cute: set({field}): the disable-output-lane mask is a vector of Int32 words, not a Boolean: use \
             set_disable_output_lane"
        );
        assert!(
            field.is_boolean(),
            "cute: set({field}): the pointer and tensor fields of tcgen05 and block-scaled MMAs arrive with m10"
        );
        if let crate::atom::mma::MmaOp::Sm100(kind) = self.op()
            && matches!(field, MmaField::NEGATE_A | MmaField::NEGATE_B)
        {
            assert!(
                kind.kind != crate::ir::UmmaDataKind::I8,
                "cute: set({field}) on {}: the PTX ISA's .kind::i8 instruction descriptor has no negation",
                kind.ptx()
            );
        }
        self.fields.values[field.index()] = Some(value.into_value());
    }

    /// Set the `{disable-output-lane}` mask of a tcgen05 MMA (CuTe DSL's
    /// `atom.set(tcgen05.Field.DISABLE_OUTPUT_LANE, value)`): 4 Int32 words for `cta_group::1` (8 for `::2`). Bit
    /// `i % 32` of word `i / 32` set leaves TMEM lane `i` of D unwritten by
    /// the MMAs [`gemm`](crate::gemm) issues (PTX ISA 9.7.18.10). Panics if
    /// the atom does not take the field.
    #[track_caller]
    pub fn set_disable_output_lane<I: IntoValue<i32> + Copy>(&mut self, mask: &[I]) {
        let field = MmaField::DISABLE_OUTPUT_LANE;
        self.admit(field, "set");
        let crate::atom::mma::MmaOp::Sm100(kind) = self.op() else { unreachable!("only tcgen05 MMAs take {field}") };
        assert!(
            mask.len() == kind.disable_lane_words(),
            "cute: set({field}) on {}: the mask has {} words; it takes {} Int32 words (4 for CtaGroup.ONE, 8 for \
             CtaGroup.TWO)",
            kind.ptx(),
            mask.len(),
            kind.disable_lane_words()
        );
        let words = mask.iter().map(|&w| w.into_value().checked()).collect();
        self.fields.disable_output_lane = Some(words);
    }

    /// The `{disable-output-lane}` words set, if any.
    pub(crate) fn disable_output_lane(&self) -> Option<&[ValueId]> {
        self.fields.disable_output_lane.as_deref()
    }

    /// The value of runtime field `field` (CuTe DSL's `atom.get(field)`):
    /// what [`set`](Self::set) gave it, `false` if nothing did.
    #[track_caller]
    pub fn get(&self, field: MmaField) -> Boolean {
        self.admit(field, "get");
        assert!(
            field != MmaField::DISABLE_OUTPUT_LANE,
            "cute: get({field}): a write-only field; the disable-output-lane mask can only be set \
             (set_disable_output_lane, or gemm_disable_output_lane) and passed to the MMA"
        );
        assert!(
            field.is_boolean(),
            "cute: get({field}): the pointer and tensor fields of tcgen05 and block-scaled MMAs arrive with m10"
        );
        self.fields.values[field.index()].unwrap_or_else(|| false.into_value())
    }

    /// A copy of the atom with `field` set to `value` (CuTe DSL's
    /// `atom.with_(...)` for one field).
    #[track_caller]
    pub fn with_(&self, field: MmaField, value: impl IntoValue<bool>) -> MmaAtom {
        let mut out = self.clone();
        out.set(field, value);
        out
    }

    /// The values of the fields the atom takes, in [`MmaField::ALL`]'s
    /// order, `None` for those not set: what [`gemm`](crate::gemm) gives the
    /// instructions that take fields (none of the `mma.sync` atoms does).
    pub fn field_values(&self) -> Vec<(MmaField, Option<Boolean>)> {
        self.fields.admissible.iter().map(|&f| (f, self.fields.values[f.index()])).collect()
    }

    /// Make the atom take `fields`, for testing the field mechanism before an
    /// atom that takes them exists. Not part of the API.
    #[doc(hidden)]
    pub fn __admit_fields_for_tests(mut self, fields: &'static [MmaField]) -> MmaAtom {
        self.fields = MmaFields::taking(fields);
        self
    }
}

impl TiledMma {
    /// The runtime fields the tiled atom takes.
    pub fn fields(&self) -> &'static [MmaField] {
        self.atom.fields()
    }

    /// [`MmaAtom::set`] on the tiled atom (CuTe DSL's `tiled_mma.set(field,
    /// value)`).
    #[track_caller]
    pub fn set(&mut self, field: MmaField, value: impl IntoValue<bool>) {
        self.atom.set(field, value)
    }

    /// [`MmaAtom::get`] on the tiled atom.
    #[track_caller]
    pub fn get(&self, field: MmaField) -> Boolean {
        self.atom.get(field)
    }

    /// [`MmaAtom::set_disable_output_lane`] on the tiled atom.
    #[track_caller]
    pub fn set_disable_output_lane<I: IntoValue<i32> + Copy>(&mut self, mask: &[I]) {
        self.atom.set_disable_output_lane(mask)
    }

    /// [`MmaAtom::with_`] on the tiled atom.
    #[track_caller]
    pub fn with_(&self, field: MmaField, value: impl IntoValue<bool>) -> TiledMma {
        let mut out = self.clone();
        out.set(field, value);
        out
    }

    /// [`MmaAtom::field_values`] of the tiled atom.
    pub fn field_values(&self) -> Vec<(MmaField, Option<Boolean>)> {
        self.atom.field_values()
    }

    /// Make the tiled atom take `fields` (see
    /// [`MmaAtom::__admit_fields_for_tests`]). Not part of the API.
    #[doc(hidden)]
    pub fn __admit_fields_for_tests(mut self, fields: &'static [MmaField]) -> TiledMma {
        self.atom = self.atom.__admit_fields_for_tests(fields);
        self
    }
}

/// Every field the atom takes, in order, an unset one as `false`; the
/// tcgen05 fields `NEGATE_A`, `NEGATE_B` and `DISABLE_OUTPUT_LANE` only when set before the loop.
impl Carry for MmaAtom {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        for &f in self.fields.admissible {
            if f == MmaField::DISABLE_OUTPUT_LANE {
                out.extend(self.fields.disable_output_lane.iter().flatten());
            } else if !carried_when_set(f) || self.fields.values[f.index()].is_some() {
                out.push(self.fields.values[f.index()].unwrap_or_else(|| false.into_value()).checked());
            }
        }
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        let mut out = self.clone();
        let mut next = || values.next().expect("carry: too few values for the MMA atom's fields");
        for &f in self.fields.admissible {
            if f == MmaField::DISABLE_OUTPUT_LANE {
                if let Some(words) = &self.fields.disable_output_lane {
                    out.fields.disable_output_lane = Some(words.iter().map(|_| next()).collect());
                }
            } else if !carried_when_set(f) || self.fields.values[f.index()].is_some() {
                out.fields.values[f.index()] = Some(Value::from_id(next()));
            }
        }
        out
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        if self.op() != other.op() || self.fields.admissible != other.fields.admissible {
            return Some(format!("an MMA atom {:?} became {:?}", self.op(), other.op()));
        }
        let set = |a: &MmaAtom, f: MmaField| {
            if f == MmaField::DISABLE_OUTPUT_LANE {
                a.fields.disable_output_lane.is_some()
            } else {
                a.fields.values[f.index()].is_some()
            }
        };
        self.fields.admissible.iter().find(|&&f| carried_when_set(f) && set(self, f) != set(other, f)).map(|f| {
            format!(
                "the MMA atom's field {f} is set inside the loop but not before it: set it before the loop too, so \
                 that the loop carries it"
            )
        })
    }
}

/// The tiled atom's fields (see [`MmaAtom`]'s `Carry`).
impl Carry for TiledMma {
    #[track_caller]
    fn flatten(&self, out: &mut Vec<ValueId>) {
        self.atom.flatten(out)
    }

    fn rebuild(&self, values: &mut dyn Iterator<Item = ValueId>) -> Self {
        let mut out = self.clone();
        out.atom = self.atom.rebuild(values);
        out
    }

    fn structure_mismatch(&self, other: &Self) -> Option<String> {
        self.atom.structure_mismatch(&other.atom)
    }
}
