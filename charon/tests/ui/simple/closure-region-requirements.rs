// The closure returns a borrow derived from its argument: its body requires the region of its
// input to outlive the region of its output, which rustc only propagates to the creator.
pub enum Kind {
    Definition(u32),
    Axiom,
}

pub struct Decl {
    pub kind: Kind,
}

pub fn class(decls: &[Decl], n: usize) -> u32 {
    match decls.get(n).map(|d| &d.kind) {
        Some(Kind::Definition(x)) => *x,
        _ => 0,
    }
}

// A requirement on a region of the creator's signature.
pub fn first<'a>(x: &'a u32) -> Option<&'a u32> {
    Some(x).map(|r| r)
}
