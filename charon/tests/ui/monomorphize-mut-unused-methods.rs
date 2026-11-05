//@ charon-arg=--monomorphize-mut=except-types
//@ charon-arg=--remove-adt-clauses
//@ charon-args=--remove-associated-types=*
//! An instantiation of a trait declaration only keeps the methods used on it.
trait Counter {
    fn get(&mut self) -> u32;
    fn twice(&mut self) -> u32 {
        self.get() + self.get()
    }
    fn thrice(&mut self) -> u32 {
        self.get() * 3
    }
}

struct Shared(u32);
impl Counter for Shared {
    fn get(&mut self) -> u32 {
        self.0
    }
}

struct Mut<'a>(&'a mut u32);
impl<'a> Counter for Mut<'a> {
    fn get(&mut self) -> u32 {
        *self.0 += 1;
        *self.0
    }
}

fn use_thrice<C: Counter>(c: &mut C) -> u32 {
    c.thrice()
}

fn use_twice<C: Counter>(c: &mut C) -> u32 {
    c.twice()
}

pub fn f(x: &mut u32) -> u32 {
    use_thrice(&mut Shared(1)) + use_twice(&mut Mut(x))
}
