//@ charon-arg=--monomorphize-mut=except-types
//@ charon-arg=--remove-adt-clauses
//@ charon-args=--remove-associated-types=* --hide-marker-traits
//@ charon-args=--extract-opaque-bodies --opaque core --opaque alloc
//@ charon-args=--include core::option::Option --include core::option::_::map
// Closures that capture a `&mut` are passed to specialized copies of the generic helpers, which
// call the closure's methods directly.

pub struct Eng<'a> {
    pub t: &'a mut Vec<u32>,
    pub d: u32,
}

impl<'a> Eng<'a> {
    fn telescope(&mut self, n: u32, mut next: impl FnMut(&mut Self, u32) -> bool) -> u32 {
        let mut i = 0;
        let mut c = 0;
        while i < n {
            if next(self, i) {
                c += 1;
            }
            i += 1;
        }
        c
    }

    pub fn count(&mut self, n: u32) -> (u32, bool) {
        let mut seen = false;
        let c = self.telescope(n, |e, i| {
            e.t.push(i);
            if i == 2 {
                seen = true;
            }
            i % 2 == 0
        });
        (c, seen)
    }

    fn f(&mut self, x: u32) -> u32 {
        self.d += x;
        self.d
    }

    pub fn opt(&mut self, o: Option<u32>) -> Option<u32> {
        o.map(|x| self.f(x))
    }

    fn guarded<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.d += 1;
        f(self)
    }

    // A closure that captures no `&mut` isn't baked in: the bound of the helper stays about the
    // generic `FnOnce`, which the closure implements.
    pub fn use_guarded(&mut self, x: u32) -> u32 {
        self.guarded(|e| e.f(x))
    }
}

fn tel(n: u32, mut f: impl FnMut(u32)) {
    let mut i = 0;
    while i < n {
        f(i);
        i += 1;
    }
}

// The same helper with two different closures, in a loop.
pub fn two_closures(n: u32) -> (u32, u32) {
    let mut a = 0;
    let mut b = 0;
    let mut k = 0;
    while k < 2 {
        tel(n, |i| a += i);
        tel(n, |i| b += 2 * i);
        k += 1;
    }
    (a, b)
}

// A closure capturing a closure that captures a `&mut`.
pub fn nested(n: u32) -> u32 {
    let mut c = 0;
    let mut inner = |i: u32| c += i;
    tel(n, move |i| inner(i + 1));
    c
}

// A recursive helper reuses its specialization.
fn rec<F: FnMut(u32)>(n: u32, f: &mut F) {
    if n > 0 {
        f(n);
        rec(n - 1, f)
    }
}
pub fn use_rec(n: u32) -> u32 {
    let mut s = 0;
    rec(n, &mut |i| s += i);
    s
}

// Passing the closure on as `FnOnce` uses the `FnOnce` impl of the closure.
fn once<G: FnOnce(u32) -> u32>(g: G) -> u32 {
    g(7)
}
fn through<F: FnMut(u32) -> u32>(mut f: F) -> u32 {
    f(1);
    once(f)
}
pub fn use_through(x: &mut u32) -> u32 {
    through(|i| {
        *x += i;
        *x
    })
}

// A closure in a generic function.
pub fn generic<T: Copy>(v: &mut Vec<T>, x: T, n: u32) {
    tel(n, |_| v.push(x));
}
