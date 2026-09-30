//! A reference counted allocation holding a header and room for `cap`
//! items, the first `count` initialized. Both kinds of trie node are
//! built on it, so a lookup makes one dependent load per level, and a
//! node no other version holds gains or loses an item in place.

use std::{
    alloc::{self, Layout},
    marker::PhantomData,
    mem,
    ptr::{self, NonNull},
    slice,
    sync::atomic::{
        AtomicUsize,
        Ordering::{Acquire, Relaxed, Release},
        fence,
    },
};

#[repr(C)]
struct Head<H> {
    rc: AtomicUsize,
    count: u32,
    cap: u32,
    h: H,
}

pub(crate) struct Raw<H, T> {
    ptr: NonNull<Head<H>>,
    own: PhantomData<(H, T)>,
}

// SAFETY: a `Raw` is shared like an `Arc<(H, [T])>`, and mutated only
// through `&mut` when its count shows no other holder.
unsafe impl<H: Send + Sync, T: Send + Sync> Send for Raw<H, T> {}
unsafe impl<H: Send + Sync, T: Send + Sync> Sync for Raw<H, T> {}

impl<H, T> Raw<H, T> {
    fn layout(cap: usize) -> (Layout, usize) {
        let items = Layout::array::<T>(cap).expect("node size");
        let (l, off) = Layout::new::<Head<H>>().extend(items).expect("node size");
        (l.pad_to_align(), off)
    }

    fn head(&self) -> &Head<H> {
        // SAFETY: the head is initialized for the node's life.
        unsafe { self.ptr.as_ref() }
    }

    fn base(&self) -> *mut T {
        // SAFETY: the items start `off` bytes into the allocation.
        unsafe { self.ptr.as_ptr().cast::<u8>().add(Self::layout(0).1).cast() }
    }

    fn count(&self) -> usize {
        self.head().count as usize
    }

    fn cap(&self) -> usize {
        self.head().cap as usize
    }

    /// An allocation for `cap` items holding none yet.
    fn alloc(h: H, cap: usize) -> Self {
        let layout = Self::layout(cap).0;
        // SAFETY: the layout has a nonzero size, the head's.
        let raw = unsafe { alloc::alloc(layout) }.cast::<Head<H>>();
        let Some(ptr) = NonNull::new(raw) else { alloc::handle_alloc_error(layout) };
        let head = Head { rc: AtomicUsize::new(1), count: 0, cap: cap as u32, h };
        // SAFETY: freshly allocated for a head.
        unsafe { ptr.write(head) };
        Self { ptr, own: PhantomData }
    }

    /// A node of `h` and the first `cap` of `items`, with room for `cap`.
    pub(crate) fn new(h: H, cap: usize, items: impl IntoIterator<Item = T>) -> Self {
        let mut node = Self::alloc(h, cap);
        /// The items written so far, dropped if `items` unwinds; the
        /// node, still empty, then frees itself.
        struct Written<T>(*mut T, usize);
        impl<T> Drop for Written<T> {
            fn drop(&mut self) {
                // SAFETY: the first `self.1` items were written.
                unsafe {
                    ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.0, self.1))
                }
            }
        }
        let mut written = Written(node.base(), 0);
        for item in items.into_iter().take(cap) {
            // SAFETY: the node is ours alone and has room for `cap`.
            unsafe { written.0.add(written.1).write(item) };
            written.1 += 1;
        }
        let count = written.1 as u32;
        mem::forget(written);
        // SAFETY: the node is ours alone, and `count` items are written.
        unsafe { node.ptr.as_mut().count = count };
        node
    }

    pub(crate) fn header(&self) -> &H {
        &self.head().h
    }

    pub(crate) fn items(&self) -> &[T] {
        // SAFETY: the first `count` items are initialized.
        unsafe { slice::from_raw_parts(self.base(), self.count()) }
    }

    /// The allocation's address, equal for two handles to one node.
    pub(crate) fn addr(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }

    fn is_unique(&self) -> bool {
        self.head().rc.load(Acquire) == 1
    }

    /// This node to change, first copied if another version holds it.
    pub(crate) fn make_mut(&mut self) -> RawMut<'_, H, T>
    where
        H: Clone,
        T: Clone,
    {
        if !self.is_unique() {
            *self = Self::new(
                self.header().clone(),
                self.count(),
                self.items().iter().cloned(),
            );
        }
        RawMut(self)
    }

    /// Adds `item` after the others.
    pub(crate) fn push(&mut self, item: T)
    where
        H: Clone,
        T: Clone,
    {
        if self.is_unique() {
            return RawMut(self).push(item);
        }
        let items = self.items().iter().cloned().chain([item]);
        *self = Self::new(self.header().clone(), self.count() + 1, items);
    }

    /// Removes the item at `i`, moving the last item into its place.
    pub(crate) fn swap_remove(&mut self, i: usize) -> T
    where
        H: Clone,
        T: Clone,
    {
        if self.is_unique() {
            return RawMut(self).swap_remove(i);
        }
        let s = self.items();
        let (last, init) = s.split_last().expect("an item to remove");
        let removed = s[i].clone();
        let items = init.iter().enumerate().map(|(j, x)| if j == i { last } else { x });
        *self = Self::new(self.header().clone(), s.len() - 1, items.cloned());
        removed
    }

    /// The items, moved out when no other version holds the node, which
    /// is then left empty, else cloned.
    pub(crate) fn take_items(&mut self) -> Vec<T>
    where
        T: Clone,
    {
        if !self.is_unique() {
            return self.items().to_vec();
        }
        let n = self.count();
        let mut v = Vec::with_capacity(n);
        // SAFETY: the node is ours alone; its items move to `v`, and its
        // count drops to zero so they are not dropped again.
        unsafe {
            ptr::copy_nonoverlapping(self.base(), v.as_mut_ptr(), n);
            self.ptr.as_mut().count = 0;
            v.set_len(n);
        }
        v
    }
}

impl<H, T> Clone for Raw<H, T> {
    fn clone(&self) -> Self {
        if self.head().rc.fetch_add(1, Relaxed) > isize::MAX as usize {
            std::process::abort()
        }
        Self { ptr: self.ptr, own: PhantomData }
    }
}

impl<H, T> Drop for Raw<H, T> {
    fn drop(&mut self) {
        if self.head().rc.fetch_sub(1, Release) != 1 {
            return;
        }
        fence(Acquire);
        let layout = Self::layout(self.cap()).0;
        // SAFETY: this was the last handle; the head and the first
        // `count` items are initialized and dropped once, then the
        // allocation is freed.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.base(), self.count()));
            ptr::drop_in_place(&mut (*self.ptr.as_ptr()).h);
            alloc::dealloc(self.ptr.as_ptr().cast(), layout);
        }
    }
}

/// A node no other version holds.
pub(crate) struct RawMut<'a, H, T>(&'a mut Raw<H, T>);

impl<'a, H, T> RawMut<'a, H, T> {
    pub(crate) fn header(&mut self) -> &mut H {
        // SAFETY: no other handle exists.
        unsafe { &mut self.0.ptr.as_mut().h }
    }

    pub(crate) fn items(&mut self) -> &mut [T] {
        // SAFETY: no other handle exists, and the first `count` items
        // are initialized.
        unsafe { slice::from_raw_parts_mut(self.0.base(), self.0.count()) }
    }

    pub(crate) fn parts(&mut self) -> (&mut H, &mut [T]) {
        // SAFETY: no other handle exists; the header and the items are
        // disjoint parts of the allocation.
        unsafe {
            let items = slice::from_raw_parts_mut(self.0.base(), self.0.count());
            (&mut self.0.ptr.as_mut().h, items)
        }
    }

    pub(crate) fn into_items(self) -> &'a mut [T] {
        // SAFETY: as `items`, for the life of the borrow.
        unsafe { slice::from_raw_parts_mut(self.0.base(), self.0.count()) }
    }

    /// Moves the node to an allocation twice the size.
    fn grow(&mut self) {
        let count = self.0.count();
        let cap = (count + 1).next_power_of_two();
        // SAFETY: both nodes are ours alone. The header and items move to
        // the new node, and the old one is freed without dropping them;
        // no code between can unwind.
        unsafe {
            let h = ptr::read(&self.0.ptr.as_ref().h);
            let mut new = Raw::<H, T>::alloc(h, cap);
            ptr::copy_nonoverlapping(self.0.base(), new.base(), count);
            new.ptr.as_mut().count = count as u32;
            let old = mem::replace(self.0, new);
            alloc::dealloc(old.ptr.as_ptr().cast(), Raw::<H, T>::layout(old.cap()).0);
            mem::forget(old);
        }
    }

    /// Adds `item` after the others.
    pub(crate) fn push(&mut self, item: T) {
        let count = self.0.count();
        if count == self.0.cap() {
            self.grow()
        }
        // SAFETY: the node is ours alone and has room; writing the first
        // free slot makes one more item.
        unsafe {
            self.0.base().add(count).write(item);
            self.0.ptr.as_mut().count += 1;
        }
    }

    /// Removes the item at `i`, moving the last item into its place.
    pub(crate) fn swap_remove(&mut self, i: usize) -> T {
        let count = self.0.count();
        assert!(i < count);
        // SAFETY: the node is ours alone. Item `i` is read out, the last
        // item moves into its slot, and the count drops by one.
        unsafe {
            let base = self.0.base();
            let item = base.add(i).read();
            if i != count - 1 {
                ptr::copy_nonoverlapping(base.add(count - 1), base.add(i), 1);
            }
            self.0.ptr.as_mut().count -= 1;
            item
        }
    }
}
