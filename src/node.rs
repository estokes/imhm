//! The trie's node: one reference counted allocation holding a header
//! and room for up to 32 slots, so a lookup makes one dependent load
//! per level, and a node no other version holds gains or loses a slot
//! in place.

use crate::{Slot, index};
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

/// The number of slots is `bitmap.count_ones()`; slots past it up to
/// `cap` are uninitialized.
#[repr(C)]
struct Header {
    rc: AtomicUsize,
    len: usize,
    bitmap: u32,
    depth: u8,
    cap: u8,
}

pub(crate) struct Node<K, V> {
    ptr: NonNull<Header>,
    own: PhantomData<Slot<K, V>>,
}

// SAFETY: a node is shared like an `Arc<[Slot<K, V>]>`, and mutated only
// through `&mut` when its count shows no other holder.
unsafe impl<K: Send + Sync, V: Send + Sync> Send for Node<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for Node<K, V> {}

fn grown(count: usize) -> usize {
    count.next_power_of_two().min(32)
}

impl<K, V> Node<K, V> {
    fn layout(cap: usize) -> (Layout, usize) {
        let slots = Layout::array::<Slot<K, V>>(cap).expect("at most 32 slots");
        let (l, off) = Layout::new::<Header>().extend(slots).expect("at most 32 slots");
        (l.pad_to_align(), off)
    }

    fn header(&self) -> &Header {
        // SAFETY: the header is initialized for the node's life.
        unsafe { self.ptr.as_ref() }
    }

    fn base(&self) -> *mut Slot<K, V> {
        // SAFETY: the slots start `off` bytes into the allocation.
        unsafe { self.ptr.as_ptr().cast::<u8>().add(Self::layout(0).1).cast() }
    }

    /// An allocation for `cap` slots holding none yet.
    fn alloc(depth: u8, len: usize, cap: usize) -> Self {
        let layout = Self::layout(cap).0;
        // SAFETY: the layout has a nonzero size, the header's.
        let raw = unsafe { alloc::alloc(layout) }.cast::<Header>();
        let Some(ptr) = NonNull::new(raw) else { alloc::handle_alloc_error(layout) };
        let header =
            Header { rc: AtomicUsize::new(1), len, bitmap: 0, depth, cap: cap as u8 };
        // SAFETY: freshly allocated for a header.
        unsafe { ptr.write(header) };
        Self { ptr, own: PhantomData }
    }

    /// A node at `depth` with `len` pairs below it, room for `cap`
    /// slots, and one of `slots` per bit of `bitmap`, in bit order.
    pub(crate) fn new(
        depth: u8,
        bitmap: u32,
        len: usize,
        cap: usize,
        slots: impl IntoIterator<Item = Slot<K, V>>,
    ) -> Self {
        let count = bitmap.count_ones() as usize;
        assert!(count <= cap && cap <= 32);
        let mut node = Self::alloc(depth, len, cap);
        /// The slots written so far, dropped if `slots` unwinds; the
        /// node, still empty, then frees itself.
        struct Written<K, V>(*mut Slot<K, V>, usize);
        impl<K, V> Drop for Written<K, V> {
            fn drop(&mut self) {
                // SAFETY: the first `self.1` slots were written.
                unsafe {
                    ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.0, self.1))
                }
            }
        }
        let mut written = Written(node.base(), 0);
        for slot in slots.into_iter().take(count) {
            // SAFETY: the node is ours alone and has room for `count`.
            unsafe { written.0.add(written.1).write(slot) };
            written.1 += 1;
        }
        assert_eq!(written.1, count, "a slot per bit");
        mem::forget(written);
        // SAFETY: the node is ours alone, and its slots are now written.
        unsafe { node.ptr.as_mut().bitmap = bitmap };
        node
    }

    pub(crate) fn len(&self) -> usize {
        self.header().len
    }

    pub(crate) fn depth(&self) -> u8 {
        self.header().depth
    }

    pub(crate) fn bitmap(&self) -> u32 {
        self.header().bitmap
    }

    fn count(&self) -> usize {
        self.bitmap().count_ones() as usize
    }

    pub(crate) fn slots(&self) -> &[Slot<K, V>] {
        // SAFETY: the first `count` slots are initialized.
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
        self.header().rc.load(Acquire) == 1
    }

    /// Frees a node no other version holds, whose slots were moved out.
    fn free_moved(self) {
        let layout = Self::layout(self.header().cap as usize).0;
        // SAFETY: no other handle exists and the slots were moved out.
        unsafe { alloc::dealloc(self.ptr.as_ptr().cast(), layout) };
        mem::forget(self)
    }

    /// This node to change, first copied if another version holds it.
    pub(crate) fn make_mut(&mut self) -> NodeMut<'_, K, V>
    where
        K: Clone,
        V: Clone,
    {
        if !self.is_unique() {
            let (d, b, l) = (self.depth(), self.bitmap(), self.len());
            *self = Self::new(d, b, l, self.count(), self.slots().iter().cloned());
        }
        NodeMut(self)
    }

    /// Adds `slot` at the free fragment `bit`.
    pub(crate) fn insert_slot(&mut self, bit: u32, slot: Slot<K, V>)
    where
        K: Clone,
        V: Clone,
    {
        if self.is_unique() {
            return NodeMut(self).insert(bit, slot);
        }
        let (b, i) = (self.bitmap(), index(self.bitmap(), bit));
        let (l, r) = self.slots().split_at(i);
        let len = self.len() + slot.len();
        let slots = l.iter().cloned().chain([slot]).chain(r.iter().cloned());
        *self = Self::new(self.depth(), b | bit, len, self.count() + 1, slots);
    }

    /// Removes the slot at fragment `bit`.
    pub(crate) fn remove_slot(&mut self, bit: u32) -> Slot<K, V>
    where
        K: Clone,
        V: Clone,
    {
        if self.is_unique() {
            return NodeMut(self).remove(bit);
        }
        let i = index(self.bitmap(), bit);
        let s = self.slots();
        let removed = s[i].clone();
        let slots = s[..i].iter().chain(&s[i + 1..]).cloned();
        let len = self.len() - removed.len();
        *self =
            Self::new(self.depth(), self.bitmap() & !bit, len, self.count() - 1, slots);
        removed
    }
}

impl<K, V> Clone for Node<K, V> {
    fn clone(&self) -> Self {
        if self.header().rc.fetch_add(1, Relaxed) > isize::MAX as usize {
            std::process::abort()
        }
        Self { ptr: self.ptr, own: PhantomData }
    }
}

impl<K, V> Drop for Node<K, V> {
    fn drop(&mut self) {
        if self.header().rc.fetch_sub(1, Release) != 1 {
            return;
        }
        fence(Acquire);
        let layout = Self::layout(self.header().cap as usize).0;
        // SAFETY: this was the last handle; the first `count` slots are
        // initialized and dropped once, then the allocation is freed.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.base(), self.count()));
            alloc::dealloc(self.ptr.as_ptr().cast(), layout);
        }
    }
}

/// A node no other version holds.
pub(crate) struct NodeMut<'a, K, V>(&'a mut Node<K, V>);

impl<'a, K, V> NodeMut<'a, K, V> {
    pub(crate) fn depth(&self) -> u8 {
        self.0.depth()
    }

    pub(crate) fn bitmap(&self) -> u32 {
        self.0.bitmap()
    }

    fn header_mut(&mut self) -> &mut Header {
        // SAFETY: no other handle exists.
        unsafe { self.0.ptr.as_mut() }
    }

    pub(crate) fn add_len(&mut self, n: usize) {
        self.header_mut().len += n
    }

    pub(crate) fn sub_len(&mut self, n: usize) {
        self.header_mut().len -= n
    }

    pub(crate) fn slots(&mut self) -> &mut [Slot<K, V>] {
        // SAFETY: no other handle exists, and the first `count` slots
        // are initialized.
        unsafe { slice::from_raw_parts_mut(self.0.base(), self.0.count()) }
    }

    pub(crate) fn into_slots(self) -> &'a mut [Slot<K, V>] {
        // SAFETY: as `slots`, for the life of the borrow.
        unsafe { slice::from_raw_parts_mut(self.0.base(), self.0.count()) }
    }

    /// Adds `slot` at the free fragment `bit`, moving the node to a
    /// larger allocation when it is full.
    pub(crate) fn insert(mut self, bit: u32, slot: Slot<K, V>) {
        let count = self.0.count();
        let i = index(self.bitmap(), bit);
        let add = slot.len();
        if count == self.0.header().cap as usize {
            let (d, l) = (self.depth(), self.0.len());
            let mut new = Node::alloc(d, l, grown(count + 1));
            // SAFETY: both nodes are ours alone. The slots move to the
            // new node around `slot` and the old one is freed without
            // dropping them; no code between can unwind.
            unsafe {
                let (src, dst) = (self.0.base(), new.base());
                ptr::copy_nonoverlapping(src, dst, i);
                ptr::copy_nonoverlapping(src.add(i), dst.add(i + 1), count - i);
                new.ptr.as_mut().bitmap = self.bitmap();
            }
            mem::replace(self.0, new).free_moved();
        } else {
            // SAFETY: the node is ours alone and has room: the slots
            // after `i` shift up one, leaving `i` to write.
            unsafe {
                ptr::copy(self.0.base().add(i), self.0.base().add(i + 1), count - i)
            };
        }
        // SAFETY: slot `i` was vacated above.
        unsafe { self.0.base().add(i).write(slot) };
        let h = self.header_mut();
        h.bitmap |= bit;
        h.len += add;
    }

    /// Removes the slot at fragment `bit`.
    pub(crate) fn remove(mut self, bit: u32) -> Slot<K, V> {
        let count = self.0.count();
        let i = index(self.bitmap(), bit);
        // SAFETY: the node is ours alone. Slot `i` is read out and the
        // slots after it shift down over it before the bitmap drops it.
        let slot = unsafe {
            let base = self.0.base();
            let slot = base.add(i).read();
            ptr::copy(base.add(i + 1), base.add(i), count - i - 1);
            slot
        };
        let h = self.header_mut();
        h.bitmap &= !bit;
        h.len -= slot.len();
        slot
    }
}
