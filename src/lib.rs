//-
// Copyright 2020 Axel Boldt-Christmas
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # A singly-linked persistent thread safe list
//!
//! [`List`] is a basic singly-linked list which uses
//! structural sharing and [`Arc`] + [clone-on-write](`std::sync::Arc::make_mut`)
//! mechanics to provide a persistent thread safe list.
//!
//! Because of the the structure is only cloned when it needs
//! too it has relatively little overhead when no structural sharing
//! occurs.
//!
//! ## Immutability
//!
//! Purist would probably never call this structure immutable as there many
//! provided ways to modify the underlying data. However with respect to rusts
//! strict mutability and borrowing mechanics this crate provides a way to have
//! a persistent data structure which can share underlying memory / state, while
//! still appearing immutable to everyone sharing. Even if somewhere some instance
//! is declared as mutable and starts modifying their view.
//!
//! Much inspiration was taken from the [`im`](http://immutable.rs/) crate. It is worth
//! looking at as it gives both some great motivations for when and why to use these types
//! of structures as well as providing some excellent implementations of the most important
//! structural sharing persistent data structures Maps, Sets and Vectors (using [HAMT][hamt],
//! [RRB trees][rrb-tree] and [B-trees][b-tree])
//!
//! # Examples
//!
//! ```
//! # #[macro_use] extern crate persistent_list;
//! # use persistent_list::{List, cons};
//! # fn main() {
//! // list1 and list2 structurally share the data
//! let list1 = list![1,2,3,4];
//! let mut list2 = list1.clone();
//!
//! // they still share a tail even if one starts
//! // to be modified.
//! assert_eq!(list2.pop_front(), Some(1));
//!
//! // Every time around the loop they share less and
//! // less data
//! for i in &mut list2 {
//!     *i *= 2;
//! }
//!
//! // Until finally no structural sharing is possible
//! assert_eq!(list1, list![1,2,3,4]); // unchanged
//! assert_eq!(list2, list![4,6,8]);   // modified
//! # }
//! ```
//!
//! [rrb-tree]: https://infoscience.epfl.ch/record/213452/files/rrbvector.pdf
//! [hamt]: https://en.wikipedia.org/wiki/Hash_array_mapped_trie
//! [b-tree]: https://en.wikipedia.org/wiki/B-tree
//!
#[cfg(test)]
extern crate rand;

use std::{
    borrow::Borrow,
    fmt,
    hash::Hash,
    iter::{FromIterator, IntoIterator, Iterator},
    mem,
    sync::Arc,
};

/// Construct a list from a sequence of elements.
///
/// # Examples
///
/// ```
/// #[macro_use] extern crate persistent_list;
/// # use persistent_list::{List, cons};
/// # fn main() {
/// assert_eq!(
///   list![1, 2, 3],
///   List::from(vec![1, 2, 3])
/// );
///
/// assert_eq!(
///   list![1, 2, 3],
///   cons(1, cons(2, cons(3, List::new())))
/// );
/// # }
/// ```
#[macro_export]
macro_rules! list {
    [] => {List::new()};
    [$ele:expr] => {crate::cons($ele, List::new())};
    [$ele:expr, $($tail:expr),*] => {crate::cons($ele, list![$($tail),*])};
    [$ele:expr, $($tail:expr ,)*] => {crate::cons($ele, list![$($tail),*])};
}

/// A singly-linked persistent thread safe list.
#[derive(Clone)]
pub struct List<E> {
    size: usize,
    node: Option<Arc<Node<E>>>,
}

#[derive(Clone)]
struct Node<E>(E, Option<Arc<Node<E>>>);

/// An iterator over the elements of a `List`.
///
/// This `struct` is created by the [`iter`] method on [`List`]. See its
/// documentation for more.
///
/// [`iter`]: struct.List.html#method.iter
/// [`List`]: struct.List.html
#[derive(Clone)]
pub struct Iter<'a, E> {
    node: &'a Option<Arc<Node<E>>>,
    len: usize,
}

impl<'a, E: 'a + fmt::Debug + Clone> fmt::Debug for Iter<'a, E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Iter")
            .field(&self.len)
            .field(&self.node)
            .finish()
    }
}

/// A mutable iterator over the elements of a `List`.
///
/// This `struct` is created by the [`iter_mut`] method on [`List`]. See its
/// documentation for more.
///
/// [`iter_mut`]: struct.List.html#method.iter_mut
/// [`List`]: struct.List.html
pub struct IterMut<'a, E> {
    node: Option<&'a mut Arc<Node<E>>>,
    len: usize,
}

impl<'a, E: 'a + fmt::Debug + Clone> fmt::Debug for IterMut<'a, E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("IterMut")
            .field(&self.node)
            .field(&self.len)
            .finish()
    }
}

/// An owning iterator over the elements of a `List`.
///
/// This `struct` is created by the [`into_iter`] method on [`List`][`List`]
/// (provided by the `IntoIterator` trait). See its documentation for more.
///
/// [`into_iter`]: struct.List.html#method.into_iter
/// [`List`]: struct.List.html
#[derive(Clone)]
pub struct IntoIter<E> {
    list: List<E>,
}

impl<E: fmt::Debug + Clone> fmt::Debug for IntoIter<E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("IntoIter").field(&self.list).finish()
    }
}

impl<E: Clone> Default for List<E> {
    /// Creates an empty `List<E>`.
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Construct a `List` with a new element at the front of the
/// current `List`.
///
/// Alternative to using [`List::cons`], but enables
/// writing list construction from front to back.
///
/// ```
/// # #[macro_use] extern crate persistent_list;
/// # use persistent_list::{List, cons};
/// # fn main() {
/// // Enables this:
/// let list = cons(1, cons(2, List::new()));
///
/// // Instead of
/// let list = List::new().cons(2).cons(1);
///
/// // Or
/// let mut list = List::new();
/// list.push_front(2);
/// list.push_front(1);
///
/// // Which all result in the equivalent
/// let list = list![1, 2];
/// # }
/// ```
///
/// # Examples
///
/// ```
/// #[macro_use] extern crate persistent_list;
/// # use persistent_list::{List, cons};
/// # fn main() {
///
/// assert_eq!(
///   cons(1, cons(2, cons(3, List::new()))),
///   list![1, 2, 3]
/// );
/// # }
/// ```
#[inline]
pub fn cons<E: Clone, T: Borrow<List<E>>>(first: E, rest: T) -> List<E> {
    let mut list: List<E> = rest.borrow().clone();
    // List {
    //     size: list.size + 1,
    //     node: Some(Arc::new(Node(first, list.node))),
    // }
    list.push_front(first);
    list
}

impl<E: Clone> List<E> {
    /// Creates an empty `List`.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let list: List<u32> = List::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            size: 0,
            node: None,
        }
    }

    /// Moves all elements from `other` to the end of the list.
    ///
    /// This reuses all the nodes from `other` and moves them into `self`. After
    /// this operation, `other` becomes empty.
    ///
    /// This operation should compute in O(`self.len()`) time and O(`self.len()`)* memory.
    /// The memory usage depends on how much of the `self` List is shared. Nodes are taken
    /// using clone-on-write mechanics, only cloning any shared tail part.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list1 = List::new();
    /// list1.push_front('a');
    ///
    /// let mut list2 = List::new();
    /// list2.push_front('c');
    /// list2.push_front('b');
    ///
    /// list1.append(&mut list2);
    ///
    /// let mut iter = list1.iter();
    /// assert_eq!(iter.next(), Some(&'a'));
    /// assert_eq!(iter.next(), Some(&'b'));
    /// assert_eq!(iter.next(), Some(&'c'));
    /// assert!(iter.next().is_none());
    ///
    /// assert!(list2.is_empty());
    /// ```
    pub fn append(&mut self, other: &mut Self) {
        if other.node.is_none() {
            return;
        }
        let mut node = &mut self.node;
        while let Some(next) = node {
            node = &mut Arc::make_mut(next).1;
        }
        mem::swap(node, &mut other.node);
        self.size += other.size;
        other.size = 0;
    }

    /// Provides a forward iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list: List<u32> = List::new();
    ///
    /// list.push_front(2);
    /// list.push_front(1);
    /// list.push_front(0);
    ///
    /// let mut iter = list.iter();
    /// assert_eq!(iter.next(), Some(&0));
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, E> {
        Iter {
            node: &self.node,
            len: self.size,
        }
    }

    /// Provides a forward iterator with mutable references.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list: List<u32> = List::new();
    ///
    /// list.push_front(2);
    /// list.push_front(1);
    /// list.push_front(0);
    ///
    /// for element in list.iter_mut() {
    ///     *element += 10;
    /// }
    ///
    /// let mut iter = list.iter();
    /// assert_eq!(iter.next(), Some(&10));
    /// assert_eq!(iter.next(), Some(&11));
    /// assert_eq!(iter.next(), Some(&12));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, E> {
        IterMut {
            node: self.node.as_mut(),
            len: self.size,
        }
    }

    /// Returns `true` if the `List` is empty.
    ///
    /// This operation should compute in O(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list = List::new();
    /// assert!(list.is_empty());
    ///
    /// list.push_front("foo");
    /// assert!(!list.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.node.is_none()
    }

    /// Returns the length of the `List`.
    ///
    /// This operation should compute in O(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list = List::new();
    ///
    /// list.push_front(2);
    /// assert_eq!(list.len(), 1);
    ///
    /// list.push_front(1);
    /// assert_eq!(list.len(), 2);
    ///
    /// list.push_front(3);
    /// assert_eq!(list.len(), 3);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }
    /// Construct a `List` with a single value.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let list = List::unit(1);
    /// assert_eq!(1, list.len());
    /// assert_eq!(
    ///   list.front(),
    ///   Some(&1)
    /// );
    /// ```
    #[inline]
    pub fn unit(first: E) -> Self {
        crate::cons(first, Self::new())
    }

    /// Removes all elements from the `List`.
    ///
    /// This operation should compute in O(n) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list = List::new();
    ///
    /// list.push_front(2);
    /// list.push_front(1);
    /// assert_eq!(list.len(), 2);
    /// assert_eq!(list.front(), Some(&1));
    ///
    /// list.clear();
    /// assert_eq!(list.len(), 0);
    /// assert_eq!(list.front(), None);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Construct a `List` with a new element at the front of the
    /// current `List`.
    ///
    /// See [`crate::cons`] for alternative version.
    #[inline]
    pub fn cons(&self, first: E) -> Self {
        Self {
            size: self.size + 1,
            node: Some(Arc::new(Node(first, self.node.clone()))),
        }
    }

    /// Get the head and the tail of a list.
    ///
    /// This function performs both the `head` function and
    /// the `tail` function in one go, returning a tuple
    /// of the head and the tail, or `None` if the list is
    /// empty.
    ///
    /// # Examples
    ///
    /// This can be useful when pattern matching your way through
    /// a list:
    ///
    /// ```
    /// # #[macro_use] extern crate persistent_list;
    /// use persistent_list::{List, cons};
    /// use std::fmt::Debug;
    ///
    /// fn walk_through_list<E: Clone>(list: &List<E>) where E: Debug {
    ///     match list.uncons() {
    ///         None => (),
    ///         Some((ref head, ref tail)) => {
    ///             print!("{:?}", head);
    ///             walk_through_list(tail)
    ///         }
    ///     }
    /// }
    /// # fn main() {
    /// # }
    /// ```
    #[inline]
    pub fn uncons(&self) -> Option<(&E, Self)> {
        self.head().and_then(|h| self.tail().map(|t| (h, t)))
    }

    /// Returns `true` if the `List` contains an element equal to the
    /// given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list: List<u32> = List::new();
    ///
    /// list.push_front(2);
    /// list.push_front(1);
    /// list.push_front(0);
    ///
    /// assert_eq!(list.contains(&0), true);
    /// assert_eq!(list.contains(&10), false);
    /// ```
    pub fn contains(&self, x: &E) -> bool
    where
        E: PartialEq<E>,
    {
        self.iter().any(|e| e == x)
    }

    /// Provides a reference to the front element, or `None` if the `List` is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list = List::new();
    /// assert_eq!(list.front(), None);
    ///
    /// list.push_front(1);
    /// assert_eq!(list.front(), Some(&1));
    /// ```
    #[inline]
    pub fn front(&self) -> Option<&E> {
        match &self.node {
            Some(node) => Some(&node.0),
            None => None,
        }
    }

    /// Provides a mutable reference to the front element, or `None` if the list
    /// is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list = List::new();
    /// assert_eq!(list.front(), None);
    ///
    /// list.push_front(1);
    /// assert_eq!(list.front(), Some(&1));
    ///
    /// match list.front_mut() {
    ///     None => {},
    ///     Some(x) => *x = 5,
    /// }
    /// assert_eq!(list.front(), Some(&5));
    /// ```
    #[inline]
    pub fn front_mut(&mut self) -> Option<&mut E> {
        self.node.as_mut().map(|node| &mut Arc::make_mut(node).0)
    }

    /// Adds an element first in the list.
    ///
    /// This operation should compute in O(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list = List::new();
    ///
    /// list.push_front(2);
    /// assert_eq!(list.front().unwrap(), &2);
    ///
    /// list.push_front(1);
    /// assert_eq!(list.front().unwrap(), &1);
    /// ```
    pub fn push_front(&mut self, element: E) {
        let node = self.node.take();
        self.node = Some(Arc::new(Node(element, node)));
        self.size += 1;
    }

    /// Removes the first element and returns it, or `None` if the list is
    /// empty.
    ///
    /// This operation should compute in O(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list = List::new();
    /// assert_eq!(list.pop_front(), None);
    ///
    /// list.push_front(1);
    /// list.push_front(3);
    /// assert_eq!(list.pop_front(), Some(3));
    /// assert_eq!(list.pop_front(), Some(1));
    /// assert_eq!(list.pop_front(), None);
    /// ```
    pub fn pop_front(&mut self) -> Option<E> {
        match self.node.take() {
            Some(node) => {
                self.size -= 1;
                match Arc::try_unwrap(node) {
                    Ok(node) => {
                        self.node = node.1;
                        Some(node.0)
                    }
                    Err(node) => {
                        self.node = node.1.clone();
                        Some(node.0.clone())
                    }
                }
            }
            None => None,
        }
    }

    /// Splits the list into two at the given index. Returns everything after the given index,
    /// including the index.
    ///
    /// This operation should compute in O(at) time.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list = List::new();
    ///
    /// list.push_front(1);
    /// list.push_front(2);
    /// list.push_front(3);
    ///
    /// let mut splitted = list.split_off(2);
    ///
    /// assert_eq!(splitted.pop_front(), Some(1));
    /// assert_eq!(splitted.pop_front(), None);
    ///
    /// assert_eq!(list.pop_front(), Some(3));
    /// assert_eq!(list.pop_front(), Some(2));
    /// assert_eq!(list.pop_front(), None);
    /// ```
    pub fn split_off(&mut self, at: usize) -> Self {
        let len = self.len();
        assert!(at <= len, "Cannot split off at a nonexistent index");
        if at == 0 {
            return mem::replace(self, Self::new());
        } else if at == len {
            return Self::new();
        }

        let mut iter = self.iter_mut();
        for _ in 0..at - 1 {
            iter.next();
        }
        match iter.node.take() {
            Some(node) => {
                let node = Arc::make_mut(node);
                match node.1.take() {
                    Some(node) => {
                        self.size = at;
                        List {
                            size: len - at,
                            node: Some(node),
                        }
                    }
                    None => unreachable!(),
                }
            }
            None => unreachable!(),
        }
    }
    /// Split a `List` at a given index.
    ///
    /// Split a `List` at a given index, consuming the `List` and
    /// returning a pair of the left hand side and the right hand side
    /// of the split.
    ///
    /// This operation should compute in O(at) time.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate persistent_list;
    /// # use persistent_list::{List,cons};
    /// # fn main() {
    /// let mut list = list![1, 2, 3, 7, 8, 9];
    /// let (left, right) = list.split_at(3);
    /// assert_eq!(list![1, 2, 3], left);
    /// assert_eq!(list![7, 8, 9], right);
    /// # }
    /// ```
    pub fn split_at(mut self, at: usize) -> (Self, Self) {
        let right = self.split_off(at);
        (self, right)
    }

    /// Reverses the list
    ///
    /// This operation should compute in O(n) time and O(n)* memory allocations, O(1) if
    /// this list does not share any node (tail) with another list.
    ///
    /// The memory usage depends on how much of the `self` List is shared. Nodes are taken
    /// using clone-on-write mechanics, only cloning any shared tail part.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list1 = List::from(vec![1, 2, 3, 4]);
    /// list1.reverse();
    /// assert_eq!(list1, List::from(vec![4, 3, 2, 1]));
    ///
    /// let list2 = list1.clone();
    /// list1.reverse();
    /// assert_eq!(list2, List::from(vec![4, 3, 2, 1]));
    ///
    /// list1.reverse();
    /// assert_eq!(list1, list2);
    /// ```
    #[inline]
    pub fn reverse(&mut self) {
        if self.node.is_none() {
            return;
        }
        // New list new.node == None
        let mut new = Self::new();
        // After take head from self
        // node == head of old list
        // and self.node == None
        let mut node = self.node.take();
        while node.is_some() {
            // current head of the old list tail was not the end
            // swap head of the old list tail with the head of our new list
            mem::swap(&mut new.node, &mut node);
            match &mut new.node {
                // Get inner reference
                Some(new_node) => {
                    // new_node is the head of the old list tail which may be shared.
                    // So we clone-on-write and get a non shared ref mut into our new
                    // head location but with the old list tail data.
                    let new_node = Arc::make_mut(new_node);
                    // swap the next item in the old list tail and or new tail.
                    mem::swap(&mut new_node.1, &mut node);
                    // The local variable node now has the head of the old list tail
                    // new_node == new.node now contains the head of the previous
                    // old list tail and new.node.1 == old new_node.
                }
                None => unreachable!(),
            }
        }
        new.size = self.size;
        *self = new;
    }

    /// Get the rest of the `List` after any potential
    /// front element has been removed.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list = List::new();
    /// assert_eq!(list.rest(), List::new());
    ///
    /// list.push_front(2);
    /// assert_eq!(list.rest(), List::new());
    ///
    /// list.push_front(1);
    /// assert_eq!(list.rest(), List::unit(2));
    /// ```
    #[inline]
    pub fn rest(&self) -> Self {
        match &self.node {
            None => Self::new(),
            Some(node) => Self {
                size: self.size - 1,
                node: node.1.clone(),
            },
        }
    }

    /// Get the first element of a `List`.
    ///
    /// If the `List` is empty, `None` is returned.
    ///
    /// This is an alias for the [`front`][front] method.
    ///
    /// [front]: #method.front
    #[inline]
    #[must_use]
    pub fn head(&self) -> Option<&E> {
        self.front()
    }

    /// Get the tail of a `List`.
    ///
    /// The tail means all elements in the `List` after the
    /// first item. If the list only has one element, the
    /// result is an empty list. If the list is empty, the
    /// result is `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_list::List;
    ///
    /// let mut list = List::new();
    /// assert_eq!(list.tail(), None);
    ///
    /// list.push_front(2);
    /// assert_eq!(list.tail(), Some(List::new()));
    ///
    /// list.push_front(1);
    /// assert_eq!(list.tail(), Some(List::unit(2)));
    /// ```
    #[inline]
    pub fn tail(&self) -> Option<Self> {
        match &self.node {
            None => None,
            Some(node) => Some(Self {
                size: self.size - 1,
                node: node.1.clone(),
            }),
        }
    }

    pub fn skip(&self, count: usize) -> Self {
        if count > self.size {
            Self::new()
        } else {
            let mut rest = &self.node;
            for _ in 0..count {
                match rest {
                    Some(node) => rest = &node.1,
                    None => unreachable!(),
                }
            }
            Self {
                size: self.size - count,
                node: rest.clone(),
            }
        }
    }
}

impl<E: fmt::Debug + Clone> fmt::Debug for List<E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}
impl<E: fmt::Debug + Clone> fmt::Debug for Node<E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("ListNode")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

impl<E: Clone> FromIterator<E> for List<E> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = E>>(iterator: I) -> Self {
        iterator
            .into_iter()
            .fold(Self::new(), |list, element| crate::cons(element, list))
    }
}

impl<'s, 'a, E, OE> From<&'s List<&'a E>> for List<OE>
where
    E: ToOwned<Owned = OE>,
    OE: Borrow<E> + Clone,
{
    fn from(vec: &List<&E>) -> Self {
        let mut list: Self = vec.iter().map(|a| (*a).to_owned()).collect();
        list.reverse();
        list
    }
}

impl<'a, E: Clone> From<&'a [E]> for List<E> {
    fn from(slice: &[E]) -> Self {
        slice.iter().rev().cloned().collect()
    }
}

impl<E: Clone> From<Vec<E>> for List<E> {
    /// Create a `List` from a [`std::vec::Vec`][vec].
    ///
    /// Time: O(n)
    ///
    /// [vec]: https://doc.rust-lang.org/std/vec/struct.Vec.html
    fn from(vec: Vec<E>) -> Self {
        vec.into_iter().rev().collect()
    }
}

impl<'a, E: Clone> From<&'a Vec<E>> for List<E> {
    /// Create a vector from a [`std::vec::Vec`][vec].
    ///
    /// Time: O(n)
    ///
    /// [vec]: https://doc.rust-lang.org/std/vec/struct.Vec.html
    fn from(vec: &Vec<E>) -> Self {
        vec.iter().rev().cloned().collect()
    }
}

impl<E: Clone> IntoIterator for List<E> {
    type Item = E;
    type IntoIter = IntoIter<E>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { list: self }
    }
}

impl<'a, E: Clone> IntoIterator for &'a List<E> {
    type Item = &'a E;
    type IntoIter = Iter<'a, E>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, E: Clone> IntoIterator for &'a mut List<E> {
    type Item = &'a mut E;
    type IntoIter = IterMut<'a, E>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<E: Clone> Iterator for IntoIter<E> {
    type Item = E;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.list.pop_front()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.list.len(), Some(self.list.len()))
    }
}

impl<E: Clone> ExactSizeIterator for IntoIter<E> {}

impl<'a, E> Iterator for Iter<'a, E> {
    type Item = &'a E;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match &self.node {
            Some(node) => {
                self.len -= 1;
                self.node = &node.1;
                Some(&node.0)
            }
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, E> ExactSizeIterator for Iter<'a, E> {}

impl<'a, E: Clone> Iterator for IterMut<'a, E> {
    type Item = &'a mut E;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.node.take() {
            Some(node) => {
                let node = Arc::make_mut(node);
                self.len -= 1;
                self.node = node.1.as_mut();
                Some(&mut node.0)
            }
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, E: Clone> ExactSizeIterator for IterMut<'a, E> {}

impl<E: Hash + Clone> Hash for List<E> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for i in self {
            i.hash(state);
        }
        // Not hashing size for consistency with im::Vector
        // self.len().hash(state);
    }
}

impl<E> Drop for List<E> {
    fn drop(&mut self) {
        let mut node = self.node.take();
        while let Some(next) = node {
            match Arc::try_unwrap(next) {
                Ok(mut next) => {
                    node = next.1.take();
                }
                Err(_) => {
                    break;
                }
            }
        }
    }
}

impl<E: PartialEq + Clone> PartialEq for List<E> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other)
    }
}

impl<T: Eq + Clone> Eq for List<T> {}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn list_macro() {
        assert_eq!(list![1, 2, 3], cons(1, cons(2, cons(3, List::new()))));
        assert_eq!(list![1, 2, 3,], cons(1, cons(2, cons(3, List::new()))));
        assert_eq!(List::<i32>::new(), list![]);
    }

    #[test]
    fn list_iterator() {
        let list1 = list![1, 2, 3, 4];
        let list2: List<String> = list1
            .into_iter()
            .filter(|&x| x > 3)
            .map(|x| x.to_string())
            .collect();

        assert_eq!(list2, list![String::from("4")]);

        let list1 = list![1, 2, 3, 4];
        let list2: List<_> = list1
            .iter().copied()
            .collect();
        assert_eq!(list![4, 3, 2, 1], list2);

        let mut list1 = list![1, 2, 3, 4];
        for i in &mut list1 {
            *i *= 2;
        }
        assert_eq!(list1, list![2, 4, 6, 8]);

        let mut list1 = list![1, 2, 3, 4];
        let list2 = list1.clone();
        for i in &mut list1 {
            *i *= 2;
        }
        assert_eq!(list1, list![2, 4, 6, 8]);
        assert_eq!(list2, list![1, 2, 3, 4]);
    }

    #[test]
    fn list_front() {
        let list1 = list![1];
        let list2 = list![1, 2];
        assert_eq!(list1.front(), list2.front());

        let list1 = list![1];
        let list2 = list![2, 1];
        assert!(list1.front() != list2.front());

        let list: List<i32> = list![];
        assert_eq!(list.front(), None);
    }

    #[test]
    fn list_rest() {
        let list1 = list![1, 2, 3];
        assert_eq!(list1.rest(), list![2, 3]);

        let list1: List<i32> = list![];
        assert_eq!(list1.rest(), list![]);

        let list1 = list![1, 2];
        let list2 = list![1, 2, 3];
        assert!(list1.rest() != list2.rest());

        let list1: List<i32> = list![];
        let list2 = list![1];
        assert!(list1.rest() == list2.rest());
    }

    #[test]
    fn list_length() {
        let list1 = list![1, 2, 3];
        assert_eq!(list1.len(), 3);

        let list1: List<i32> = list![];
        assert_eq!(list1.len(), 0);

        let list1 = list![1, 2];
        let list2 = list![1, 2, 3];
        assert!(list1.len() != list2.len());
    }

    #[test]
    fn list_skip() {
        let list1 = list![1, 2, 3];
        assert_eq!(list1, list1.skip(0));

        let list2 = list![2, 3];
        assert_eq!(list2, list1.skip(1));

        let list2 = list![3];
        assert_eq!(list2, list1.skip(2));

        let list2: List<i32> = list![];
        assert_eq!(list2, list1.skip(3));
    }

    #[test]
    fn list_append() {
        let mut list1 = list![1, 2, 3];
        let mut list2 = list![4, 5];

        list1.append(&mut list2);
        assert_eq!(list1, list![1, 2, 3, 4, 5]);
        assert_eq!(list2, List::new());

        let mut list1 = list![1, 2, 3];
        let mut list2 = list![4, 5];
        let list3 = list1.clone();
        let list4 = list2.rest();

        list1.append(&mut list2);
        assert_eq!(list1, list![1, 2, 3, 4, 5]);
        assert_eq!(list2, List::new());
        assert_eq!(list3, list![1, 2, 3]);
        assert_eq!(list4, list![5]);
    }

    #[test]
    fn list_thread() {
        use crate::rand::RngCore;
        use std::collections::VecDeque;
        use std::thread;
        let size = 10000;
        let list: List<u32> = (0..size).collect();
        let vec: VecDeque<_> = list.iter().cloned().collect();

        let mut threads = Vec::new();

        for i in 0..24 {
            let list = list.clone();
            let vec = vec.clone();
            threads.push(thread::spawn(move || {
                let mut rng = rand::thread_rng();
                let mut list = list;
                let mut vec = vec;
                assert!(list.iter().eq(vec.iter()));
                match i {
                    i if i < 6 => {
                        for i in 0..size {
                            if rng.next_u32() % 2 == 0 {
                                list.pop_front();
                                vec.pop_front();
                            } else {
                                list.push_front(i);
                                vec.push_front(i);
                            }
                        }
                    }
                    i if i < 12 => {
                        for i in 0..size {
                            if i < size / 2 {
                                list.pop_front();
                                vec.pop_front();
                            } else {
                                list.push_front(i);
                                vec.push_front(i);
                            }
                        }
                    }
                    i if i < 18 => {
                        for (i1, i2) in list.iter_mut().zip(vec.iter_mut()) {
                            *i1 *= 2;
                            *i2 *= 2;
                        }
                    }
                    _ => {
                        for _ in 0..(size / 10) {
                            let at = rng.next_u32() % size;
                            let (left, mut right) = list.split_at(at as usize);
                            list = left;
                            list.append(&mut right);
                        }
                    }
                }
                assert!(list.iter().eq(vec.iter()));
                println!("Thread {} done!", i);
            }))
        }
        assert!(list.iter().eq(vec.iter()));
        while let Some(handle) = threads.pop() {
            handle.join().expect("Thread panicked.")
        }
        assert!(list.iter().eq(vec.iter()));
    }
    #[test]
    fn list_split_append() {
        // Recursion test. Fails unless we specify a non recursive Drop
        use crate::rand::RngCore;
        use std::collections::VecDeque;
        let size = 10000;
        let list: List<u32> = (0..size).collect();
        let vec: VecDeque<_> = list.iter().cloned().collect();

        let mut rng = rand::thread_rng();
        let mut list = list;

        for _ in 0..(size / 10) {
            let at = rng.next_u32() % size;
            let (left, mut right) = list.split_at(at as usize);
            list = left;
            list.append(&mut right);
            assert!(list.iter().eq(vec.iter()));
        }
    }
}
