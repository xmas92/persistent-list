# persistent-list

## A singly-linked persistent thread safe list

[`List`] is a basic singly-linked list which uses
structural sharing and [`Arc`] + [clone-on-write](`std::sync::Arc::make_mut`)
mechanics to provide a persistent thread safe list.

Because of the the structure is only cloned when it needs
too it has relatively little overhead when no structural sharing
occurs.

### Immutability

Purist would probably never call this structure immutable as there many
provided ways to modify the underlying data. However with respect to rusts
strict mutability and borrowing mechanics this crate provides a way to have
a persistent data structure which can share underlying memory / state, while
still appearing immutable to everyone sharing. Even if somewhere some instance
is declared as mutable and starts modifying their view.

Much inspiration was taken from the [`im`](http://immutable.rs/) crate. It is worth
looking at as it gives both some great motivations for when and why to use these types
of structures as well as providing some excellent implementations of the most important
structural sharing persistent data structures Maps, Sets and Vectors (using [HAMT][hamt],
[RRB trees][rrb-tree] and [B-trees][b-tree])

## Examples

```rust
// list1 and list2 structurally share the data
let list1 = list![1,2,3,4];
let mut list2 = list1.clone();

// they still share a tail even if one starts
// to be modified.
assert_eq!(list2.pop_front(), Some(1));

// Every time around the loop they share less and
// less data
for i in &mut list2 {
    *i *= 2;
}

// Until finally no structural sharing is possible
assert_eq!(list1, list![1,2,3,4]); // unchanged
assert_eq!(list2, list![4,6,8]);   // modified
```

[rrb-tree]: https://infoscience.epfl.ch/record/213452/files/rrbvector.pdf
[hamt]: https://en.wikipedia.org/wiki/Hash_array_mapped_trie
[b-tree]: https://en.wikipedia.org/wiki/B-tree


Current version: 0.1.1

## TODO

This is pretty much the first project I've ever written in rust so I
think there are a lot of things that could be improved upon.

Except for all the mistakes I have inevitably made there are some definitely
improvements that can be made to the testing of all the functionality.

Especially introducing some fuzzing or property testing framework would make it magnitudes 
better.

*Copyright 2020 Axel Boldt-Christmas*

License: MIT OR Apache-2.0
