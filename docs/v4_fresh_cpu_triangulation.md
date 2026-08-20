# V4 exact fresh-CPU triangulation stage

This is an engineering-only acceleration.  The scientific
`robust_triangulate_associations` implementation and its 25-field output
schema are unchanged.  One temporary packed binary contains the canonical CPU
input tensors.  Fresh `exec`-started Python workers map that file copy-on-write,
consume fixed contiguous Track-index ranges, and publish private shard files.
The parent returns only after every shard succeeds and concatenates them in
the original Track order.  A failed or incomplete run returns no geometry and
its temporary directory is removed, so the same call can be retried safely.

The formal bootstrap and frozen Track-factor runner use two CPU workers for
maps with at least 5,000 Tracks.  Smaller inputs stay serial because process
startup and module loading dominate.  `worker_count=1` remains available for
diagnosis.  This is not a post-CUDA Python `fork`: workers start a fresh module
through the current interpreter and receive no inherited Python/CUDA objects.

On the fixed-size 791-camera, 10,000-Track deterministic oracle, serial time
was 27.664 s and the fresh two-worker stage took 15.621 s (1.77x including
packing, process startup, loading, and concatenation).  All 25 fields were
bitwise equal.  The earlier isolated two-process compute probe reported 2.03x;
the smaller end-to-end number is expected because it now includes the real
stage boundaries.

A real Cambridge/ShopFacade slice with 2,000 frozen Tracks and 68,740
observations was also exact for all 25 fields.  It took 4.867 s serial and
4.915 s fresh-process (0.99x), which supports the 5,000-Track lower bound rather
than claiming a universal speedup.  A negative test lets one worker finish and
forces the other to fail; the call fails closed, discards partial shards, and a
clean retry is bitwise exact.

The stable conflict-aware DSU remains Python.  Its complete 10,000-Track
oracle cost is already only 1.224 s and uses Python integer camera masks, whose
bitwise operations execute in compact C storage.  A Torch C++ extension would
need a custom ragged/fixed-word bitset, pybind ABI and compiler availability,
runtime-identity coverage, deterministic edge-order tests, and a fallback.
The upper bound is under 1.224 s per uncached build while a dense-list attempt
was slower (1.329 s).  There is therefore no measured compilation or memory
advantage that justifies this risk; C++ DSU remains Stop.
