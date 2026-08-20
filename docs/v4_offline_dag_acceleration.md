# V4 offline DAG and deterministic Track acceleration

This change is method-neutral. It does not change candidate construction,
selection, descriptor fusion, localization, or PoseLib. It ports only the
strictly equivalent parts of `b1fb3a3` onto integration baseline `791af97` and
adds an opt-in content-addressed cache for the completed mapping Observation,
Track, and Geometry node.

The three historical acceleration commits cannot be cherry-picked wholesale:
the integration branch deleted their old render-only runner, and the oldest
commit also carries the subsequently rejected fixed-camera nonlinear point
refinement. The retained changes are GPU-resident immutable descriptor tables,
stable ordered scalar materialization, integer camera-membership bitsets,
vectorized pair-to-Track CSR construction, one batched observation-ray pass,
and reuse of the final projection.

The DAG is disabled unless `--artifact-cache` is explicit. Its key binds the
mapping-only camera schedule and RGB hashes (test RGB is not opened), the
mapping-mask subset, resolved Gaussian PLY, Gaussian manifest when present,
SuperPoint checkpoint, canonical config, and exact source-file hashes for the
node producer. Cache nodes have schema/version checks, SHA and size checks for
every artifact, atomic-last publication, a root publish lock, and configurable
hard node/store limits (8/20 GiB defaults). Publication attempts a reflink and
falls back to a byte copy. It never evicts data implicitly.

On a real 531,439,443-byte ShopFacade Observation/Track/Geometry artifact set,
key hashing took 0.531 s, cold byte-copy publication plus verification took
2.922 s, a verified hit took 0.519 s, and path materialization took 9 us. A
second 490,572,506-byte run explicitly recorded `byte_copy`, 1.755 s cold and
0.454 s hit verification. These costs replace the historical 75.23 s optimized
or 768.10 s unoptimized Shop Track-map build when only downstream policy changes.

A fixed synthetic replay was run once with the untouched `791af97` module and
once with this branch. Track construction fell from 2.673 s to 1.224 s (2.18x),
with Track rows, diagnostics, and the complete pair sidecar bitwise equal.
Ten-thousand-Track triangulation fell from 31.127 s to 28.740 s (1.08x), with
all 25 formal integration geometry fields bitwise equal. The four extra fields
reported in older render-branch evidence belong to the rejected nonlinear
refinement and are intentionally absent. Historical render-only full-map
artifacts are not direct integration-oracle inputs because camera-pair policy
code differs across those branches.

The complete integration CPU suite passes: 555 passed, one CUDA renderer smoke
test skipped by its explicit environment gate.

Two remaining structural options were bounded rather than enabled blindly. A
dense-list DSU was bitwise exact but slowed the same Track oracle from 1.224 s
to 1.329 s, so it was reverted. Splitting the 10k-Track geometry oracle into
two fixed landmark ranges reduced 28.740 s to 14.153 s (2.03x), with all fields
still bitwise equal. That benchmark used a clean CPU `fork`; forking the real
builder after it has initialized CUDA is unsafe. The viable production form is
a separate fresh CPU triangulation stage with shared read-only packed arrays,
not an in-process fork hidden inside the current GPU builder.
