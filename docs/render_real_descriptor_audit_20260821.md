# Render–real descriptor audit (2026-08-21)

This evidence bundle records experiments only. It changes no production code,
configuration, map default, or deployment gate. Large tensor artifacts remain
under `/mnt/pool/sqy`; the repository stores only paths, hashes, protocol
boundaries, and compact conclusions.

## Frozen G+RGB-Desc resource oracle

Real mapping RGB was sampled at the already-frozen rendered keypoint rows. The
Anchor row order, IDs, Track identity, xyz, selected set, and completion rows
were unchanged. Mapping used no test query; official test was evaluated only
after the descriptor arm was frozen.

| Scene | Arm | Median cm | R2 | R5 | >100 cm |
|---|---:|---:|---:|---:|---:|
| Stairs | Ours-G | 2.040 | 49.00% | 87.00% | 4 |
| Stairs | G+RGB-Desc | 1.890 | 53.50% | 88.40% | 3 |
| ShopFacade | Ours-G | 2.039 | 49.51% | 82.52% | 0 |
| ShopFacade | G+RGB-Desc | 2.020 | 49.51% | 84.47% | 0 |

The oracle establishes descriptor-domain headroom, especially on Stairs. It
does not authorize source RGB in the formal source-image-free method.

## Symmetric photometric arms: No-Go

The percentile-grayscale and CLAHE arms were preregistered before test. Both
used Gaussian-rendered mapping evidence only and applied the same frozen
canonicalization to test queries. Percentile reduced R5 in both scenes. CLAHE
reduced ShopFacade R2 and increased the Stairs catastrophic tail. Neither arm
is a deployment candidate.

## Descriptor-gap attribution

The paired observation audit contains 713,302 Stairs and 169,484 ShopFacade
observation rows. Mean render–real cosine was 0.824 and 0.786 respectively.
Paired cosine/self-consistency described the gap, but alpha, depth boundary,
image border, descriptor-grid phase, view angle/bin, and Track/completion type
did not yield a stable cross-scene reliability rule.

Strict mapping-query-local LOO retrieval samples produced zero RGB-repaired
queries (0/63 Stairs and 0/29 ShopFacade). Therefore the official-test Pose
improvement cannot be attributed to improved mapping self-retrieval; it more
likely reflects changed false-match structure and multi-correspondence Pose
consensus. No test result was used to fit an attribution rule.

## Render perturbation stability: Stop

Small exposure, contrast, gamma, and ±0.25-pixel sampling perturbations were
tested as a source-free proxy. The preregistered GO threshold required rho at
least 0.2 plus cross-scene decile consistency. ShopFacade correlations were
0.132/0.107; Stairs correlations were 0.200/0.101 with excess monotonicity
violations. The proxy failed before any test evaluation or map mutation.

Exact artifact paths and SHA-256 digests are in
`docs/evidence/render_real_descriptor_audit_20260821.json`.
