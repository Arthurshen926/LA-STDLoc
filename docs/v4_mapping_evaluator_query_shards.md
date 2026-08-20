# V4 formal mapping-evaluator query shards

The formal mapping-only evaluator can now split its already frozen query
registry without changing the localization method.  `--shard-index I
--shard-count N` assigns fixed contiguous positions using integer range
boundaries; `--query-start/--query-stop` exposes the same registry-position
contract for schedulers.  Query selection (`--query-count`), teacher order,
global Top-1 matching, the PoseLib configuration, and the seed are resolved
before sharding.  Each selected query still executes exactly one standard
PoseLib call.

Every shard atomically publishes `mapping_cache_statistics.pt` followed by
`mapping_cache_summary.json`.  The statistics contain ordered per-query rows,
additive counters, the local range, and the complete evaluation contract.  The
contract binds all input paths and SHA-256 values, evaluation-code identity,
calibration parameters, descriptor protocol, device, seed, row limit, and the
full ordered query registry.  The merge CLI verifies the statistics SHA and
size, contract identity, report/statistics agreement, per-shard recomputed
summary, non-overlapping gap-free ranges, exact query indices, and the ordered
query-name hash.  It then concatenates rows in original registry order, sums
counters, and recomputes the summary.  Missing, partial, overlapping, stale,
or tampered inputs fail before a merged summary is published.

Example:

```bash
python -m scripts.evaluate_mapping_cache ... \
  --shard-index 0 --shard-count 2 --output shard0
python -m scripts.evaluate_mapping_cache ... \
  --shard-index 1 --shard-count 2 --output shard1
python -m scripts.merge_mapping_cache_evaluations \
  --shard-summary shard0/mapping_cache_summary.json \
  --shard-summary shard1/mapping_cache_summary.json \
  --output merged
```

## Real ShopFacade benchmark

The frozen V1.2-support ShopFacade identity map was replayed at commit
`3a033fe` for all 231 mapping queries with seed 2026.  The unsharded run on one
GPU took 111.398 s.  Two shards on two GPUs took 87.511 s plus 2.173 s to merge
(1.24x end-to-end).  Four concurrent shards on the same two GPUs took 85.017 s
plus 2.189 s to merge (1.28x end-to-end).  Both merged artifacts were exactly
equal to the unsharded artifact for all 22 summary fields, every ordered query
row, and every counter tensor.  Four shards are only marginally faster because
each fresh evaluator independently loads the frozen map, metric, teacher, and
query cache; this implementation favors isolation and fail-closed lineage over
unsafe shared mutable state.

Unit coverage additionally checks exact 2/4-shard parity and rejects partial,
tampered, and missing statistics.  Benchmark outputs live under the dedicated
non-production root
`/mnt/pool/sqy/lafgs_mapping_eval_shard_probe_20260820_3a033fe`.
