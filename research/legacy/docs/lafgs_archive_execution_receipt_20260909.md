# LaFGS / STDLoc full archive execution receipt

Status: `COMPLETE`
Started: `2026-09-09T21:29:49+08:00`
Finished: `2026-09-09T21:29:56+08:00`
Manifest SHA-256: `cee95bd73a0337d322b7652d40243d6bcc75be4e8fdde2028a575e53c53cedd8`

本次执行只处理 dry-run 中的 ARCHIVE 与 DELETE_CANDIDATE。KEEP、AnyGSLoc 当前目录、原始数据、G4Splat_runs 均未被移动或删除。

| action/result | count | allocated from manifest |
|---|---:|---:|
| `ARCHIVE` | 40514 | 46.58 GiB |
| `DELETE_CANDIDATE` | 21599 | 692.27 GiB |
| `KEEP` | 27451 | 0.00 GiB |
| `already_archived` | 40514 | 46.58 GiB |
| `already_absent` | 21599 | 692.27 GiB |

Empty directories pruned: `0`.
Pool free bytes before: `5375988334592` (5006.78 GiB); after: `5375988334592` (5006.78 GiB).
Errors recorded: `0`.

Machine receipt: [`lafgs_archive_execution_receipt_20260909.tsv`](/root/STDLoc/docs/lafgs_archive_execution_receipt_20260909.tsv)
