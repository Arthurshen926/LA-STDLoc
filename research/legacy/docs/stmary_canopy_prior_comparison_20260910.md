# StMarysChurch matched-view canopy comparison

Output: `/mnt/pool/sqy/anygsloc_stmary_canopy_comparison_20260910_v2`.
No reconstruction was started for this diagnostic; only the existing MAtCHA
30k RGB prior was rendered on GPU 0. The localization queue uses its frozen
code snapshot and is unaffected by the added diagnostic scripts.

## Available artifacts

The inspected fast-prior, off-the-shelf and current corrected experiment
directories contain no completed StMarysChurch Long-LRM, i3DGS, Vanilla 2DGS
or Vanilla 3DGS reconstruction. Long-LRM/i3DGS corrected inputs exist, but
are not Gaussian outputs. Existing ShopFacade results cannot replace them.
MAtCHA StMarysChurch 30k is available and was used.

## Matched views

G4Splat indices 632/633/634 correspond to seq2/frame00075/76/77.png;
408 corresponds to seq1/frame00075.png. Camera poses match the Cambridge
records within 1.73e-6 absolute matrix difference (5e-6 tolerance for the
float32 versus JSON-double round trip), and 640x360 intrinsics match.
The metadata field `pose_equal_within_5e6` in the v2 generated report is a
label typo; the implemented tolerance was **5e-6**, as recorded here and
corrected in the script for future runs.

For the three seq2 views, reference/render pairs come from
`diagnostic_v310_source_sh_slow800/view_*_0800.png` in the G4Splat canopy
experiment root. Their camera entries exactly equal the audited reference
registry. This is the named saved v310 diagnostic, not a claim to represent
the latest or best G4Splat model. For seq1 the available comparison instead
uses `validation_v113_mixed_foliage24_2k` and is labelled historical.

## Observations and controls

MAtCHA has a broad smooth canopy sheet in the seq2 views, losing branches,
foliage texture and gaps despite retaining substantial facade structure.
Saved v310 retains more visible foliage detail on these samples, but has
grain, blur and boundary/background errors. The historical cross-sequence
seq1 view is poor for both saved models, in different ways.

Native official 2DGS versus gsplat MAtCHA RGB parity (native float output
against the saved quantized gsplat PNG): PSNR 53.31 / 54.13 / 53.72 dB for
632/633/634; mean absolute differences ~0.00124–0.00132 on [0,1]. Thus the
large canopy-sheet failure in these three images is not explained by a
gsplat-only display error. Seq1 parity is weaker, 33.65 dB; its backend
difference has not been explained.

Only frame seq2/77 is in the MAtCHA 512-view Gaussian optimization subset;
the other three are not. Different source-view exposure, masks, budgets,
backgrounds and renderers prevent a controlled overall method ranking.
These are qualitative rendered RGB diagnostics, not proof of the exact 3D
cause, not held-out localization scores, and not ground truth for buildings
hidden behind opaque foliage. No numeric quality ranking is asserted.
