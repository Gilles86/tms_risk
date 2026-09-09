# Does the decoded magnitude show the safe-prior shift the model wants?

**No.** Quick check, and it comes back null.

## The prediction

`log-power-percpmu` (converged, r̂ 1.000, ESS 11 016) fits a cTBS shift in the
magnitude priors alongside the noise increase, and puts it on the **safe**
option: `safe_prior_mu` −14%, P(Δ<0) = 0.944, with the risky prior unmoved
(+4%, P = 0.66). If that is a real change in how safe magnitudes are
represented, the decoded magnitude of safe options should be biased further
downward under IPS than under vertex.

## The test

`derivatives/decoded_pdfs.crosssession.volume.denoise`, NPC12r, 100 voxels —
the decoder is trained on the *other* session, so it is not contaminated by the
condition it decodes. `decode.py` decodes log(n1), the FIRST-presented option,
which is the safe option on risky-second trials and the risky option on
risky-first trials. E[log n₁] per trial minus log n₁ is the bias.

| trials | bias, IPS | bias, vertex | IPS − vertex | t |
|---|---|---|---|---|
| n₁ = **safe** (risky second) | +0.404 | +0.384 | **+0.033** | +0.56 |
| n₁ = risky (risky first) | −0.358 | −0.363 | +0.036 | +0.78 |

No differential bias, no safe-specific shift, and the sign is *upward* where the
model wants downward. 21 participants after the cross-session decode's own
coverage losses, so this is underpowered — but there is not a hint of the
predicted effect, and it is the same in both option roles, which is what a null
looks like rather than a small true effect.

## What the check did show

The baseline bias is strongly **regressive**, and by exactly the amount the
model's architecture implies: safe options (7–28 CHF) are decoded **+0.40 log
units too high** and risky options (7–112 CHF) **−0.36 too low**. Both are
pulled toward the middle of the range. That is prior attraction, measured in the
neural data, independent of any choice model — worth a sentence somewhere even
though it is not the question that was asked.

## Caveats

* 44 of 51 candidate files used; 7 had a stimulus grid outside the expected
  0–6 range and were skipped rather than silently averaged in (they were, at
  first, and produced impossible biases of +11 log units).
* 23 of 35 participants have a usable cross-session decode at this mask and
  voxel count.
* One mask (NPC12r) and one voxel count (100). CLAUDE.md notes NPC12r dilutes
  relative to the individualised target mask, so the individualised mask would
  be the better test if this is worth pursuing.
