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

## What I claimed, and why it was wrong

I read the baseline pattern — safe options decoded +0.40 log units too high,
risky options −0.36 too low — as prior attraction visible in the neural data,
and said it was a stronger independent validation of the model's shrinkage
architecture than anything in Figure 5. **That was wrong, and it is an artefact
of the decoder.**

`decode.py:24` sets `stimulus_range = np.linspace(0, 6, 1000)` and
`get_stimulus_pdf` evaluates the likelihood on that grid with no prior — which
is a FLAT prior on a bounded interval, and a flat prior on a bounded interval is
not neutral. Its posterior mean is pulled toward the grid centre, log 3.0, which
is 20 CHF. Measured:

* corr(bias, distance below the grid centre) = **+0.63**
* regression slope = **0.92**, where 1.0 is complete attraction to the centre
* the bias crosses zero at **19–20 CHF**, i.e. e³, not at any payoff the
  paradigm or the fitted priors single out

Safe payoffs (7–28) sit below log 3.0 and are pulled up; risky payoffs (7–112)
straddle it and are pulled down on average. The whole pattern is the grid. It
says nothing about prior attraction in the brain, and it must not go in the
paper.

**A real version of that test would need** a decoding grid whose centre is not
confounded with the payoff distribution — or, better, the comparison run within
payoff level, where the grid's pull is constant and any residual role- or
condition-dependence is interpretable.

## Caveats

* 44 of 51 candidate files used; 7 had a stimulus grid outside the expected
  0–6 range and were skipped rather than silently averaged in (they were, at
  first, and produced impossible biases of +11 log units).
* 23 of 35 participants have a usable cross-session decode at this mask and
  voxel count.
* One mask (NPC12r) and one voxel count (100). CLAUDE.md notes NPC12r dilutes
  relative to the individualised target mask.
* **Wrong derivative.** `notes/PROVENANCE.md` says the paper's decoding
  analyses use
  `derivatives/decoded_pdfs.volume.cv_voxel_selection.denoise.natural_space/` —
  cross-validated voxel selection, natural space. I used
  `decoded_pdfs.crosssession.volume.denoise`, which is neither, and which is
  also far less complete (44 usable files against 626 ses-2 files in the CV
  tree). That is why the count fell to 23 participants rather than the 35 the
  PRF analyses use. Any rerun should use the CV tree.
