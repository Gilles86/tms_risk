# Clean refit plan — bauer merge, model grid, priors, output

Written 2026-08-28 after the audit that found the linear/log comparison was
confounded by three different group-SD priors and two hyperprior specs. Nothing
below is fitted yet; this is the spec to agree before anything runs.

---

## 0. Bugs to fix first (all verified, all silent)

**B1 — `safe_n` selects the risky option.** `risky_choice.py`,
`get_free_parameters`:

```python
risky_n = np.where(paradigm['p1'] != 1.0, paradigm['n1'], paradigm['n2'])
safe_n  = np.where(paradigm['p2'] != 1.0, paradigm['n2'], paradigm['n1'])   # BUG
```

If `p2 != 1.0` then option 2 is the risky one, so the safe payoff is `n1`.
Both branches return the risky option: `safe_n` is **bit-identical to
`risky_n`** (verified on all 8335 trials). Consequence: `safe_prior_mu` is
centred at the risky mean log-payoff **3.409 (30.2 CHF)** instead of the safe
one **2.645 (14.1 CHF)** — a 0.76 log-unit (2.1×) offset — and `safe_prior_sd`
at 0.606 instead of 0.490. Every log-space fit carries this. Fix:
`np.where(p1 != 1.0, n2, n1)`.

**B2 — `prior_sd` centred through the wrong side of the transform.** The code
sets `mu_intercept = std(log payoff)` with `transform='softplus'`, but bauer
applies softplus *after* the Normal, so the prior median is
`softplus(0.606) = 1.042`, not 0.606. Fix: `mu_intercept =
inverse_softplus(0.606) = -0.182`.

**B3 — noise coefficients centred at zero.** `mu_intercept = 0` with a softplus
link puts the prior median at `softplus(0) = 0.693`, roughly 3× above the
fitted relative noise (0.13–0.29). Fix by centring on a target σ (below).

**B4 — one group-SD scale for every parameter.** `cauchy_sigma_intercept =
cauchy_sigma_regressors = 0.25` regardless of units or basis. Coefficients live
on the reciprocal of their basis column's magnitude: measured cTBS-slope group
SDs are 0.15–0.63 for B-spline coefficients and **4.67** for the
generalized-Weber 1/x coefficient.

**B5 — three different group-SD priors across checkouts, silently.** bauer
0.3.0 (`e05f73a`, the GPU boxes) defaults `group_sd_dist = 'halfnormal'`;
`bauer_lfx`/`bauer_af` (dd6feab) call `pm.HalfCauchy` directly and ignore
`GROUP_SD_DIST` entirely; `bauer_fp`/`bauer-powerlaw` (34e64a6) honour it.
**Every natural-space fit used HalfNormal and every log-space fit used
HalfCauchy.** This alone may explain why the linear family converged and the
log both-channels cells did not.

**B6 — softplus applied to the basis sum, not the coefficients.** See §2.

---

## 1. bauer cleanup into `main`

Five checkouts, none a superset. Merge order:

| Feature | Lives in | Action |
|---|---|---|
| `group_sd_dist` per instance | HEAD `e05f73a` | keep, this is the good design |
| `gw` / `pl` spline bases | all clones | port to main |
| `fix_safe_prior_sd` prior | `34e64a6` | port |
| `memory_composition` | `34e64a6` | port |
| per-coefficient noise link | `bauer_af` only | port, see §2 |
| `GROUP_SD_SCALE` | `bauer_fp` | **drop** — superseded by §3 |
| `RANDOM_SLOPES` | fit_model side | keep as a model kwarg |

Then delete the scratch clones and pin `libs/bauer` to the merged commit. Every
trace already stamps `tms_risk_bauer_commit`; add `tms_risk_prior_spec`.

---

## 2. Reparameterise: anchor values, not pre-transform coefficients

The softplus scale is the source of B2, B3, B4 and B6 at once. The fix is to
stop parameterising the noise function by coefficients on a hidden scale and
parameterise it by **its own values at two anchor payoffs**, 7 and 112 CHF.

Let `s_lo`, `s_hi` be the relative noise SD at 7 and 112 CHF. Both are positive
quantities a person can read, argue about and put a prior on directly. Sample
them as `log s_lo`, `log s_hi ~ Normal(...)`, so positivity is automatic and
there is no bending anywhere.

| Form | Interpolation between the anchors | Free parameters |
|---|---|---|
| Weber | `s(x) = s_lo` (`s_hi` dropped) | 1 |
| Affine in log payoff | `s(x)` **linear** in log x from `s_lo` to `s_hi` | 2 |
| Power law | `log s(x)` linear in log x | 2 |
| Generalized Weber | `s(x) = k + c/x`, solved for the two anchors | 2 |
| 5-df spline | 5 anchors, log-spaced, linear interpolation of `s` | 5 |

Notes:

* The **affine** form is now genuinely affine — a convex combination of two
  positive endpoint values, exactly linear in log x, no softplus of a sum. This
  is what `bauer_af` already does via per-coefficient softplus; the anchor
  parameterisation makes it the natural reading rather than a trick.
* **Power law and affine stop being near-duplicates by accident.** They differ
  only in whether `s` or `log s` is interpolated — a stated modelling choice,
  not a link-function side effect.
* Basis-column magnitude disappears from the prior entirely (kills B4), because
  every coefficient is on the same interpretable scale: log relative noise.
* Same treatment for the natural-space models, with `s` in CHF instead of log
  units. That makes linear-vs-log a change of *one* declared thing.

---

## 2b. Structurally different models get structurally different names

The failure documented in `CLAUDE.md` — a published trace loading into a
differently-built graph with **no error**, every free parameter present, design
matrices matching, and predictions off by 0.12 — is possible only because
different models share parameter names. `memory_noise_sd_spline2` is the second
B-spline hat coefficient in one model (posterior ~0.5) and the 1/x coefficient
in another (posterior ~4.67). Nothing in the file says which.

**Rule: the name encodes every structural choice that changes what the number
means.** Scheme:

```
<space>_<channel>_<form>_<parameter>

space     log | chf                     inference scale
channel   perc | mem | n1 | n2          noise coordinates
form      weber | affine | power |      noise function
          genweber | spl5
parameter sd7 | sd112 | sd1..sd5        the anchor it names
```

Examples: `log_perc_affine_sd7`, `chf_n1_spl5_sd3`, `log_mem_weber_sd`,
`log_safe_prior_mu`, `chf_risky_prior_sd`.

What this buys, case by case:

| Confusable pair | Today | Under the scheme |
|---|---|---|
| affine vs power law (same design matrix, different link) | identical names | `..._affine_sd7` vs `..._power_sd7` |
| B-spline vs generalized Weber coefficient 2 | identical names, 10× different scale | `..._spl5_sd2` vs `..._genweber_sd112` |
| log-space σ vs natural-space σ (15× apart) | identical names | `log_...` vs `chf_...` |
| `memory_composition` sum-then-softplus vs additive | identical names | folded into `form` |
| n1/n2 vs memory/perceptual | **already distinct** | unchanged, this one was safe |

Enforcement, so the rule cannot be quietly broken:

1. Each trace stores its full parameter-name set in
   `posterior.attrs['tms_risk_parameters']`.
2. Any script that rebuilds a model to re-evaluate a trace asserts **exact set
   equality** between the graph's free parameters and that attribute, and
   raises otherwise. This replaces the current posterior-predictive grand-mean
   tripwire in `decompose_pmc_channels.py` — a 0.02 threshold that catches
   large mismatches and misses small ones — with a check that cannot be passed
   by accident.
3. Because the names change, **no existing trace can be loaded by the new
   code at all.** That is the desired behaviour: the archive boundary in §5
   becomes a hard wall rather than a convention, and the version-sensitivity
   section of `CLAUDE.md` stops describing a live hazard.

Cost: names get longer, and every extraction script needs its parameter
patterns updated once. Worth it — this single change would have prevented the
`safe_n` bug from being invisible (a `log_safe_prior_mu` centred at the risky
mean is obvious the moment it is printed next to `log_risky_prior_mu`), the
softplus-scale confusion, and the whole linear-vs-log comparison being run on
mismatched priors.

---

## 3. Prior specification — every hyperparameter, no exceptions

Empirical anchors, computed from the 8335 modelled trials:

```
log safe  payoff   mean 2.645  sd 0.490   (14.1 CHF)
log risky payoff   mean 3.409  sd 0.606   (30.2 CHF)
fitted relative noise            0.13 – 0.29 log units
fitted prior spreads             0.36 – 0.44 log units
```

Every parameter is hierarchical and non-centred:
`θ_i = f(μ + τ · z_i)`, `z_i ~ Normal(0, 1)`.

### Noise anchors — `log s_lo`, `log s_hi` (and 5-df: `log s_1..5`)

| | value | reasoning |
|---|---|---|
| group mean μ | `Normal(log 0.25, 0.75)` | median σ = 0.25, the middle of the fitted range. 2 SD spans σ ≈ 0.056–1.1, comfortably wider than anything observed |
| group SD τ (intercept) | `HalfNormal(0.30)` | measured τ on the current winner is 0.11–0.38, and subject estimates use only 34–59% of the allowance — the weakly-identified parameter, so pool it hard |
| group SD τ (cTBS slope) | `HalfNormal(0.30)` | 2 SD = 0.6 on log σ, i.e. a ±80% noise change; the observed effects are ~10–25% |

For natural-space models the same numbers apply to `log s` in CHF, with the
group mean centred at `log(0.25 × 15.8) = log 3.95` — the same relative noise
at the mean safe payoff.

### Magnitude priors

| parameter | transform | group mean μ | group SD τ (intercept) | τ (cTBS slope) |
|---|---|---|---|---|
| `safe_prior_mu` | identity, log CHF | `Normal(2.645, 1.0)` **(B1 fix)** | `HalfNormal(0.75)` | `HalfNormal(0.15)` |
| `risky_prior_mu` | identity, log CHF | `Normal(3.409, 1.0)` | `HalfNormal(0.75)` | `HalfNormal(0.15)` |
| `log safe_prior_sd` | log | `Normal(log 0.490, 0.50)` **(B2 fix)** | `HalfNormal(0.40)` | `HalfNormal(0.15)` |
| `log risky_prior_sd` | log | `Normal(log 0.606, 0.50)` | `HalfNormal(0.40)` | `HalfNormal(0.15)` |

**Pool hard where the data are thin, not where they are not.** Measured on the
current winner, subject-level estimates use 34–59% of the group-SD allowance for
the noise anchors, 67–73% for `prior_sd`, but **85–87% for `prior_mu`** — the
magnitude priors are the best-identified parameters in the model. So the noise
anchors and `prior_sd` get tight group SDs and `prior_mu` gets a loose one
(0.75, against a measured τ of 0.60–0.72). Clamping `prior_mu` would shrink the
one thing the data pin well, and would distort the prior-placement result.
`τ = 0.15` on the cTBS slopes is 3× the largest per-subject shift observed.

Aggressive pooling on the noise anchors is a claim, so **test it**: fit the
winner at `τ_noise = 0.30` and at `0.10` and report the ELPD difference. If
heavy pooling costs nothing predictively, that is the defence.

### Family

**`HalfNormal` on every group SD**, matching bauer 0.3.0's default. Justified by
prior predictive, not by convergence: `HalfCauchy(0.25)` puts ~30% of its mass
above 0.5 and 16% above 1.0, spreads the design cannot produce.

### Non-negotiables

1. Run a **prior predictive check** before any fit and look at the implied
   noise curves, priors and choice curves. If the prior predicts choice
   probabilities that could never occur, stop.
2. Stamp `tms_risk_prior_spec = 'v1-2026-08-28'` into every trace.
3. Any fit whose label requests a prior the loaded bauer cannot honour must
   **raise**, never run. (Already implemented for `-hn`/`-ts`.)

---

## 4. The model grid

Two axes only. Everything else is fixed by §2 and §3.

**Noise form (6)** — each fitted in *both* inference spaces where meaningful:

1. Weber (1 par) — the null shape
2. Affine in log payoff (2)
3. Power law (2)
4. Generalized Weber (2)
5. 5-df spline (5)
6. 5-df perceptual / 2-df memory (asymmetric, 5+2)

**cTBS placement (6)**:

`null`, `n1`, `n2`, `n1+n2` *(independent parameterisation)*,
`perceptual`, `memory`, `perceptual+memory` *(shared parameterisation)*

Independent and shared are exact reparameterisations of the likelihood, so the
null is shared between them: **6 forms × (1 null + 3 independent + 3 shared) =
42 fits** per inference space. Do log space first (42), then the natural-space
subset that matters for the linear-vs-log claim (forms 1, 2, 5 × the same 7 =
21). **63 fits total**, against 110 already on disk of which 42 converged.

### Delete the rescue machinery, do not merely stop using it

Every one of these was a convergence workaround for priors that §3 now fixes.
Leaving them in the code means they get reached for again, and each is a way for
two fits to differ without the label saying so.

| Token / flag | What it did | Action |
|---|---|---|
| `dp` / `tp` | default vs tightened noise hyperpriors | **delete** — §3 sets them once |
| `-p1..-p5` | perceptual df separate from memory df | **delete** — folded into the form name |
| `-fx` | cTBS as a fixed effect, no random slope | **delete** |
| `-op` `-sp` `-fs` `-f1` `-fp` | five prior rescues | **delete** |
| `-hn` `-ts` `-hp` | three group-SD switches | **delete** — one prior spec, no switch |
| `--find_init mapjitter` | MAP + jitter initialisation | **delete** (see below) |
| `--constrain` | the natural-space hyperprior override | **delete** — superseded by §3 |
| `cr3` basis, `sd2/sd3/sd5` coordinates | unused branches | **delete** |
| `-i` (independent vs shared) | genuine structural axis | **keep**, it is an axis of the grid |

The label grammar collapses to `<space>-<form>-<placement>`:
`log-affine-perc`, `chf-spl5-n1n2`, `log-genweber-null`. Three fields, each one
of the two grid axes plus the inference space, and nothing that is a knob.

**mapjitter did not work.** Of the four jobs, one has finished
(`lfx2-bs3-m2-dp-b`, ~3 h) and its log ends with *"the effective sample size per
chain is smaller than 100 for some parameters"* — the standard fit of that same
label was r̂ 2.40 / ESS 5, so the initialisation changed nothing that mattered.
The other three are still running at 4 h 45 m. It is an initialisation fix for a
geometry problem, which is the wrong tool; delete it with the rest.

**Deferred to a later pass: parameter recovery.** Simulating from the winning
form at known values and refitting is the only way to settle whether `n1/n2`
and `memory/perceptual` are separately identifiable — a doubt that has run
through the whole analysis — but it is not part of this refit. Noted here so it
does not get lost.

---

## 5. Archive before anything runs

```
derivatives/cogmodels.archive-2026-08-28/
  {lfxgrid,power,ladder,overnight,noisefix,mapjitter}/     traces, untouched
  ladder_v12.tsv  ladder_v12.md                            the HalfCauchy ladder
  notes/data/{cards,delta,ploo}/                           extracted summaries
  MANIFEST.md     label -> bauer commit -> prior spec -> machine
```

Nothing is deleted. The manifest is the point: every archived trace gets its
bauer commit and, where known, its group-SD family recorded, so the old numbers
stay interpretable instead of merely stored.

---

## 6. The output PDF

One document, `notes/figures/model_report.pdf`, built by one script from TSVs
only. Sections, each a page:

1. **The paradigm and the decision axis** — payoff grid, P(risky) on the CHF
   axis vs the log-ratio axis, the stake-slope ratios.
2. **Why the observer's scale matters** — ×p as a shift; perceived value vs
   objective; the prior each observer needs against the payoff distribution.
3. **The six noise forms** — functional forms with their anchor parameters
   marked, so the reader sees what each can and cannot do.
4. **Prior predictive** — implied noise curves, priors and psychometrics under
   §3, before any data. *(New; we have never shown this.)*
5. **Convergence** — r̂/ESS/divergences for all 63, by form and placement.
   Which cells are hard, and whether the new priors fixed them.
6. **The ladder** — ELPD with paired dSE, grouped by form, converged only, with
   an explicit companion panel of what the gate removed.
7. **Fitted noise functions** — absolute and relative, per form, IPS vs vertex
   with within-draw difference strips.
8. **cTBS localisation** — forest of the effect on each channel/option at the
   two anchors, across forms.
9. **Posterior predictive** — by ratio, by stake, by order, and the
   cTBS × order × stake cells, one row per form.
10. **Provenance** — every number's source TSV, script, bauer commit, prior
    spec. Feeds `notes/PROVENANCE.md`.

Design rules: one bold row header per block; matched columns across all rows so
a model is a column everywhere; red/green reserved for IPS/vertex; blue/orange
for log/linear inference space; significance shown as within-draw difference
strips, never as overlapping marginal bands.
