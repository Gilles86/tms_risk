# Code history of bauer's flexible B-spline noise path

Forensic git archaeology on `libs/bauer` (`origin/main` = `e05f73a`, 2026-06-10).
Everything below was read with `git show <ref>:<path>`, never from the working tree.
All dates are author dates from `git log --date=iso`.

**File move**: before `b66c806` (2026-04-03) everything lived in a flat `bauer/models.py`;
after, in `bauer/models/{psychophysics,magnitude,risky_choice}.py`. `b66c806`'s parent is
`98db5a6` (2026-12-06, tip of `refactor`). Because the refactor is recorded as a
delete + add, **git shows no line diff across it** — which is exactly why the two behaviour
changes inside it are invisible to ordinary review.

Which tms_risk model uses which branch (`tms_risk/behavior/fit_model.py::_build_flexible`):

| label family | class | `memory_model` | noise branch |
|---|---|---|---|
| `flexible1*` | `FlexibleNoiseRiskRegressionModel` | `'independent'` | `n1_evidence_sd` / `n2_evidence_sd` |
| `flexible2*` | `FlexibleNoiseRiskRegressionModel` | `'shared_perceptual_noise'` | `memory_noise_sd` / `perceptual_noise_sd` |

---

## 1. Timeline

"Silent" = a behaviour change inside a commit whose message describes something else.

| # | SHA | Date | Subject | What it changed functionally | Silent? |
|---|---|---|---|---|---|
| 1 | `57ce53d` | 2023-07-25 | New flexible model based on splines | First spline noise model (`FlexibleSDComparisonModel`) | |
| 2 | `7fdd81f` | **2024-05-07** | Revised `FlexibleNoiseComparisonModel` | **Three things at once.** (a) `_get_evidence_sd_labels` swapped: `key1` went `perceptual_noise_sd` → `memory_noise_sd`, `key2` the reverse. (b) `_get_trialwise_evidence_sd` rewritten from `softplus(perc)+softplus(mem)` to `softplus(perc+mem)` — and introduced the `labels1`-twice bug (Q2). (c) new `get_sd_curve` that kept the *old* sum-of-softplus form and the *old* label ordering, and contains the `'memory_noise'` typo (Q3a/b/c all born here). | **yes** |
| 3 | `9030cec` | 2024-05-23 | `FlexibleNoiseRisRegressionkModel` | Adds `pars=`/`idata=` split to `get_sd_curve`; introduces the double `.to_dataframe()` (Q3d) and the early `return perceptual_noise + memory_noise` that skips the column/stack post-processing. Adds `FlexibleNoiseRiskModel._get_choice_predictions` with `diff_sd = sqrt(ν₁²+ν₂²)` on **raw** evidence SDs. | |
| 4 | `76673b0` | 2024-06-10 09:43 | `memory_noise_*sd*` not `memory_noise`. | Fixes the `memory_noise` → `memory_noise_sd` typo in `FlexibleNoiseRiskRegressionModel.__init__`'s **inner** `po` lookup only. Leaves the identical typo in `get_sd_curve` untouched. Creates an inconsistent state (outer membership test still `'memory_noise'`). | |
| 5 | `3608d40` | 2024-06-18 | Bugfix memory/perceptual-noise | Fixes the same typo in the **outer** membership test of `FlexibleNoiseRiskRegressionModel.__init__`. Still leaves `get_sd_curve`'s copy. | |
| 6 | `f4a57fa` | 2024-07-25 | `prior_estimate='fix_prior_sd'` | New prior mode; `'full'` path untouched | |
| 7 | `47db1c7` | 2024-11-01 | New psychometric model | Unrelated | |
| 8 | `ecc6454` | **2024-11-05 11:37** | `mutable` is always True in new versions of pymc | `bauer/core.py` only, 4 × dropping `mutable=True` from `pm.Data`. **Behaviourally a no-op for this path.** | |
| 9 | `3d994f7` | 2024-11-25 | Update of Flexible models | Adds `FlexibleNoiseRiskRegressionModel.get_sd_curve` **override** (conditionwise, per-component, correct label↔variable mapping, no shared-branch composition). Removes a `print`. Does **not** touch `_get_choice_predictions`. | partly |
| 10 | `98db5a6` | 2024-12-06 | Save some memory … | Last pre-refactor state; noise math identical to `7fdd81f` | |
| 11 | `b66c806` | **2026-04-03** | Add v0.2.0: tutorial notebooks, docs infrastructure, pyproject.toml | **Two silent behaviour changes.** (a) `_get_choice_predictions` payoff/ev branches now feed `post_n*_sd**2 / n*_evidence_sd * p*` instead of raw `n*_evidence_sd`. (b) `_get_trialwise_evidence_sd` `shared_perceptual_noise` branch fixed: `parameters[l2] for l2 in labels2`. Neither is mentioned in the commit message, HISTORY.rst 0.2.0, or CHANGELOG.md. | **yes, twice** |
| 12 | `316fc2d` | 2026-04-09 | PowerLawNoise models | Adds a parallel family; flexible path untouched | |
| 13 | `dc73bec` | **2026-05-05 08:14** | Stabilize DDM/race with CustomDist + fit_prior + fixed spline knots | **Knot anchoring fixed, deliberately and explicitly** (Q1). Adds `_spline_x_for`, `_spline_order_for` (then `_polynomial_order_for`), `_initialize_design_infos`, `_build_and_cache_design_info`; `make_dm` becomes `build_design_matrices([cached design_info], {"x": x})`. Also: `_get_trialwise_evidence_sd` reads `model['n1'].get_value()` instead of `self.paradigm['n1']`. | no — message spells it out |
| 14 | `708e6f0` | 2026-05-06 14:48 | Fix `FlexibleNoiseRiskModel` skipping `FlexibleNoiseComparisonModel.__init__` | `FlexibleNoiseRiskModel.__init__` calls `RiskModel.__init__` directly, so the design-info setup never ran → `make_dm` crashed. **Every flexible *risk* model was broken between `dc73bec` and here (~30 h).** | |
| 15 | `3dac300` | 2026-05-06 | Rename `polynomial_order` → `spline_order` | API rename only | |
| 16 | `f2cc0d7` | 2026-05-19 | Fix: `RegressionModel.build_hierarchical_nodes` silently accept `min_value`/`transform` | Accepts `min_value`/`**kwargs`; applies the `min_value` floor in `get_trialwise_variable`. Does **not** add the softplus prior branch. | |
| 17 | `188ff9a` | **2026-05-23 10:48** | Fix: `RegressionModel` softplus-prior bug (intercept silently defaulted) | **The actual `core.py` fix** (Q5). Collapses the `identity`/`logistic` if/elif to unconditional `mu[0]=mu_intercept; sigma[0]=sigma_intercept`. Also changes `RegressionModel.build_hierarchical_nodes` defaults `sigma_intercept 1.0 → 0.5`, `cauchy_sigma_intercept → 0.25`. | no |
| 18 | `34f8777` | 2026-05-26 | fix(RegressionModel): missing softplus branch silently zeroed Intercept priors | Same bug, but this commit touches **no `core.py`** — it ships the tests (`test_intercept_only_regression_matches_basic_priors`), `notes/regression_model_bug_briefing.md`, `notes/ddm_convergence_lessons.md`, and the `subject_mapping` root-fix in `magnitude.py`. | |
| 19 | `4cd98a4` | 2026-05-26 | fix(core): `get_conditionwise_parameters` works for per-subject regression fits | Fixes the function the **regression** `get_sd_curve` depends on | |
| 20 | `e05f73a` | 2026-06-10 | lint: clear flake8 backlog… | HEAD-of-record. `get_sd_curve` functionally unchanged vs `4cd98a4` (whitespace + `fit_seperate`→`fit_separate`). | |
| 21 | `adcb0bf` | **2026-07-31** | fix(risky_choice): `safe_prior_sd` was given a near-point-mass prior | Adds `sigma_intercept: 25.` to `safe_prior_sd`. **On branch `fix/safe-prior-sd-sigma`, NOT an ancestor of `origin/main`.** | no |

---

## 2. Q1 — Knot anchoring

### At `ecc6454`, confirmed

`bauer/models.py`, `FlexibleNoiseComparisonModel.make_dm`:

```python
def make_dm(self, x, variable='n1_evidence_sd'):

    min_n, max_n = self.paradigm[['n1', 'n2']].min().min(), self.paradigm[['n1', 'n2']].max().max()
    ...
    if polynomial_order > 1:
        dm = np.asarray(dmatrix(f"bs(x, degree=3, df={polynomial_order}, include_intercept=True, lower_bound={min_n}, upper_bound={max_n}) - 1",
                        {"x": x}))
```

The basis is rebuilt **from scratch on every call** from whatever `x` is handed in.
`lower_bound`/`upper_bound` are pinned to the paradigm, but patsy places interior knots at
**quantiles of `x`**. With `degree=3, df=5, include_intercept=True` there is exactly one
interior knot = the median of `x`.

Measured on the real tms_risk paradigm (8335 trials, `get_data('/data/ds-tmsrisk')`,
payoffs 7–112 CHF), with patsy 1.0.2:

| `x` handed to `make_dm` | full knot vector |
|---|---|
| `paradigm['n1']` (the FIT) | `[7, 7, 7, 7, **20**, 112, 112, 112, 112]` |
| `paradigm['n2']` (the FIT) | `[7, 7, 7, 7, **20**, 112, 112, 112, 112]` |
| `np.linspace(7, 112, 100)` (`get_sd_curve` default) | `[7, 7, 7, 7, **59.5**, 112, …]` |
| `np.arange(7, 50)` (what `figure4.ipynb` / `neurobehavioral_correlates.ipynb` pass) | `[7, 7, 7, 7, **28**, 112, …]` |
| `np.arange(7, 113)` (`nov25/*_distortion_curves.ipynb`) | `[7, 7, 7, 7, **59.5**, 112, …]` |

So yes: the fit used a knot at 20 CHF; every published plot used a knot at 28 or 59.5 CHF.
Measured basis mismatch on `np.arange(7,50)`: `max |B_paradigm − B_grid| = 0.209`
(basis values are in [0,1]).

**Side finding, contra `notes/pmc_refit_results.md`:** `median(n1) == median(n2) == 20.0`
for this paradigm, so `B_n1` and `B_n2` are **numerically identical**
(`max |B_n1 − B_n2| = 0.0` on `arange(7,50)`). HEAD's n1-vs-n2 `_spline_x_for` distinction
is a **no-op for tms_risk**. The 7–29 → 7–13 CHF shift the note reports must therefore be
the *grid-vs-paradigm* effect, not the n1-vs-n2 effect.

### When it was fixed

`dc73bec` (**2026-05-05 08:14**), deliberately. From the commit message:

> `FlexibleNoiseComparisonModel`: fix two interrelated knot bugs.
>
> 1. `patsy.bs()` places knots at quantiles of the input x, so calling `make_dm(x=...)`
>    with different x at training vs evaluation gave different bases for the same spline
>    coefs. Knots are now anchored to the paradigm columns at `__init__` via cached
>    `design_info`; later `make_dm()` calls use `build_design_matrices` to evaluate the
>    *fixed* basis at any x.

Code at `origin/main`:

```python
def _build_and_cache_design_info(self, variable):
    x = self._spline_x_for(variable)
    ...
    dm = dmatrix(formula, {"x": x})
    self._dm_design_infos[variable] = dm.design_info

def make_dm(self, x, variable='n1_evidence_sd'):
    """... Knot positions DO NOT depend on ``x`` — they were determined
    once when the model was instantiated. ..."""
    if variable not in self._dm_design_infos:
        self._build_and_cache_design_info(variable)
    dm = build_design_matrices([self._dm_design_infos[variable]], {"x": x})[0]
    return np.asarray(dm)
```

No other commit between `ecc6454` and `origin/main` touches knot placement
(`git log --all -S'build_design_matrices' -- bauer/` returns only `dc73bec` and the
unrelated 2024 `dd17b02`). No test covers it (see §7). `bauer/notes/*.md` contain zero
occurrences of "knot".

### Was the mismatch live when the 2024 fits were plotted?

**Yes, unavoidably.** `dc73bec` is 2026-05-05; the published traces are 2024-06 / 2024-11.
Any 2024-era `get_sd_curve` call — base class or the `3d994f7` regression override —
went through the rebuild-from-`x` `make_dm`. `figure4.ipynb` passes `x=np.arange(7, 50)`
→ knot 28 vs the fit's 20.

This is consistent with `CLAUDE.md`'s measurement (`ecc6454`-formula reconstruction
r = −0.3792, matching the paper's saved −0.379173; HEAD's paradigm-anchored
reconstruction r = −0.3167). **The paper's saved brain–behaviour correlation is the
mismatched-basis number.**

---

## 3. Q2 — Noise composition (`shared_perceptual_noise`, i.e. `flexible2*`)

### Confirmed at `ecc6454`

```python
elif self.memory_model == 'shared_perceptual_noise':
    dm1 = self.make_dm(x=self.paradigm['n1'], variable=key1)              # key1 = 'memory_noise_sd'
    spline_pars1 = pt.stack([parameters[l1] for l1 in labels1], axis=1)   # memory splines
    dm2 = self.make_dm(x=self.paradigm['n1'], variable=key2)              # key2 = 'perceptual_noise_sd'
    spline_pars2 = pt.stack([parameters[l1] for l1 in labels1], axis=1)   # BUG: labels1 again
    return pt.softplus(pt.sum(spline_pars1 * dm1, 1) + pt.sum(spline_pars2 * dm2, 1))
```

Sharper than "perceptual noise never entered the first option": **`dm1` and `dm2` are the
same matrix.** Both are built from `paradigm['n1']`; `dm1` uses `polynomial_order[1]` and
`dm2` uses `polynomial_order[0]`, which are equal because `__init__` expands an `int`
`polynomial_order` to `(k, k)`. So at `ecc6454`

    ν₁ = softplus( 2 · η_mem(n1) )        ν₂ = softplus( η_perc(n2) )

— an exact factor of two on the linear predictor, and the `perceptual_noise_sd_spline*`
parameters affect **only** the second-presented option. Family 2 is structurally identical
to family 1 with relabelled parameters and a hard-coded 2× on the first option.

### Was it ever correct? Yes — before 2024-05-07

At `7fdd81f^` the code did the decomposition explicitly and correctly:

```python
dm = self.make_dm(x=self.data['n1'], variable='perceptual_noise_sd')
perceptual_noise_poly_pars = pt.stack([... f'perceptual_noise_sd_poly{n}' ...], axis=1)
perceptual_noise_sd = pt.softplus(pt.sum(perceptual_noise_poly_pars * dm, 1))

dm = self.make_dm(x=self.data['n1'], variable='memory_noise_sd')
memory_noise_poly_pars = pt.stack([... f'memory_noise_sd_poly{n}' ...], axis=1)
memory_noise_sd = pt.softplus(pt.sum(memory_noise_poly_pars * dm, 1))

n1_evidence_sd = perceptual_noise_sd + memory_noise_sd     # sum of softplus
```

`7fdd81f` (**2024-05-07**) rewrote this into the compact form and introduced the bug in the
same edit. Note it *also* changed the composition semantics from `softplus(a)+softplus(b)`
to `softplus(a+b)` — the source of Q3a's mismatch, because `get_sd_curve` (rewritten in the
same commit) kept the old form.

The bug survives byte-identically through `98db5a6` (2024-12-06). I verified that
`_get_trialwise_evidence_sd` + `make_dm` are **byte-identical** (modulo the
`self.data`→`self.paradigm` rename) across `7fdd81f`, `9030cec`, `76673b0`, `3608d40`,
`f4a57fa`, `47db1c7`, `ecc6454`, `98db5a6`.

### Fixed at `b66c806` (2026-04-03), silently

```python
     dm2 = self.make_dm(x=self.paradigm['n1'], variable=key2)
-    spline_pars2 = pt.stack([parameters[l1] for l1 in labels1], axis=1)
+    spline_pars2 = pt.stack([parameters[l2] for l2 in labels2], axis=1)
```

(The `-` line is `98db5a6:bauer/models.py`, the `+` line is `b66c806:bauer/models/magnitude.py`;
git records the refactor as delete+add so this never appears as a diff.)

**Family 1 (`independent`) is unaffected at every commit.**

---

## 4. Q3 — `get_sd_curve` is still buggy at HEAD

The `shared_perceptual_noise` branch at `origin/main:bauer/models/magnitude.py` (≈ L536–566)
is **byte-identical** to `ecc6454:bauer/models.py` (≈ L1214–1240):

```python
if (variable == 'n1_evidence_sd') & (self.memory_model == 'shared_perceptual_noise'):

    if pars is None:
        pars1 = idata.posterior[labels1].to_dataframe()
        pars2 = idata.posterior[labels2].to_dataframe()
    else:
        pars1 = pars[labels1]
        pars2 = pars[labels2]

    perceptual_noise = pars1.to_dataframe()
    memory_noise = pars2.to_dataframe()

    dm1 = self.make_dm(x=x, variable='perceptual_noise_sd')
    dm2 = self.make_dm(x=x, variable='memory_noise_sd')

    perceptual_noise = softplus_np(perceptual_noise.dot(dm1.T))
    memory_noise = softplus_np(memory_noise.dot(dm2.T))

    return perceptual_noise + memory_noise

else:
    if variable in ['n1_evidence_sd', 'memory_noise']:
        labels = labels1
    else:
        labels = labels2
    ...
```

| defect | verdict | introduced | fixed |
|---|---|---|---|
| **(a)** returns `softplus(η_p)+softplus(η_m)` while `_get_trialwise_evidence_sd` computes `softplus(η_p+η_m)` | **confirmed** | discrepancy created by `7fdd81f` (2024-05-07), which changed the likelihood to softplus-of-sum but wrote `get_sd_curve` as sum-of-softplus | **never** |
| **(b)** `pars1 = idata.posterior[labels1]` is the **memory** splines (`_get_evidence_sd_labels` → `key1='memory_noise_sd'`), assigned to `perceptual_noise`, dotted with `dm1 = make_dm(variable='perceptual_noise_sd')` | **confirmed — coefficients and basis are swapped** | `7fdd81f` swapped `key1`/`key2` in `_get_evidence_sd_labels` and did not swap the `get_sd_curve` assignment | **never** |
| **(c)** `if variable in ['n1_evidence_sd', 'memory_noise']` — `'memory_noise'` is unreachable (the assert at the top requires `'memory_noise_sd'`), so `get_sd_curve(variable='memory_noise_sd')` falls through to `labels = labels2` = **perceptual** splines, while `dm = make_dm(variable='memory_noise_sd')` | **confirmed** | `7fdd81f` | **never** — even though the *identical* typo elsewhere in the file was fixed twice, in `76673b0` (2024-06-10) and `3608d40` (2024-06-18) |
| **(d)** `pars1 = idata.posterior[labels1].to_dataframe()` then `perceptual_noise = pars1.to_dataframe()` | **confirmed — the branch cannot run at all.** `pars is None` path: xarray `.to_dataframe()` → pandas DataFrame → second `.to_dataframe()` → `AttributeError`. `pars is not None` path: `pars[labels1]` is already a DataFrame → same `AttributeError`. Both paths raise. | `9030cec` (2024-05-23) | **never** |
| (e, extra) the branch `return`s early, skipping `output.columns = x` and `output.stack().to_frame(variable)`, so its return type/shape differs from the `else` branch | confirmed | `9030cec` | never |

**There has never been a version in which this branch was correct.** Before `7fdd81f` the
shared-noise reconstruction lived in a different, correctly-labelled `get_sd_curve(model, idata, …)`;
after `7fdd81f` the label ordering was inverted under it; after `9030cec` it also crashes.

**Mitigation that matters for tms_risk:** `FlexibleNoiseRiskRegressionModel` — the class
`flexible1*`/`flexible2*` actually use — has **overridden** `get_sd_curve` since `3d994f7`
(2024-11-25). The override never enters the shared branch: for `shared_perceptual_noise` it
asserts `variable in ['memory_noise_sd', 'perceptual_noise_sd', 'both']`, maps
`variable in ['n1_evidence_sd', 'memory_noise_sd'] → labels1` (with the `_sd`, correct), and
returns `softplus_np(pars.dot(dm.T))` per component. So (a)–(e) do not reach tms_risk's plotting
path **from 2024-11-25 onward** — but note the override reports *components*
`softplus(η_m)` and `softplus(η_p)` separately, which is **not** a decomposition of
ν₁ = softplus(η_m+η_p) (and, under `ecc6454`, is not even a decomposition of anything, per §3).

Before 2024-11-25 there was no override and the base shared branch would have raised
`AttributeError` — so **no 2024-06-era `flexible2` figure can have come out of the base
`get_sd_curve`**; it must have come from the regression override (post-2024-11-25) or
hand-rolled patsy. I could not determine which from git alone.

---

## 5. Q4 — The choice rule

`ecc6454:bauer/models.py`, `FlexibleNoiseRiskModel._get_choice_predictions`, `'payoff'` branch:

```python
diff_mu, diff_sd = get_diff_dist(post_n2_mu * model_inputs['p2'], model_inputs['n2_evidence_sd'],
                                 post_n1_mu * model_inputs['p1'], model_inputs['n1_evidence_sd'])
```

`b66c806:bauer/models/risky_choice.py`:

```python
n1_noise = post_n1_sd**2 / model_inputs['n1_evidence_sd'] * model_inputs['p1']
n2_noise = post_n2_sd**2 / model_inputs['n2_evidence_sd'] * model_inputs['p2']
diff_mu, diff_sd = get_diff_dist(post_n2_mu * model_inputs['p2'], n2_noise,
                                 post_n1_mu * model_inputs['p1'], n1_noise)
```

with `get_diff_dist(mu1, sd1, mu2, sd2) = (mu2-mu1, sqrt(sd1²+sd2²))` and
`get_posterior` returning `post_sd = sqrt(σ²ν²/(σ²+ν²))`, so `post_sd²/ν = ν·σ²/(σ²+ν²) = w·ν`.

**No intermediate state.** I extracted `_get_choice_predictions` at `9030cec`, `76673b0`,
`3608d40`, `f4a57fa`, `47db1c7`, `ecc6454`, `98db5a6` — all **byte-identical** to each other.
`3d994f7` "Update of Flexible models" does **not** touch it (its only changes are the
regression `get_sd_curve` override, an `assert(~x)`→`assert(not x)`, a `subject_mapping`
signature, and removing two `print`/`warn` calls).

**`b66c806` is the commit that made the change**, and it also changed the `'ev'` branch to be
identical to `'payoff'` (at `ecc6454` `'ev'` scaled the *raw* evidence SD by `p`; at HEAD both
branches are the same code). Neither the commit message, `HISTORY.rst` 0.2.0, nor `CHANGELOG.md`
mentions it.

---

## 6. Q5 — Priors on `*_prior_sd` and the `softplus` branch

### The `risky`/`safe` asymmetry

`ecc6454:bauer/models.py`, `FlexibleNoiseRiskModel.get_free_parameters`, `prior_estimate='full'`:

```python
free_parameters['risky_prior_mu'] = {'mu_intercept':risky_prior_mu, 'sigma_intercept':25., 'transform':'identity'}
free_parameters['risky_prior_sd'] = {'mu_intercept':risky_prior_sd, 'sigma_intercept':25., 'transform':'softplus'}
free_parameters['safe_prior_mu']  = {'mu_intercept':safe_prior_mu,  'sigma_intercept':25., 'transform':'identity'}

if self.prior_estimate == 'full':
    free_parameters['safe_prior_sd'] = {'mu_intercept':safe_prior_sd, 'transform':'softplus'}    # no sigma_intercept
```

**Still true at `origin/main`** (`risky_choice.py` L910–917) — byte-identical apart from PEP8
spacing. Fixed only on `adcb0bf` (2026-07-31, branch `fix/safe-prior-sd-sigma`), which is
**not** an ancestor of `origin/main`.

### The missing `softplus` branch

`ecc6454:bauer/core.py`, `RegressionModel.build_hierarchical_nodes`:

```python
def build_hierarchical_nodes(self, name, mu_intercept=0.0, sigma_intercept=1.,
                             cauchy_sigma_intercept=0.25, sigma_regressors=1.,
                             cauchy_sigma_regressors=0.25, transform='identity'):
    ...
    mu = np.zeros(self.design_matrices[name].shape[1])
    sigma = np.ones(self.design_matrices[name].shape[1]) * sigma_regressors
    ...
    if self.design_matrices[name].design_info.column_names[0] == 'Intercept':
        ...
        if transform == 'identity':
            mu[0] = mu_intercept
            sigma[0] = sigma_intercept
        elif transform == 'logistic':
            mu[0] = mu_intercept
            sigma[0] = sigma_intercept
        # Possibly use inverse of softplus
```

`git log -S'# Possibly use inverse of softplus' -- bauer/core.py` → `e2f8f75` (2022-11-22),
`abc46f2` (2024-02-15), `188ff9a` (2026-05-23). **The softplus branch never existed until
`188ff9a`.**

This applies to *every* free parameter of a `*RegressionModel`, not just regressed ones:
`RegressionModel._get_paradigm` loops `for key in free_parameters:` and
`build_design_matrix` defaults the formula to `'1'`. So `risky_prior_sd` / `safe_prior_sd`
(both `softplus`) got `Normal(0, sigma_regressors=1.0)` on the untransformed scale
(softplus(0) = 0.693 CHF), regardless of `mu_intercept ≈ 33 / 22`.
The spline coefficients are `'transform': 'identity'` with `mu_intercept=5., sigma_intercept=5.`,
so **they did get their declared prior** — the bug is confined to the softplus parameters.

**Fixed: `188ff9a`, 2026-05-23 10:48**, `bauer/core.py`:

```python
-            if transform == 'identity':
-                mu[0] = mu_intercept
-                sigma[0] = sigma_intercept
-            elif transform == 'logistic':
-                mu[0] = mu_intercept
-                sigma[0] = sigma_intercept
-            # Possibly use inverse of softplus
+            # See note in build_prior: the regression operates on the
+            # *untransformed* scale, so the Intercept prior is the same
+            # Normal(mu_intercept, sigma_intercept) for all three
+            # transforms. The previously-missing softplus branch was the
+            # source of the regression-DDM convergence pathology.
+            mu[0] = mu_intercept
+            sigma[0] = sigma_intercept
```

The same commit changed the defaults `sigma_intercept 1. → None (→0.5)` and
`cauchy_sigma_intercept 0.25 → None (→0.25)`.

`34f8777` (2026-05-26) carries a near-identical commit message but touches **no `core.py`**;
it ships the tests and `notes/regression_model_bug_briefing.md`. `f2cc0d7` (2026-05-19) is a
different fix (`min_value`/`**kwargs` acceptance, `min_value` floor in `get_trialwise_variable`).

**Net effect on the two prior SDs, by era:**

| era | `risky_prior_sd_mu` prior | `safe_prior_sd_mu` prior |
|---|---|---|
| `ecc6454` (2024 fits) | `Normal(0, 1)` (declared 33/25 ignored) | `Normal(0, 1)` (declared 22 ignored) |
| `188ff9a` … `e05f73a` (2026-07 refits) | `Normal(≈33, 25)` | **`Normal(≈22, 0.5)`** — near point mass |
| `adcb0bf` (unmerged) | `Normal(≈33, 25)` | `Normal(≈22, 25)` |

The `safe_prior_sd` omission was **harmless in 2024** (the whole softplus branch was dead) and
**only started biting on 2026-05-23**, i.e. it is live in the overnight refits.

---

## 7. Test coverage

`grep -c sd_curve` over every file in `origin/main:tests/` returns **0** in all seven test files
(`test_models.py`, `test_estimation_models.py`, `test_mcmc_smoke.py`, `test_transforms.py`,
`test_legacy.py`, `test_utils.py`, `__init__.py`). Same for `make_dm` and `design_info` in
`test_models.py` (the 5 `shared_perceptual_noise` hits there are all in `memory_as_sv`
build-only tests). `test_flexible_noise_builds` only asserts that a parameter name exists.

**None of the Q3 defects, the knot anchoring, or the noise composition is covered by any test.**
The only regression-guard in this area is
`test_intercept_only_regression_matches_basic_priors` (added in `34f8777`), which covers Q5.
`bauer/notes/*.md` and `HISTORY.rst` mention none of it.

---

## 8. Claims in `notes/pmc_refit_results.md` that this archaeology contradicts or sharpens

1. **"bauer builds family 2 as ν₁ = softplus(η_memory + η_perceptual) … So the family-1
   rotation is η_perc = η₂, η_mem = η₁ − η₂."** True at HEAD; **false for the published
   `ecc6454` traces**, where ν₁ = softplus(2·η_mem) and η_perc never enters ν₁. The
   correct `ecc6454` relation is η_mem^{fam2} = η₁/2, *not* η₁ − η₂.
   The `published flexible2` row of the "In the paper's own coordinates" table was therefore
   built with a HEAD-era formula applied to an `ecc6454`-era trace.

2. **"The family-1 ↔ family-2 reparameterisation is verified."** Only the *perceptual*
   column is verified, and trivially so: ν₂ = softplus(η_perc) in both eras, so
   η_perc^{fam2} ≡ η₂^{fam1} by construction. The memory column of the published rows
   disagrees badly (−0.92 vs −0.08 at 7 CHF), and the note's later, substantive explanation
   ("the published family-1 fit needs memory noise to fall by ~0.5 CHF … to offset the
   perceptual rise") is not needed — the `ecc6454` `labels1`-twice bug fully accounts for it.
   Under HEAD, where the composition is correct, the two memory columns agree (+0.06/+0.05/
   +0.03/−0.03 vs +0.07/+0.04/+0.03/−0.03), exactly as expected.

3. **"bauer anchors each variable's `design_info` at construction time … n1 for
   `n1_evidence_sd`/`memory_noise_sd`, **n2** for `n2_evidence_sd`/`perceptual_noise_sd`.
   The two bases therefore differ."** Anchoring is true only from `dc73bec` (2026-05-05);
   and for **this** paradigm the two bases do **not** differ — `median(n1) = median(n2) = 20`,
   so `max|B_n1 − B_n2| = 0.0`. The 7–29 → 7–13 CHF shift the note attributes to n1-vs-n2
   mixing is the *grid-vs-paradigm* knot effect (20 vs 28/59.5), which is real and large.

4. **"At `ecc6454` … `risky_prior_sd` and `safe_prior_sd` got `Normal(0, 1)`."** Confirmed
   exactly, including the mechanism (all free parameters go through `RegressionModel`
   because `build_design_matrix` defaults to `'1'`). The note's HEAD figures
   (`Normal(33, 25)` / `Normal(22, 0.5)`) are also confirmed, by `adcb0bf`'s commit message.
   Worth adding: **`adcb0bf` is not merged**, so the overnight refits carry the 0.5.

5. **"ν₁ = softplus(η_mem·B₁ + η_mem·B₂)"** — correct but understated: B₁ ≡ B₂, so it is
   exactly `softplus(2·η_mem·B)`.

6. Nothing in the note contradicts what I found about the choice rule; `b66c806` is indeed
   the sole commit, with no intermediate state.

---

## 9. Which of my traces are affected, and how

Legend: **(i)** changes the likelihood that produced the posterior → the numbers in the trace
mean something different; **(ii)** post-hoc reconstruction only → the trace is fine, the
figure/number derived from it is not; **(iii)** neither.

### `derivatives/cogmodels/model-flexible{1,2}*_trace.netcdf` — 2024, `ecc6454`-era

File `created_at` (from the netCDF `posterior` attrs) and the code state that applies:

| trace | created_at | applicable bauer state |
|---|---|---|
| `model-flexible2_trace.netcdf` | 2024-06-10 11:44 UTC | see caveat below |
| `model-flexible2_null_trace.netcdf` | 2024-11-04 15:57 UTC | `47db1c7` ≡ `ecc6454` |
| `model-flexible1_trace.netcdf` | 2024-11-04 17:11 UTC | `47db1c7` ≡ `ecc6454` |
| `model-flexible1_null_trace.netcdf` | 2024-11-04 18:15 UTC | `47db1c7` ≡ `ecc6454` |
| `model-flexible2.6_trace.netcdf` | 2024-11-05 19:05 UTC | `ecc6454` |

`ecc6454` (2024-11-05 11:37) only drops `mutable=True` from four `pm.Data` calls, so
`47db1c7` and `ecc6454` are behaviourally the same model. The 2024-11-04 timestamps are
therefore not a problem.

| defect | `flexible1*` (`independent`) | `flexible2*` (`shared_perceptual_noise`) |
|---|---|---|
| Q1 knot anchoring | **(ii)** — fit used knot 20; every plotted curve used 28 or 59.5 | **(ii)** — same |
| Q2 noise composition | **(iii)** — branch not taken | **(i) severe** — ν₁ = softplus(2·η_mem); the trace does **not** implement the Methods' ν₁ = ν_perc + ν_mem, and `perceptual_noise_sd_spline*` is really "second-option noise" |
| Q3a/b/d base `get_sd_curve` | **(iii)** — `else` branch | **(iii) in practice** — the shared branch raises `AttributeError`, so it cannot have produced any figure |
| Q3c `'memory_noise'` typo | **(iii)** — variable is invalid for `independent` | **(ii)** if the *base* class were used; **(iii)** via the `3d994f7` override, which spells it `'memory_noise_sd'` |
| Q4 choice rule | **(i)** — fitted with `diff_sd = sqrt(ν₁²+ν₂²)` on raw ν; ν is not `w·ν` | **(i)** — same |
| Q5 softplus prior branch | **(i)** — `risky/safe_prior_sd ~ Normal(0,1)` untransformed | **(i)** — same |
| Q5 `safe_prior_sd` missing `sigma_intercept` | **(iii)** — moot, the whole branch was dead | **(iii)** |

**Usability verdict.** The `flexible1*` 2024 traces are internally consistent posteriors of a
well-defined (if since-superseded) model; their *parameters* are fine to report as long as ν is
read under the old choice rule and the priors are described as `Normal(0,1)`. Every *curve* drawn
from them is on the wrong basis and should be regenerated with knots at the paradigm quantile
(20 CHF) — this is what `extract_pmc_parameters.py` already does by rebuilding with patsy.

The `flexible2*` 2024 traces are a different matter: the fitted model is not the model the
Methods describe. Anything that reads `memory_noise_sd_spline*` as "memory noise" or
`perceptual_noise_sd_spline*` as "shared perceptual noise" is mis-labelled. To salvage them,
read them as a family-1 fit: ν₁ = softplus(2·η_mem·B_20), ν₂ = softplus(η_perc·B_20).
**If the preprint's Fig 4B/4C came from `flexible2`, its axis labels are wrong.**

### `derivatives/cogmodels.overnight/model-flexible*_noisefix*.head_trace.netcdf` — 2026-07, `e05f73a`

| defect | status |
|---|---|
| Q1 knot anchoring | **(iii)** — fixed by `dc73bec`; `make_dm` now evaluates the fixed basis at any x, so fit and plot agree. (For this paradigm the n1/n2 anchoring distinction is a no-op.) |
| Q2 noise composition | **(iii)** — fixed by `b66c806`; `flexible2nf` really is a shared-perceptual model |
| Q3a/b/c/d/e base `get_sd_curve` | **still present at `e05f73a`, byte-identical to `ecc6454`.** **(ii)** in principle, **(iii)** in practice for tms_risk because `FlexibleNoiseRiskRegressionModel.get_sd_curve` overrides it. **Do not call the base-class `get_sd_curve` on a `shared_perceptual_noise` model — it raises, and if you patch the crash you get memory coefficients × perceptual basis, summed as softplus(a)+softplus(b) instead of softplus(a+b).** |
| Q4 choice rule | **(i)** — new (and, per the note's algebra, correct) rule; ν means `w·ν`-consistent noise |
| Q5 softplus prior branch | **(i)** — now honoured, so priors are `Normal(33,25)` / `Normal(22,0.5)` |
| Q5 `safe_prior_sd` missing `sigma_intercept` | **(i) — LIVE.** `adcb0bf` (2026-07-31) fixes it but is on `fix/safe-prior-sd-sigma`, not an ancestor of `origin/main`, so **every overnight refit has `safe_prior_sd_untransformed ~ Normal(22.08, 0.5)`** — effectively a point mass, 50× tighter than its `risky_prior_sd` sibling. This is a strong, undocumented, asymmetric constraint on exactly the flat direction the note identifies. **A refit against `adcb0bf` would be worth doing before reporting any prior-scale or percept-compression number from the overnight family.** |

---

## 10. What I could not determine

1. **Which bauer state produced `model-flexible2_trace.netcdf` (2024-06-10 11:44 UTC).**
   The trace's `memory_noise_sd_spline1_regressors` coord is
   `['Intercept', 'stimulation_condition[T.vertex]']`, so the regressor→spline expansion
   in `FlexibleNoiseRiskRegressionModel.__init__` worked with `_sd`-suffixed keys. **No
   committed state supports that on 2024-06-10**: before `76673b0` (09:43) both the outer
   membership test and the inner `po` lookup used `'memory_noise'`/`'perceptual_noise'`, which
   would have produced regressor keys `memory_noise_spline{i}` that match no free parameter;
   between `76673b0` and `3608d40` (2024-06-18) the two lists disagree, so `memory_noise_sd`
   fails the outer test and `memory_noise` raises `UnboundLocalError` on `po`. The most
   plausible reading is that the fit ran from a working tree with the `3608d40` fix already
   applied locally, committed eight days later. **The flexible-noise math itself is
   byte-identical across that whole window**, so this does not change any conclusion.
2. **Which code path drew the preprint's Fig 4B/4C.** The base `get_sd_curve` shared branch
   crashes; the regression override only exists from 2024-11-25; `figure4.ipynb` calls the
   override with `x=np.arange(7, 50)`. Whether the published figure predates the override
   (and so used hand-rolled patsy) is not recoverable from git.
3. **Whether `flexible2.6` / `flexible2.4` etc. were ever plotted at all**, and with which x-grid.
4. The 2024 traces carry no `tms_risk_bauer_commit` stamp (that convention starts with
   `fit_pmc_noisefix.py`), so every 2024 attribution above rests on file `created_at` plus the
   fact that `47db1c7`≡`ecc6454` on this path.
