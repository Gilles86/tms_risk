"""Are the noise channels separately identified, or only their combination?

Motivating question (Gilles, 2026-09-09): the position-indexed models will not
sample, while the shared perc/mem ones all do. Is that because n1 and n2 cannot
be estimated independently -- because the data only constrain some combination
of them?

The answer this produces is more specific than that, and the distinction
matters for how the paper words it. Run on `log-power-n1n2.mapjitter.klw`:

    WITHIN participant       n1@7 x n2@7  r = -0.07   (and +0.02 to +0.19 for
                                                       the other three pairs)
    GROUP level (the _mu's)  every pair   r = +0.96 to +0.98

So per participant the two channels are fine -- they are not trading off. The
ridge is at the GROUP level: the four group means (n1@7, n1@112, n2@7, n2@112)
slide together almost perfectly, so the data pin the overall noise LEVEL
tightly and barely say which channel or which anchor carries it. That is why
the worst-mixing parameter is always a between-participant SD
(`log_n1_power_sd7_sd[Intercept]`), why a prior on the group SDs (`--tau_noise`)
rescues sampling, and why the shared family samples unaided: sigma_n1 =
sigma_perc + sigma_mem with sigma_mem >= 0 makes sigma_n1 > sigma_n2 true by
construction, which cuts exactly the direction the four means slide along.

    python -m tms_risk.behavior.scripts.anchor_identifiability \\
        <trace_dir>/model-log-power-n1n2.mapjitter.klw_trace.netcdf


Reads a trace and reports, for each pair of noise anchors, the posterior
correlation WITHIN each participant (median over participants) and at the group
level. A strongly negative within-participant correlation means the data
constrain a COMBINATION of the two, not each on its own -- the signature of a
ridge, and exactly what makes NUTS crawl.
"""
import sys, numpy as np, xarray as xr, itertools
path = sys.argv[1]
ds = xr.open_dataset(path, group='posterior')
names = [n for n in ds.attrs['tms_risk_parameters'].split(',')]
print(f'{path.split("/")[-1]}\n  parameters: {names}')
# subject-level values: intercept column of each parameter's design matrix
vals = {}
for p in names:
    if p not in ds: continue
    rdim = f'{p}_regressors'
    a = ds[p].stack(sample=('chain','draw'))
    if rdim in ds.dims:
        a = a.transpose('subject','sample',rdim).values[..., 0]   # IPS cell
    else:
        a = a.transpose('subject','sample').values
    vals[p] = a
def rowcorr(x, y):
    x = x - x.mean(-1, keepdims=True); y = y - y.mean(-1, keepdims=True)
    d = np.sqrt((x**2).sum(-1) * (y**2).sum(-1))
    return np.where(d > 0, (x*y).sum(-1)/np.where(d>0,d,1), np.nan)
ks = list(vals)
print(f'\n  {"pair":48s} {"within-subj r":>14s} {"n_sub":>6s}')
for a_, b_ in itertools.combinations(ks, 2):
    r = rowcorr(vals[a_], vals[b_])
    print(f'  {a_+" x "+b_:48s} {np.median(r):+14.3f} {len(r):6d}')
# group level
print()
for p in ks:
    mu = f'{p}_mu'
    if mu in ds:
        pass
gp = {n: ds[n].stack(sample=('chain','draw')).values.ravel()
      for n in ds.data_vars if n.endswith('_mu') and ds[n].size < 1e6}
gk = [k for k in gp if 'power_sd' in k]
if len(gk) > 1:
    print(f'  {"GROUP-level pair":48s} {"r":>14s}')
    for a_, b_ in itertools.combinations(gk, 2):
        x, y = gp[a_], gp[b_]
        if x.shape != y.shape: continue
        print(f'  {a_+" x "+b_:48s} {np.corrcoef(x,y)[0,1]:+14.3f}')
ds.close()
