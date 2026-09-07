"""Out-of-sample comparison of the nPRF encoding-model variants m0 / m1 / m2 (/m3).

Recomputes cross-validated R2 PER SESSION from the stored per-fold parameter maps,
which the pipeline does not store: `fit_regression_nprf_cv.py` holds out run N from
BOTH sessions at once and writes a single pooled cvR2 per fold.

Predictions are rebuilt analytically, exactly as braincoder does
(`braincoder/models/prf_1d.py::_basis_predictions_without_amplitude`,
`utils/math.py::norm`, `utils/stats.py::get_rsq`):

    pred = amplitude * exp(-0.5 * (x - mu)**2 / sd**2) + baseline,      x = log(n1)
    R2   = 1 - sum(resid**2) / sum((data - data.mean(0))**2)

A gate asserts that the in-sample R2 rebuilt this way reproduces the stored
`desc-r2` map (r > 0.999, max|diff| < 1e-4) before any cross-validated number is
written. Verified on sub-01: r = 1.000000, max|diff| <= 1e-6 for m0/m1/m2.

Writes one row per (subject, session, model, mask) plus per-subject parameter-shift
and trade-off columns.

    python -m tms_risk.modeling.scripts.extract_encoding_model_cv \
        --bids_folder /data/ds-tmsrisk --out_dir notes/data
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from tms_risk.utils.data import Subject, get_all_behavior

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31,
            34, 35, 36, 37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]
PARAMS = ['mu', 'sd', 'amplitude', 'baseline']
SESSIONS = [2, 3]
ROIS = ['NPC12r', 'NPCr', 'NPCl']


def predict(x, pars):
    """x (n_trials,), pars (n_voxels, 4) [mu, sd, amplitude, baseline]."""
    mu, sd, amp, base = [pars[None, :, i] for i in range(4)]
    return amp * np.exp(-0.5 * (x[:, None] - mu) ** 2 / sd ** 2) + base


def rsq(data, pred):
    ssq_resid = ((data - pred) ** 2).sum(0)
    ssq_data = ((data - data.mean(0)) ** 2).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = 1 - ssq_resid / ssq_data
    out[ssq_data == 0] = np.nan
    return out


class SubjectFits:
    """Loads paradigm, data, masks and parameter maps for one subject."""

    def __init__(self, subject, bids_folder, models):
        self.subject = f'{subject:02d}' if isinstance(subject, int) else subject
        self.bids = Path(bids_folder)
        self.deriv = self.bids / 'derivatives'
        self.models = models
        self.sub = Subject(self.subject, bids_folder=self.bids)

        par = self.sub.get_paradigm()
        self.par = (par.reset_index('session')[['log(n1)', 'session']]
                       .rename(columns={'log(n1)': 'x'}).astype(np.float32))
        self.runs = sorted(self.par.index.unique(level='run'))

        mask = self.sub.get_volume_mask(session=1, roi=None, epi_space=True)
        self.masker = NiftiMasker(mask_img=mask)
        d = []
        for ses in SESSIONS:
            fn = (self.deriv / 'glm_stim1.denoise.smoothed' / f'sub-{self.subject}'
                  / f'ses-{ses}' / 'func'
                  / f'sub-{self.subject}_ses-{ses}_task-task_space-T1w_desc-stims1_pe.nii.gz')
            d.append(self.masker.fit_transform(str(fn)))
        self.data = np.vstack(d).astype(np.float32)
        assert len(self.data) == len(self.par)

        # ROI indices within the whole-brain mask
        self.roi_idx = {}
        for roi in ROIS:
            try:
                m = self.sub.get_volume_mask(session=1, roi=roi, epi_space=True)
                v = self.masker.transform(m).squeeze() > 0
                self.roi_idx[roi] = v
            except Exception as e:
                logging.warning(f'sub-{self.subject}: no {roi} mask ({e})')

    def _pars(self, model, session, run=None):
        cv = run is not None
        root = (self.deriv
                / (f'encoding_model2.model-{model}.smoothed.cv' if cv
                   else f'encoding_model2.model-{model}.smoothed')
                / f'sub-{self.subject}' / f'ses-{session}')
        out = []
        for p in PARAMS:
            stem = (f'sub-{self.subject}_ses-{session}_run-{run}_desc-{p}' if cv
                    else f'sub-{self.subject}_ses-{session}_desc-{p}')
            out.append(self.masker.transform(
                str(root / f'{stem}.optim_space-T1w_pars.nii.gz')).squeeze())
        return np.stack(out, axis=-1)

    def gate(self):
        """In-sample R2 rebuilt from stored params must match the stored r2 map."""
        rows = []
        for m in self.models:
            pred = np.zeros_like(self.data)
            for ses in SESSIONS:
                idx = self.par.session.values == ses
                pred[idx] = predict(self.par.x.values[idx], self._pars(m, ses))
            mine = rsq(self.data, pred)
            fn = (self.deriv / f'encoding_model2.model-{m}.smoothed'
                  / f'sub-{self.subject}' / f'sub-{self.subject}_desc-r2.optim_space-T1w_pars.nii.gz')
            stored = self.masker.transform(str(fn)).squeeze()
            ok = np.isfinite(mine) & np.isfinite(stored)
            r = np.corrcoef(mine[ok], stored[ok])[0, 1]
            md = float(np.max(np.abs(mine[ok] - stored[ok])))
            rows.append(dict(subject=self.subject, model=f'm{m}', gate_r=r, gate_maxdiff=md,
                             passed=bool(r > 0.999 and md < 1e-4)))
        return pd.DataFrame(rows)

    def cvr2_per_session(self):
        """(n_voxels,) cvR2 for each (model, session), averaged over the 6 folds."""
        out = {}
        for m in self.models:
            acc = {ses: [] for ses in SESSIONS}
            for run in self.runs:
                for ses in SESSIONS:
                    idx = ((self.par.index.get_level_values('run') == run)
                           & (self.par.session.values == ses))
                    if idx.sum() < 3:
                        continue
                    p = self._pars(m, ses, run=run)
                    acc[ses].append(rsq(self.data[idx], predict(self.par.x.values[idx], p)))
            for ses in SESSIONS:
                out[(m, ses)] = np.nanmean(np.stack(acc[ses]), axis=0) if acc[ses] else None
        return out

    def param_shifts(self, cond):
        """Per-voxel IPS - vertex for every parameter, per model. cond maps session->arm."""
        ips_ses = [s for s in SESSIONS if cond.get(s) == 'ips']
        vtx_ses = [s for s in SESSIONS if cond.get(s) == 'vertex']
        if len(ips_ses) != 1 or len(vtx_ses) != 1:
            return None
        out = {}
        for m in self.models:
            a = self._pars(m, ips_ses[0])
            b = self._pars(m, vtx_ses[0])
            out[m] = {p: a[:, i] - b[:, i] for i, p in enumerate(PARAMS)}
            out[(m, 'ips')] = {p: a[:, i] for i, p in enumerate(PARAMS)}
            out[(m, 'vertex')] = {p: b[:, i] for i, p in enumerate(PARAMS)}
        return out


def main(bids_folder, out_dir, models, subjects):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    beh = get_all_behavior(bids_folder=bids_folder)
    cond = (beh.reset_index()[['subject', 'session', 'stimulation_condition']]
               .drop_duplicates())

    gates, cvrows, shiftrows, traderows = [], [], [], []
    for i, s in enumerate(subjects):
        t0 = time.time()
        try:
            F = SubjectFits(s, bids_folder, models)
        except Exception as e:
            logging.error(f'sub-{s}: load failed: {e}')
            continue
        g = F.gate()
        gates.append(g)
        if not g.passed.all():
            logging.error(f'sub-{s}: GATE FAILED\n{g}')
            continue

        c = cond[cond.subject == s]
        cmap = dict(zip(c.session.astype(int), c.stimulation_condition))

        cv = F.cvr2_per_session()
        masks = {'wholebrain': np.ones(F.data.shape[1], bool), **F.roi_idx}
        for m in models:
            for ses in SESSIONS:
                v = cv[(m, ses)]
                if v is None:
                    continue
                for mname, midx in masks.items():
                    x = v[midx]
                    x = x[np.isfinite(x)]
                    if not len(x):
                        continue
                    cvrows.append(dict(
                        subject=s, session=ses, arm=cmap.get(ses, 'unknown'),
                        model=f'm{m}', mask=mname, n_voxels=int(midx.sum()),
                        cvr2_mean=float(x.mean()), cvr2_median=float(np.median(x)),
                        frac_positive=float((x > 0).mean())))

        sh = F.param_shifts(cmap)
        if sh is not None:
            # model-neutral voxel set: NPC12r voxels where the POOLED model m0 has cvR2 > 0
            neutral = None
            if 0 in models and 'NPC12r' in F.roi_idx:
                base = np.nanmean(np.stack([cv[(0, ses)] for ses in SESSIONS]), axis=0)
                neutral = F.roi_idx['NPC12r'] & np.isfinite(base) & (base > 0)
            for m in models:
                for mname, midx in [('NPC12r_all', F.roi_idx.get('NPC12r')),
                                    ('NPC12r_m0cv_pos', neutral)]:
                    if midx is None or midx.sum() < 5:
                        continue
                    row = dict(subject=s, model=f'm{m}', mask=mname,
                               n_voxels=int(midx.sum()))
                    for p in PARAMS:
                        d_ = sh[m][p][midx]
                        d_ = d_[np.isfinite(d_)]
                        row[f'd_{p}_mean'] = float(d_.mean()) if len(d_) else np.nan
                        row[f'd_{p}_median'] = float(np.median(d_)) if len(d_) else np.nan
                    shiftrows.append(row)

                # (c) voxelwise amplitude-vs-dispersion trade-off, within subject
                midx = neutral if neutral is not None else F.roi_idx.get('NPC12r')
                if midx is not None and midx.sum() >= 20:
                    da, ds = sh[m]['amplitude'][midx], sh[m]['sd'][midx]
                    ok = np.isfinite(da) & np.isfinite(ds)
                    if ok.sum() >= 20:
                        from scipy import stats as st
                        traderows.append(dict(
                            subject=s, model=f'm{m}', n_voxels=int(ok.sum()),
                            r_amp_sd=float(np.corrcoef(da[ok], ds[ok])[0, 1]),
                            rho_amp_sd=float(st.spearmanr(da[ok], ds[ok])[0]),
                            r_amp_baseline=float(np.corrcoef(
                                da[ok], sh[m]['baseline'][midx][ok])[0, 1]),
                            r_amp_mu=float(np.corrcoef(
                                da[ok], sh[m]['mu'][midx][ok])[0, 1])))
        print(f'[{i+1}/{len(subjects)}] sub-{s} done in {time.time()-t0:.1f}s', flush=True)

    pd.concat(gates).to_csv(out_dir / 'encoding_cv_gate.tsv', sep='\t', index=False)
    pd.DataFrame(cvrows).to_csv(out_dir / 'encoding_cvr2_by_session.tsv', sep='\t', index=False)
    pd.DataFrame(shiftrows).to_csv(out_dir / 'encoding_param_shifts.tsv', sep='\t', index=False)
    pd.DataFrame(traderows).to_csv(out_dir / 'encoding_amp_sd_tradeoff.tsv', sep='\t', index=False)
    print('wrote 4 TSVs to', out_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--out_dir', default='notes/data')
    p.add_argument('--models', default='0,1,2')
    p.add_argument('--subjects', default='')
    a = p.parse_args()
    subs = [int(x) for x in a.subjects.split(',')] if a.subjects else SUBJECTS
    main(a.bids_folder, a.out_dir, [int(x) for x in a.models.split(',')], subs)
