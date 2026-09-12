"""Interactive cortical-surface viewers for the paper, as a static website.

    <out_dir>/index.html     landing page: the group viewer embedded, every
                             participant's numerosity map in a gallery, and the
                             amplitude effect participant by participant
    <out_dir>/group/         pycortex viewer on fsaverage (inflates and flattens)
    <out_dir>/sub-XX/        pycortex viewer on that participant's own surface
    <out_dir>/img/           gallery thumbnails (fsaverage flatmap crops)

Everything is plain HTML/JS/binary, so the folder can be pushed to GitHub Pages as
it is (the neural_priors viewers live at ruffgroup.github.io/neural_priors_viewers).

Inputs
------
* per-vertex m1 maps from ``tms_risk.surface.sample_model1_to_surface``
  (``<bids>/derivatives/surface_viewer``)
* the Figure-2b voxel table ``notes/data/prf_voxels_m1.tsv``, for each participant's
  amplitude in the individualised ROI -- the numbers the paper tests
* pycortex subjects ``tms.sub-XX`` (FreeSurfer imports without flatmaps, so the
  individual viewers inflate but do not flatten) and ``fsaverage``

Run in the ``pycortex2`` env, from the repo root:

    ~/mambaforge/envs/pycortex2/bin/python -m tms_risk.visualize.make_static_viewers \\
        --out_dir ~/git/tms_risk_viewers
    python -m http.server -d ~/git/tms_risk_viewers 8000

Maps are blended onto curvature in Python -- pycortex cannot threshold vertex data
live (see the pycortex skill) -- so every colour scale is drawn by the page itself,
from the same ranges used to colour the vertices.
"""
import argparse
import json
from pathlib import Path

import cortex
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy import stats

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31, 34, 35, 36,
            37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]
REPO = Path(__file__).resolve().parents[2]
BIDS = Path('/data/ds-tmsrisk')
VOXEL_TSV = REPO / 'notes' / 'data' / 'prf_voxels_m1.tsv'

PAPER = dict(
    title='Risk attitudes causally rely on parietal magnitude representations',
    authors='Gilles de Hollander<sup>*</sup>, Marius Moisa<sup>*</sup> &amp; '
            'Christian C. Ruff',
    affil='Zurich Center for Neuroeconomics, University of Zurich',
    code='https://github.com/Gilles86/tms_risk',
)

# The canonical IPS / vertex palette of every paper figure
IPS_RED, VERTEX_GREEN = '#d62728', '#2ca02c'

# A vertex is shown where the nPRF model predicts held-out runs better than the
# null model (the Figure-2b signal definition); opacity then grows with the margin
# from MARGIN_FLOOR to full at MARGIN_FULL, so the eye goes to where tuning is
# strongest rather than to how many vertices scraped past the null (25-40% of all
# vertices beat it by some margin in the smoothed data).
MARGIN_FLOOR, MARGIN_FULL = 0.005, 0.04
# Preferred numerosity is shown for every tuned vertex. The colour scale runs in
# doublings from 2 to 64 and saturates at both ends: a population whose peak lies
# below the presented range (7-112) responds most to the smallest payoffs, which is
# worth seeing, but its exact peak is not identified -- hence "<= 2" / ">= 64".
MU_SHOWN = (0.0, np.inf)
# The colour scale is histogram-equalised over the pooled preferred numerosities of
# every participant's tuned vertices (see pooled_mu_cdf), in log space, between:
MU_LIMITS = (1.0, 128.0)
MU_TICKS = [1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 56, 80, 128]
# nipy_spectral (the paper's Fig. 2a map) without its black and grey ends, which
# read as holes against the curvature
if 'nipy_spectral_mid' not in mpl.colormaps:
    mpl.colormaps.register(mpl.colors.LinearSegmentedColormap.from_list(
        'nipy_spectral_mid', mpl.colormaps['nipy_spectral'](np.linspace(0.05, 0.93, 256))))


class Scale:
    """A colour scale: colours the vertices and describes itself to the page."""

    def __init__(self, label, cmap, vmin, vmax, ticks, log=False, fmt='{:g}', ends=None):
        self.label, self.cmap, self.log = label, cmap, log
        self.vmin, self.vmax, self.ticks, self.fmt = vmin, vmax, ticks, fmt
        self.ends = ends                  # labels for saturating first/last ticks
        self.grid = self.cdf = None

    def _x(self, v):
        if not self.log:
            return v
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(v)

    def set_cdf(self, grid, cdf, candidates, min_gap=0.07):
        """Histogram-equalise: colour by the pooled CDF, so each step of the
        colour map holds the same share of vertices. Keeps the candidate ticks
        that land at least `min_gap` apart on the bar (ends always kept)."""
        self.grid, self.cdf = np.asarray(grid, float), np.asarray(cdf, float)
        pos = self._pos(candidates)
        keep, last = [candidates[0]], pos[0]
        for c, p in zip(candidates[1:-1], pos[1:-1]):
            if p - last >= min_gap and pos[-1] - p >= min_gap:
                keep.append(c)
                last = p
        self.ticks = keep + [candidates[-1]]

    def _pos(self, values):
        v = self._x(np.asarray(values, dtype=float))
        if self.cdf is not None:
            g0, g1 = self.grid[0], self.grid[-1]
            p = np.interp(np.nan_to_num(v, nan=g0, posinf=g1, neginf=g0), self.grid, self.cdf)
        else:
            lo, hi = self._x(self.vmin), self._x(self.vmax)
            p = (np.nan_to_num(v, nan=lo, posinf=hi, neginf=lo) - lo) / (hi - lo)
        return np.clip(p, 0, 1)

    def rgb(self, values):
        return mpl.colormaps[self.cmap](self._pos(values))[:, :3]

    def spec(self):
        cols = mpl.colormaps[self.cmap](np.linspace(0, 1, 48))[:, :3]
        stops = ', '.join('rgb(%d,%d,%d)' % tuple(int(round(255 * c)) for c in row)
                          for row in cols)
        ticks = [[float(p), self.fmt.format(t)] for t, p in zip(self.ticks, self._pos(self.ticks))]
        if self.ends:
            ticks[0][1], ticks[-1][1] = self.ends
        return dict(label=self.label, gradient=f'linear-gradient(to right, {stops})',
                    ticks=ticks)


MU = Scale('Preferred numerosity (colour steps = equal shares of tuned vertices; presented: 7–112)',
           'nipy_spectral_mid', MU_LIMITS[0], MU_LIMITS[1], MU_TICKS, log=True,
           ends=('≤1', '≥128'))
FIT = Scale('Out-of-sample R² gain over the null model', 'inferno', 0, 0.06,
            [0, 0.02, 0.04, 0.06])
AMP = Scale('Response amplitude (% signal change)', 'viridis', 0, 2.5, [0, 0.5, 1, 1.5, 2, 2.5])
DAMP = Scale('Amplitude change, IPS − vertex (% signal change)', 'RdBu_r', -1.0, 1.0,
             [-1, -0.5, 0, 0.5, 1])


# ------------------------------------------------------------------ surface helpers
def curvature_rgb(cx, threshold=0.0, brightness=0.5, contrast=0.25, smooth=20):
    """pycortex's standard thresholded curvature, built as ``Vertex.blend_curvature`` does."""
    curv = np.asarray(cortex.db.get_surfinfo(cx, smooth=smooth).data, dtype=float)
    g = (np.nan_to_num(curv) > threshold).astype(float) * contrast + brightness
    return np.repeat(g[:, None], 3, 1)


def adjacency(cx):
    rows, cols, off = [], [], 0
    for hemi in ('left', 'right'):
        pts, polys = cortex.db.get_surf(cx, 'wm', hemisphere=hemi)
        for a, b in ((0, 1), (1, 2), (0, 2)):
            rows.append(polys[:, a] + off)
            cols.append(polys[:, b] + off)
        off += len(pts)
    r, c = np.concatenate(rows), np.concatenate(cols)
    return sparse.coo_matrix((np.ones(2 * len(r)), (np.r_[r, c], np.r_[c, r])),
                             shape=(off, off)).tocsr()


def outline(mask, adj, width=2):
    """In-mask vertices bordering the outside, thickened by `width - 1` rings."""
    edge = mask & (adj @ (~mask).astype(float) > 0)
    for _ in range(width - 1):
        edge = edge | (mask & (adj @ edge.astype(float) > 0))
    return edge


def blend(rgb, alpha, curv):
    a = np.clip(np.nan_to_num(alpha), 0, 1)[:, None]
    return rgb * a + curv * (1 - a)


def to_vertex(rgb, cx):
    u8 = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return cortex.VertexRGB(u8[:, 0], u8[:, 1], u8[:, 2], cx)


def signal_alpha(cvr2, null):
    margin = np.nan_to_num(cvr2 - null, nan=-1)
    return np.clip((margin - MARGIN_FLOOR) / (MARGIN_FULL - MARGIN_FLOOR), 0, 1) * (margin > 0)


def mark_target(rgb, dist):
    """A two-tone ring 4.5-6.5 mm around the cTBS target: legible on any colour."""
    rgb = rgb.copy()
    rgb[(dist >= 4.5) & (dist < 5.5)] = 0.0
    rgb[(dist >= 5.5) & (dist <= 6.5)] = 1.0
    return rgb


def load(root, sid, space):
    with np.load(Path(root) / f'sub-{sid}' / f'sub-{sid}_space-{space}_desc-model1_maps.npz') as d:
        return {k: d[k] for k in d.files}


def pooled_mu_cdf(root, n=512):
    """CDF of log preferred numerosity over every participant's tuned vertices
    (own surface, cvR² above the null), within MU_LIMITS."""
    lo, hi = np.log(MU_LIMITS[0]), np.log(MU_LIMITS[1])
    vals = []
    for s in SUBJECTS:
        m = load(root, f'{s:02d}', 'fsnative')
        v = m['mu'][(m['cvr2'] - m['cvr2_null']) > 0]
        vals.append(v[(v >= lo) & (v <= hi)])
    v = np.sort(np.concatenate(vals))
    grid = np.linspace(lo, hi, n)
    return grid, np.searchsorted(v, grid) / len(v)


# ------------------------------------------------------------ participant summaries
def participant_table(tms_keys):
    """Per-participant amplitude in the individualised ROI, exactly as Figure 2b.

    Signal voxels are those beating the null model; per-participant mean over them,
    per arm. The median across participants and the paired t-test reproduce the
    paper's 1.06 -> 0.76, t(34) = 2.43 (PROVENANCE.md, Figure 2).
    """
    d = pd.read_csv(VOXEL_TSV, sep='\t')
    sig = d[d.in_mask_null]
    amp = sig.pivot_table(index='subject', columns='arm', values='amplitude', aggfunc='mean')
    fit = (d.assign(margin=d.cvr2 - d.cvr2_null)
            .groupby(['subject', 'vox']).margin.first().groupby('subject').mean())
    nvox = sig.groupby('subject').vox.nunique()
    out = pd.DataFrame({'amp_vertex': amp['vertex'], 'amp_ips': amp['ips'],
                        'fit': fit, 'n_signal_vox': nvox})
    out['first'] = [tms_keys[f'{s:02d}'][2] for s in out.index]
    t, p = stats.ttest_rel(out.amp_ips, out.amp_vertex)
    summary = dict(n=len(out), vertex=float(out.amp_vertex.median()),
                   ips=float(out.amp_ips.median()), t=float(abs(t)), p1=float(p / 2),
                   n_decrease=int((out.amp_ips < out.amp_vertex).sum()))
    print('ROI amplitude, median vertex {vertex:.3f} -> ips {ips:.3f}, t({df}) = {t:.2f}, '
          'p1 = {p1:.4f}, {n_decrease}/{n} lower after IPS cTBS'.format(df=len(out) - 1, **summary))
    return out.sort_values('fit', ascending=False), summary


# -------------------------------------------------------------------- the datasets
def subject_datasets(sid, root, burn_target=False):
    """`burn_target` paints the target ring into the maps, for a subject without
    flat surfaces (and so without overlays.svg ROIs)."""
    cx = f'tms.sub-{sid}'
    m = load(root, sid, 'fsnative')
    curv = curvature_rgb(cx)
    assert len(curv) == len(m['cvr2']), (cx, len(curv), len(m['cvr2']))
    adj = adjacency(cx)
    alpha = signal_alpha(m['cvr2'], m['cvr2_null'])
    mu = np.exp(m['mu'])
    shown = (mu >= MU_SHOWN[0]) & (mu <= MU_SHOWN[1])

    # NPC1, NPC2, the cTBS target and the analysed region are pycortex ROIs
    # (overlays.svg, see subject_rois), drawn over every map by the viewer
    dist = np.nan_to_num(m['stim_dist'], nan=1e9)

    def finish(rgb):
        return to_vertex(mark_target(rgb, dist) if burn_target else rgb, cx)

    delta = m['amp_ips'] - m['amp_vertex']
    return {
        'Preferred numerosity': finish(blend(MU.rgb(mu), alpha * shown, curv)),
        'Model fit': finish(blend(FIT.rgb(m['cvr2'] - m['cvr2_null']), alpha, curv)),
        'Amplitude after vertex cTBS': finish(blend(AMP.rgb(m['amp_vertex']), alpha, curv)),
        'Amplitude after IPS cTBS': finish(blend(AMP.rgb(m['amp_ips']), alpha, curv)),
        'Amplitude change': finish(blend(DAMP.rgb(delta), alpha, curv)),
    }


def npc_outline_fsaverage(adj):
    """Right NPC1 + NPC2 (the paper's anatomical constraint) as an fsaverage outline."""
    from nibabel import load as nload
    n_lh = len(cortex.db.get_surf('fsaverage', 'wm', hemisphere='left')[0])
    rh = np.zeros(len(cortex.db.get_surf('fsaverage', 'wm', hemisphere='right')[0]), bool)
    for lab in ('NPC1', 'NPC2'):
        g = nload(str(BIDS / 'derivatives' / 'surface_masks'
                      / f'desc-{lab}_R_space-fsaverage_hemi-rh.label.gii'))
        rh |= np.asarray(g.darrays[0].data) > 0.5
    return outline(np.r_[np.zeros(n_lh, bool), rh], adj, width=2)


def target_colours(n):
    cm = mpl.colormaps['hsv']
    return [cm(i / n)[:3] for i in range(n)]


def group_datasets(root):
    cx = 'fsaverage'
    with np.load(Path(root) / 'group_space-fsaverage_desc-model1_maps.npz') as d:
        g = {k: d[k] for k in d.files}
    curv = curvature_rgb(cx)
    adj = adjacency(cx)
    prev = g['prevalence']
    # opacity: from chance-like prevalence to a clear majority of participants
    a_prev = np.clip((prev - PREV_LO) / (PREV_HI - PREV_LO), 0, 1)
    mu = np.exp(g['mu_mean'])
    shown = (mu >= MU_SHOWN[0]) & (mu <= MU_SHOWN[1])

    # NPC1/2 etc. are drawn as pycortex ROIs from a filtered overlay (build_group)
    def finish(rgb):
        return to_vertex(rgb, cx)

    # every participant's target as a filled 5 mm disc in its own colour
    tgt = curv.copy()
    dist = g['stim_dist']
    for col, dd in zip(target_colours(len(dist)), dist):
        disc = np.nan_to_num(dd, nan=1e9) <= 5.0
        tgt[disc] = col
        tgt[outline(disc, adj, width=1)] = 0.05

    return {
        'Preferred numerosity': finish(blend(MU.rgb(mu), a_prev * shown, curv)),
        'Tuned in how many participants': finish(blend(PREV.rgb(prev), a_prev, curv)),
        'cTBS targets': finish(tgt),
        'Amplitude change': finish(blend(DAMP_GROUP.rgb(g['amp_delta_mean_signal']),
                                         a_prev, curv)),
    }


PREV_LO, PREV_HI = 0.30, 0.55
PREV = Scale('Participants with nPRF tuning at this vertex', 'magma', 0.2, 0.7,
             [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], fmt='{:.0%}')
DAMP_GROUP = Scale('Mean amplitude change, IPS − vertex (% signal change)', 'RdBu_r',
                   -0.5, 0.5, [-0.5, -0.25, 0, 0.25, 0.5])

SUBJECT_DESC = {
    'Preferred numerosity': (MU, 'The number of coins each cortical location responds to '
        'most: the peak of its numerosity tuning curve. Shown where the tuning model '
        'predicts held-out runs better than a null model (opacity grows with the margin) '
        'and the peak lies inside the presented range.'),
    'Model fit': (FIT, 'How much better the tuning model predicts held-out runs than a '
        'null model that only knows each location’s mean response '
        '(leave-one-run-out cross-validation).'),
    'Amplitude after vertex cTBS': (AMP, 'Height of the tuning curve in the session that '
        'followed control (sham) stimulation over the vertex. Preferred numerosity and '
        'tuning width are shared by the two sessions; only the amplitude may differ.'),
    'Amplitude after IPS cTBS': (AMP, 'Height of the tuning curve in the session that '
        'followed cTBS over the participant’s own numerosity-tuned parietal target.'),
    'Amplitude change': (DAMP, 'Amplitude after IPS cTBS minus amplitude after vertex '
        'cTBS. Blue: weaker numerosity-tuned responses after parietal stimulation.'),
}
GROUP_DESC = {
    'Preferred numerosity': (MU, 'Average preferred numerosity across the participants '
        'who show tuning at each fsaverage vertex. Opacity grows with the fraction of '
        'participants who do.'),
    'Tuned in how many participants': (PREV, 'Fraction of the 35 participants whose '
        'tuning model beats the null model at each vertex, after resampling every '
        'participant’s map to fsaverage.'),
    'cTBS targets': (None, 'Each disc is one participant’s cTBS target (5 mm radius), '
        'chosen on the dorsal bank of the intraparietal sulcus from their own '
        'session-1 map. Targets vary between people as much as the maps do.'),
    'Amplitude change': (DAMP_GROUP, 'Mean amplitude change after IPS relative to vertex '
        'cTBS, over the participants who are tuned at each vertex. The targets lie in '
        'different places in different people, so an average in a common space blurs a '
        'local effect; the paper tests it in each participant’s own analysed region '
        '(section 2 below).'),
}


# -------------------------------------------------------------------- page chrome
PANEL = r"""
<style>
#dataopts, #colorlegend { display: none !important; }
#tr-panel { position: fixed; left: 16px; bottom: 16px; z-index: 10000; width: 430px;
  max-width: calc(100vw - 32px); color: #eee; background: rgba(14,14,16,.9);
  border: 1px solid #3a3a3e; border-radius: 10px; box-shadow: 0 8px 28px rgba(0,0,0,.5);
  font: 13.5px/1.45 -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif; }
#tr-panel a { color: #9cc7ff; text-decoration: none; }
#tr-panel a:hover { text-decoration: underline; }
#tr-head { padding: 12px 16px 8px; border-bottom: 1px solid #2c2c30; }
#tr-kicker { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #9a9aa2; }
#tr-title { font-size: 17px; font-weight: 600; color: #fff; margin-top: 2px; }
#tr-nav { font-size: 12.5px; margin-top: 4px; color: #aaa; }
#tr-body { padding: 10px 16px 14px; }
#tr-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
#tr-chips button { font: inherit; font-size: 12.5px; color: #ddd; background: #232327;
  border: 1px solid #3f3f45; border-radius: 999px; padding: 4px 11px; cursor: pointer; }
#tr-chips button:hover { border-color: #777; color: #fff; }
#tr-chips button.on { background: #f2f2f2; color: #111; border-color: #f2f2f2; }
#tr-bar { height: 14px; border-radius: 3px; border: 1px solid #555; }
#tr-ticks { position: relative; height: 18px; font-size: 11.5px; color: #bbb;
  font-variant-numeric: tabular-nums; }
#tr-ticks span { position: absolute; transform: translateX(-50%); top: 3px; }
#tr-label { font-size: 12.5px; color: #ddd; margin: 2px 0 8px; }
#tr-desc { color: #c8c8cc; font-size: 12.5px; }
#tr-key { color: #9a9aa2; font-size: 12px; margin-top: 8px; }
#tr-extra { margin-top: 10px; padding-top: 10px; border-top: 1px solid #2c2c30; font-size: 12.5px; }
#tr-extra b { color: #fff; font-weight: 600; }
#tr-toggle { float: right; cursor: pointer; color: #aaa; border: 1px solid #444;
  border-radius: 5px; padding: 0 7px; font-size: 12px; margin-top: 2px; }
#tr-panel.min #tr-body { display: none; }
.ring { display: inline-block; width: 11px; height: 11px; border-radius: 50%;
  border: 2px solid #fff; box-shadow: inset 0 0 0 2px #000; vertical-align: -1px; }
body.embedded #tr-panel, body.embedded .dg { display: none !important; }
</style>
<div id="tr-panel">
  <div id="tr-head"><span id="tr-toggle">&ndash;</span>
    <div id="tr-kicker">__KICKER__</div>
    <div id="tr-title">__TITLE__</div>
    <div id="tr-nav">__NAV__</div>
  </div>
  <div id="tr-body">
    <div id="tr-chips"></div>
    <div id="tr-legend"><div id="tr-bar"></div><div id="tr-ticks"></div>
      <div id="tr-label"></div></div>
    <div id="tr-desc"></div>
    <div id="tr-key">__KEY__</div>
    __EXTRA__
  </div>
</div>
<script>
(function () {
  var MAPS = __MAPS__, VIEW = __VIEW__;
  if (window.self !== window.top) document.body.classList.add('embedded');
  var chips = document.getElementById('tr-chips'), current = null;
  Object.keys(MAPS).forEach(function (name) {
    var b = document.createElement('button');
    b.textContent = name;
    b.onclick = function () { if (window.viewer) viewer.setData(name); };
    b.setAttribute('data-name', name);
    chips.appendChild(b);
  });
  function show(name) {
    if (!MAPS[name]) return;
    current = name;
    var m = MAPS[name];
    Array.prototype.forEach.call(chips.children, function (b) {
      b.className = b.getAttribute('data-name') === name ? 'on' : '';
    });
    var lg = document.getElementById('tr-legend');
    if (m.scale) {
      lg.style.display = '';
      document.getElementById('tr-bar').style.background = m.scale.gradient;
      document.getElementById('tr-ticks').innerHTML = m.scale.ticks.map(function (t) {
        return '<span style="left:' + (100 * t[0]) + '%">' + t[1] + '</span>';
      }).join('');
      document.getElementById('tr-label').textContent = m.scale.label;
    } else { lg.style.display = 'none'; }
    document.getElementById('tr-desc').textContent = m.desc;
    try { window.parent.postMessage({trMap: name}, '*'); } catch (e) {}
  }
  window.trShow = show;
  document.getElementById('tr-toggle').onclick = function () {
    var p = document.getElementById('tr-panel');
    p.classList.toggle('min');
    this.innerHTML = p.classList.contains('min') ? '+' : '&ndash;';
  };
  var iv = setInterval(function () {
    if (!(window.viewer && viewer.dataviews && viewer.active && viewer.loaded)) return;
    clearInterval(iv);
    viewer.addEventListener('setData', function (e) { show(e.name); });
    show(viewer.active.name);
    viewer.loaded.done(function () {
      setTimeout(function () {
        var steps = [{state: 'mix', idx: 1.4, value: VIEW.mix}];
        if (VIEW.azimuth !== null) steps.push({state: 'camera.azimuth', idx: 1.4, value: VIEW.azimuth});
        if (VIEW.altitude !== null) steps.push({state: 'camera.altitude', idx: 1.4, value: VIEW.altitude});
        if (VIEW.radius !== null) steps.push({state: 'camera.radius', idx: 1.4, value: VIEW.radius});
        viewer.animate(steps);
        // pycortex animates on animation frames, which a background tab or an
        // off-screen iframe does not get -- it would freeze mid-way. Land on
        // the final view regardless.
        setTimeout(function () {
          steps.forEach(function (s) { try { viewer.ui.set(s.state, s.value); } catch (e) {} });
          try { viewer.schedule(); } catch (e) {}
        }, 2200);
      }, 250);
    });
  }, 100);
  // the landing page drives an embedded viewer through messages
  window.addEventListener('message', function (e) {
    var d = e.data || {};
    if (!window.viewer) return;
    if (d.setMap && viewer.dataviews[d.setMap]) viewer.setData(d.setMap);
    if (d.mix !== undefined) viewer.animate([{state: 'mix', idx: 1.0, value: d.mix}]);
    if (d.view) {
      var s = [];
      ['azimuth', 'altitude', 'radius'].forEach(function (k) {
        if (d.view[k] !== undefined) s.push({state: 'camera.' + k, idx: 1.0, value: d.view[k]});
      });
      if (d.view.mix !== undefined) s.push({state: 'mix', idx: 1.0, value: d.view.mix});
      viewer.animate(s);
    }
  });
})();
</script>
"""


def inject_panel(index_html, maps, kicker, title, nav, key, extra='', view=None):
    view = dict(dict(mix=0.5, azimuth=None, altitude=None, radius=None), **(view or {}))
    spec = {name: dict(scale=scale.spec() if scale else None, desc=desc)
            for name, (scale, desc) in maps.items()}
    block = (PANEL.replace('__MAPS__', json.dumps(spec)).replace('__VIEW__', json.dumps(view))
             .replace('__KICKER__', kicker).replace('__TITLE__', title)
             .replace('__NAV__', nav).replace('__KEY__', key).replace('__EXTRA__', extra))
    html = Path(index_html).read_text()
    html = html.split('\n<style>\n#dataopts')[0]            # idempotent re-injection
    html = html.replace('</body>', block + '\n</body>', 1) if '</body>' in html else html + block
    Path(index_html).write_text(html)


def build_static(out, ds, title, overlay_file=None, recache=False):
    out.mkdir(parents=True, exist_ok=True)
    for stale in (out / 'data').glob('*'):                  # make_static never cleans up
        stale.unlink()
    for stale in out.glob('*.ctm'):
        stale.unlink()
    # `types` lists morph targets only; a flat surface comes along by itself
    cortex.webgl.make_static(str(out), ds, types=('inflated',), recache=recache, title=title,
                             overlay_file=overlay_file, overlays_visible=('rois',),
                             labels_visible=('rois',),
                             # the default glare paints white patches that read as data
                             surface_specularity=0.1)


# ---------------------------------------------------------------------- ROIs
NPC_COLOUR, REGION_COLOUR = '#ffffff', '#ffd23f'
GROUP_ROIS = ['NPC1', 'NPC2', 'NPC3', 'NTO', 'NF1', 'NF2']      # each on both hemispheres


def has_flat(cx):
    return (Path(cortex.database.default_filestore) / cx / 'surfaces' / 'flat_rh.gii').exists()


def fsaverage_label_on_native(sid, desc, hemi='rh'):
    """An fsaverage surface label moved to the participant's surface (L+R mask),
    nearest neighbour on sphere.reg, as mri_surf2surf maps labels."""
    import nibabel as nib
    from scipy.spatial import cKDTree
    fs = BIDS / 'derivatives' / 'freesurfer'
    lab = np.asarray(nib.load(str(BIDS / 'derivatives' / 'surface_masks'
                                  / f'desc-{desc}_space-fsaverage_hemi-{hemi}.label.gii'))
                     .darrays[0].data) > 0.5
    parts = []
    for h in ('lh', 'rh'):
        subj = nib.freesurfer.read_geometry(str(fs / f'sub-{sid}' / 'surf' / f'{h}.sphere.reg'))[0]
        if h != hemi:
            parts.append(np.zeros(len(subj), bool))
            continue
        avg = nib.freesurfer.read_geometry(str(fs / 'fsaverage' / 'surf' / f'{h}.sphere.reg'))[0]
        parts.append(lab[cKDTree(avg).query(subj, workers=-1)[1]])
    return np.concatenate(parts)


def subject_rois(sid, root):
    """Write NPC1, NPC2 and the analysed region into tms.sub-XX's overlays.svg and
    check them by reading the SVG back (Dice against the source mask)."""
    from tms_risk.visualize import roi_overlays as ro
    cx = f'tms.sub-{sid}'
    m = load(root, sid, 'fsnative')
    # few labels on purpose: every ROI gets one, and more than three clutter the view
    dist = np.nan_to_num(m['stim_dist'], nan=1e9)
    rois = {'NPC1': (fsaverage_label_on_native(sid, 'NPC1_R'), NPC_COLOUR),
            'NPC2': (fsaverage_label_on_native(sid, 'NPC2_R'), NPC_COLOUR)}
    circles = {'cTBS target': (dist <= 3.0, REGION_COLOUR)}      # the centre of stimulation
    ro.write_rois(cx, rois, prune=True, circles=circles)
    back = ro.roi_masks_back(cx, list(rois) + list(circles))
    for lab, (mask, _) in {**rois, **circles}.items():
        print(f'  {cx} {lab}: Dice {ro.dice(mask, back[lab]):.3f} vs source mask')
    ro.clear_cache(cx)


def group_overlay_file(path):
    """A copy of fsaverage's overlay holding only the numerosity maps, both hemispheres.

    fsaverage is shared between projects, so its own overlays.svg is never edited;
    the copy is emptied of ROIs and the maps are contoured afresh from the
    surface labels (Barretto-García et al., 2023), which gives both hemispheres
    the same outline style -- the hand-drawn left-hemisphere paths in fsaverage's
    overlay do not render as outlines.
    """
    import shutil
    import nibabel as nib
    from tms_risk.visualize import roi_overlays as ro
    shutil.copy(Path(cortex.database.default_filestore) / 'fsaverage' / 'overlays.svg', path)
    masks_dir = BIDS / 'derivatives' / 'surface_masks'
    rois = {}
    for name in GROUP_ROIS:
        parts = []
        for H, h in (('L', 'lh'), ('R', 'rh')):
            parts.append(np.asarray(nib.load(str(masks_dir / f'desc-{name}_{H}_space-fsaverage_hemi-{h}.label.gii'))
                                    .darrays[0].data) > 0.5)
        rois[name] = (np.concatenate(parts), NPC_COLOUR)
    ro.write_rois('fsaverage', rois, prune=True, svgfile=Path(path))
    return str(path)


# ---------------------------------------------------------------- flat thumbnails
def parietal_bbox(margin=0.2):
    """fsaverage flat-coordinate box around right NPC, where every target lies."""
    pts, _ = cortex.db.get_surf('fsaverage', 'flat', merge=True, nudge=True)
    from nibabel import load as nload
    n_lh = len(cortex.db.get_surf('fsaverage', 'wm', hemisphere='left')[0])
    lab = np.asarray(nload(str(BIDS / 'derivatives' / 'surface_masks'
                                / 'desc-NPC_R_space-fsaverage_hemi-rh.label.gii'))
                     .darrays[0].data) > 0.5
    xy = pts[n_lh:][lab, :2]
    lo, hi = xy.min(0), xy.max(0)
    pad = (hi - lo) * margin
    return lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1]


def render_flat(vtx, fn, bbox=None, height=2048, width_in=4.0):
    fig = cortex.quickflat.make_figure(vtx, with_curvature=False, with_colorbar=False,
                                       with_rois=False, with_sulci=False, with_labels=False,
                                       height=height)
    ax = fig.axes[0]
    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
        w, h = bbox[1] - bbox[0], bbox[3] - bbox[2]
        fig.set_size_inches(width_in, width_in * h / w)
    fig.savefig(str(fn), dpi=200 if bbox is not None else 150, facecolor='#101012',
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def subject_thumbnail(sid, root, bbox, fn, curv, npc):
    """The participant's map resampled to fsaverage, cropped to right parietal cortex.

    Outlined is right NPC1 + NPC2 on fsaverage: the participant's own analysed
    region, resampled by nearest neighbour, has too ragged an edge to draw.
    """
    m = load(root, sid, 'fsaverage')
    mu = np.exp(m['mu'])
    shown = (mu >= MU_SHOWN[0]) & (mu <= MU_SHOWN[1])
    rgb = blend(MU.rgb(mu), signal_alpha(m['cvr2'], m['cvr2_null']) * shown, curv)
    rgb[npc] = 1.0
    rgb = mark_target(rgb, np.nan_to_num(m['stim_dist'], nan=1e9))
    render_flat(to_vertex(rgb, 'fsaverage'), fn, bbox, height=6000)


__all__ = ['subject_datasets', 'group_datasets', 'participant_table', 'inject_panel']


# ------------------------------------------------------------------------ builders
# Camera, in pycortex's conventions (checked in the rendered viewer): azimuth 180 looks
# at the occipital pole, 270 at the right hemisphere; altitude 90 is level. mix 0.5 is
# fully inflated when the subject has flat surfaces (1.0 = flat).
SUBJECT_VIEW = dict(mix=0.5, azimuth=235, altitude=50, radius=330)
GROUP_VIEW = dict(mix=0.5, azimuth=180, altitude=50, radius=215)
FLAT_VIEW = dict(mix=1.0, azimuth=180, altitude=0.1, radius=260)   # pycortex's own 2D button
SUBJECT_KEY = ('Outlines: the right numerosity maps <b>NPC1</b> and <b>NPC2</b>, and the centre '
               'of the <b>cTBS target</b>. Drag to rotate, scroll to zoom, right-drag to pan; '
               '<kbd>f</kbd> flattens, <kbd>i</kbd> inflates.')
GROUP_KEY = ('Outlines: the numerosity maps NPC1&ndash;3, NTO and NF1&ndash;2 of both '
             'hemispheres (Harvey &amp; Dumoulin, 2017; Barretto-García et al., 2023). '
             'Drag to rotate, scroll to zoom; <kbd>f</kbd> flattens, <kbd>i</kbd> inflates.')


def build_subject(sid, out_dir, root, table, order, reinject=False):
    out = Path(out_dir) / f'sub-{sid}'
    if not reinject:
        flat = has_flat(f'tms.sub-{sid}')
        if flat:
            subject_rois(sid, root)                      # also clears the .ctm cache
        build_static(out, subject_datasets(sid, root, burn_target=not flat),
                     f'sub-{sid} - numerosity maps and cTBS')
    i = order.index(sid)
    prv, nxt = order[i - 1], order[(i + 1) % len(order)]
    nav = (f'<a href="../sub-{prv}/index.html">&larr; sub-{prv}</a> &nbsp;&middot;&nbsp; '
           f'<a href="../index.html">all participants</a> &nbsp;&middot;&nbsp; '
           f'<a href="../sub-{nxt}/index.html">sub-{nxt} &rarr;</a>')
    r = table.loc[int(sid)]
    pct = 100 * (r.amp_ips - r.amp_vertex) / r.amp_vertex
    extra = (f'<div id="tr-extra">Within 2 cm of the target, inside NPC1&thinsp;+&thinsp;NPC2 '
             f'(the region the paper analyses), response amplitude was '
             f'<b style="color:#4cc35a">{r.amp_vertex:.2f}</b> after vertex cTBS and '
             f'<b style="color:#ff5a4f">{r.amp_ips:.2f}</b> after IPS cTBS ({pct:+.0f}%), '
             f'averaged over its {int(r.n_signal_vox)} tuned voxels. '
             f'{"Vertex" if r.first == "vertex" else "IPS"} cTBS came first.</div>')
    inject_panel(out / 'index.html', SUBJECT_DESC, 'Participant, own cortical surface',
                 f'sub-{sid}', nav, SUBJECT_KEY, extra, view=SUBJECT_VIEW)
    print(f'wrote {out}/index.html')


def build_group(out_dir, root, reinject=False):
    import shutil
    import tempfile
    out = Path(out_dir) / 'group'
    ds = None
    if not reinject:
        ds = group_datasets(root)
        # The .ctm cache key ignores overlay_file, so the custom overlay needs a
        # recache -- but fsaverage's cache is shared with other projects. Set it
        # aside and put it back afterwards.
        cache = Path(cortex.database.default_filestore) / 'fsaverage' / 'cache'
        keep = Path(tempfile.mkdtemp())
        for f in cache.glob('fsaverage_[[]*'):
            shutil.move(str(f), keep / f.name)
        try:
            build_static(out, ds, 'Group maps on fsaverage',
                         overlay_file=group_overlay_file(Path(root) / 'fsaverage_numerosity_rois.svg'),
                         recache=True)
        finally:
            for f in cache.glob('fsaverage_[[]*'):
                f.unlink()
            for f in keep.glob('*'):
                shutil.move(str(f), cache / f.name)
    inject_panel(out / 'index.html', GROUP_DESC, 'All 35 participants, fsaverage',
                 'Group maps', '<a href="../index.html">&larr; overview</a>', GROUP_KEY,
                 view=GROUP_VIEW)
    print(f'wrote {out}/index.html')
    return ds


def build_thumbnails(out_dir, root, subjects, group_ds=None):
    img = Path(out_dir) / 'img'
    img.mkdir(parents=True, exist_ok=True)
    bbox = parietal_bbox()
    curv, npc = curvature_rgb('fsaverage'), npc_outline_fsaverage(adjacency('fsaverage'))
    for sid in subjects:
        subject_thumbnail(sid, root, bbox, img / f'sub-{sid}.png', curv, npc)
        print(f'  thumbnail sub-{sid}', flush=True)
    if group_ds is not None:
        for name, vtx in group_ds.items():
            slug = name.lower().replace(' ', '_')
            render_flat(vtx, img / f'group_{slug}.png', height=2400)
            render_flat(vtx, img / f'group_{slug}_parietal.png', bbox)


# ---------------------------------------------------------------- landing page
INDEX = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parietal numerosity maps under cTBS - interactive viewers</title>
<meta name="description" content="Interactive cortical-surface viewers of numerosity-tuned parietal cortex in 35 participants who received cTBS on their own numerosity map.">
<style>
:root { --ink: #18181b; --muted: #5b5b66; --paper: #f7f6f3; --line: #e3e1db;
  --stage: #0c0c0f; --ips: #d62728; --vertex: #2ca02c; --link: #1f55c6; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink);
  font: 16.5px/1.6 -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 0 28px; }
.dark { background: var(--stage); color: #ececf0; }
.dark a { color: #9cc7ff; }
header.hero { padding: 64px 0 28px; }
.kicker { font-size: 12.5px; letter-spacing: .12em; text-transform: uppercase; color: #9d9da8; margin: 0 0 14px; }
h1 { font-size: clamp(30px, 4.3vw, 50px); line-height: 1.08; letter-spacing: -.015em;
  font-weight: 700; margin: 0 0 16px; max-width: 18em; color: #fff; }
.authors { color: #c9c9d1; margin: 0 0 22px; font-size: 16px; }
.authors sup { font-size: 10px; }
.lede { font-size: 19px; line-height: 1.55; color: #d9d9df; max-width: 46em; margin: 0; }
.stage { padding: 10px 0 44px; }
.frame { position: relative; border-radius: 14px; overflow: hidden; border: 1px solid #26262c;
  height: min(72vh, 700px); min-height: 440px; background: #000; }
.frame iframe { width: 100%; height: 100%; border: 0; display: block; }
.frame .legend { position: absolute; left: 18px; bottom: 16px; width: min(420px, calc(100% - 36px));
  background: rgba(10,10,12,.82); border: 1px solid #2d2d33; border-radius: 10px; padding: 12px 14px 8px;
  backdrop-filter: blur(4px); pointer-events: none; }
.legend .bar { height: 12px; border-radius: 3px; border: 1px solid #555; }
.legend .ticks { position: relative; height: 18px; font-size: 11.5px; color: #bbb; font-variant-numeric: tabular-nums; }
.legend .ticks span { position: absolute; transform: translateX(-50%); top: 3px; }
.legend .lab { font-size: 12.5px; color: #ddd; }
.legend .swatches { display: flex; flex-wrap: wrap; gap: 4px; }
.legend .swatches i { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.controls { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; justify-content: space-between; margin-top: 16px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chips button, .views button { font: inherit; font-size: 14px; color: #ddd; background: #1a1a1f;
  border: 1px solid #3a3a42; border-radius: 999px; padding: 6px 14px; cursor: pointer; }
.chips button:hover, .views button:hover { border-color: #888; color: #fff; }
.chips button.on { background: #f3f3f3; color: #111; border-color: #f3f3f3; }
.views { display: flex; gap: 8px; align-items: center; }
.views a { font-size: 14px; margin-left: 6px; }
.caption { color: #b8b8c2; font-size: 15px; max-width: 60em; margin: 14px 0 0; min-height: 3em; }
section.block { padding: 56px 0 12px; }
h2 { font-size: 27px; line-height: 1.2; letter-spacing: -.01em; margin: 0 0 12px; }
h2 .n { display: inline-block; width: 1.6em; color: #a6a39b; font-weight: 600; }
.intro { color: var(--muted); max-width: 48em; margin: 0 0 26px; }
.gallery { display: grid; gap: 14px; grid-template-columns: repeat(auto-fill, minmax(196px, 1fr)); }
.card { display: block; background: #111114; border-radius: 10px; overflow: hidden; color: #eee;
  border: 1px solid #1f1f24; transition: transform .12s ease, box-shadow .12s ease; }
.card:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(0,0,0,.18); text-decoration: none; }
.card img { width: 100%; aspect-ratio: 4 / 3; object-fit: cover; display: block; background: #101012; }
.card .meta { display: flex; justify-content: space-between; align-items: baseline; padding: 8px 11px 10px;
  font-size: 13px; }
.card .meta b { font-weight: 600; font-size: 14px; }
.card .meta span { color: #aaa; font-variant-numeric: tabular-nums; }
.card .meta .dn { color: #7fb3ff; }
.card .meta .up { color: #ff9a8a; }
.scalebar { display: flex; align-items: center; gap: 14px; margin: 0 0 18px; font-size: 13px; color: var(--muted); flex-wrap: wrap; }
.scalebar .bar { width: 260px; height: 11px; border-radius: 3px; }
.scalebar .ticks { position: relative; width: 260px; height: 16px; font-size: 11.5px; }
.scalebar .ticks span { position: absolute; transform: translateX(-50%); }
.panel { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 22px; }
.two { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 28px; align-items: center; }
@media (max-width: 860px) { .two { grid-template-columns: 1fr; } }
#paired svg { width: 100%; height: auto; display: block; }
#paired .tip { position: absolute; pointer-events: none; background: #18181b; color: #fff; font-size: 12.5px;
  padding: 4px 8px; border-radius: 5px; white-space: nowrap; display: none; }
.big { font-size: 44px; font-weight: 700; letter-spacing: -.02em; line-height: 1.1; font-variant-numeric: tabular-nums; }
.big .arrow { color: #a6a39b; font-weight: 400; margin: 0 .15em; }
.stat { color: var(--muted); margin: 6px 0 18px; }
.facts { margin: 0; padding: 0; list-style: none; }
.facts li { padding: 9px 0; border-top: 1px solid var(--line); color: #3a3a42; font-size: 15px; }
.facts b { color: var(--ink); }
.how { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
.how div { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; font-size: 15px; color: #3a3a42; }
.how b { color: var(--ink); display: block; margin-bottom: 4px; }
kbd { font: 12.5px/1 ui-monospace, SFMono-Regular, Menlo, monospace; border: 1px solid #ccc;
  border-bottom-width: 2px; border-radius: 4px; padding: 2px 5px; background: #fafafa; }
footer { margin-top: 64px; padding: 26px 0 40px; border-top: 1px solid var(--line); color: var(--muted); font-size: 14px; }
</style>
</head>
<body>
<div class="dark">
<header class="hero"><div class="wrap">
  <p class="kicker">Interactive companion &middot; cTBS &times; 3T fMRI &middot; 35 participants</p>
  <h1>__TITLE__</h1>
  <p class="authors">__AUTHORS__ &middot; __AFFIL__</p>
  <p class="lede">Numerical magnitude is encoded by tuned neural populations in the right
  intraparietal cortex, whose exact location differs from person to person. We mapped them in
  each participant, stimulated precisely there with continuous theta-burst stimulation (cTBS),
  and asked what happened to the tuned responses and to risky choice. Every map on this page is
  interactive: drag to rotate, scroll to zoom.</p>
</div></header>
<section class="stage"><div class="wrap">
  <div class="frame">
    <iframe id="gv" src="group/index.html" title="Group cortical-surface viewer" loading="eager"></iframe>
    <div class="legend" id="glegend"></div>
  </div>
  <div class="controls">
    <div class="chips" id="gchips"></div>
    <div class="views"><button data-view='__VIEW3D__'>Inflated</button><button data-view='__VIEWFLAT__'>Flat</button>
      <a href="group/index.html" target="_blank">Full screen &#8599;</a></div>
  </div>
  <p class="caption" id="gdesc"></p>
</div></section>
</div>

<section class="block"><div class="wrap">
  <h2><span class="n">1</span>Every participant carries a numerosity map</h2>
  <p class="intro">Numerosity population receptive-field (nPRF) models were fitted to every
  cortical location of every participant. Below, each participant&rsquo;s preferred-numerosity map
  around the right intraparietal sulcus, with their cTBS target (<span style="white-space:nowrap">&#9678;</span>).
  Maps are ordered by how well the tuning model predicts held-out data in the stimulated region;
  click one to open that participant&rsquo;s own cortical surface in 3D&mdash;inflatable and
  flattenable, with NPC1, NPC2 and the cTBS target outlined&mdash;and the amplitude maps of both
  sessions.</p>
  <div class="scalebar"><div><div class="bar" id="mubar"></div><div class="ticks" id="muticks"></div></div>
    <span>Preferred numerosity; each colour step holds an equal share of tuned vertices.
    Thumbnails are resampled to fsaverage and flattened (white: right NPC1&thinsp;+&thinsp;NPC2);
    the viewers show each participant&rsquo;s own anatomy.</span></div>
  <div class="gallery" id="gallery"></div>
</div></section>

<section class="block"><div class="wrap">
  <h2><span class="n">2</span>Parietal cTBS weakened numerosity-tuned responses</h2>
  <p class="intro">In the model that wins the cross-validated comparison, each location&rsquo;s tuning
  (preferred numerosity and width) is shared by the two stimulation sessions and only the height of
  the tuning curve&mdash;its amplitude&mdash;may differ. Averaged over the tuned voxels of each
  participant&rsquo;s analysed region, amplitude was lower after cTBS of the parietal target than
  after control stimulation of the vertex.</p>
  <div class="panel two">
    <div id="paired" style="position:relative"></div>
    <div>
      <div class="big"><span style="color:var(--vertex)">__AMPV__</span><span class="arrow">&rarr;</span><span style="color:var(--ips)">__AMPI__</span></div>
      <div class="stat">Median response amplitude (% signal change), vertex &rarr; IPS cTBS</div>
      <ul class="facts">
        <li><b>t(__DF__) = __T__, p = __P__</b> (paired, one-sided), n = __N__</li>
        <li><b>__NDEC__ of __N__</b> participants had lower amplitude after IPS cTBS</li>
        <li>Tuned voxels: cross-validated R&sup2; above a null model that predicts each held-out run from the others&rsquo; mean</li>
        <li>Region: within 2 cm of each participant&rsquo;s cTBS target, inside NPC1&thinsp;+&thinsp;NPC2</li>
      </ul>
    </div>
  </div>
  <p class="intro" style="margin-top:14px;font-size:14.5px">Each line is one participant; hover to
  identify, click to open their viewer. Green: after vertex (control) cTBS; red: after IPS cTBS.</p>
</div></section>

<section class="block"><div class="wrap">
  <h2><span class="n">3</span>Reading the viewers</h2>
  <div class="how">
    <div><b>Maps</b>Switch maps with the buttons in the panel. Colours are shown only where the
    tuning model predicts held-out runs better than a null model; opacity grows with the margin.</div>
    <div><b>Moving around</b>Drag to rotate, scroll to zoom, right-drag to pan. Every surface
    inflates and flattens (<kbd>i</kbd>, <kbd>f</kbd>, or the slider top right); the flatmaps
    were cut automatically with autoflatten.</div>
    <div><b>Outlines</b>Participant viewers: the right numerosity maps NPC1 and NPC2 and the
    centre of the cTBS target. Group viewer: the numerosity maps NPC1&ndash;3, NTO and
    NF1&ndash;2 of both hemispheres.</div>
    <div><b>What the model is</b>A log-normal tuning curve over numerosity per location, fitted to
    single-trial responses to the first payoff; parameters from the model in Fig.&nbsp;2d&ndash;f.</div>
  </div>
</div></section>

<footer><div class="wrap">
  Viewers built with <a href="https://github.com/gallantlab/pycortex">pycortex</a> by
  <a href="__CODE__/blob/main/tms_risk/visualize/make_static_viewers.py">make_static_viewers.py</a>
  in the analysis repository <a href="__CODE__">__CODE_SHORT__</a>.
  Contact: <a href="mailto:gilles.de.hollander@gmail.com">Gilles de Hollander</a>.
</div></footer>

<script>
var GROUP = __GROUP__, SUBJECTS = __SUBJECTS__, MU = __MUSCALE__;
(function () {
  var frame = document.getElementById('gv'), chips = document.getElementById('gchips');
  function send(msg) { try { frame.contentWindow.postMessage(msg, '*'); } catch (e) {} }
  function legend(name) {
    var m = GROUP[name], el = document.getElementById('glegend');
    if (!m) return;
    Array.prototype.forEach.call(chips.children, function (b) {
      b.className = b.getAttribute('data-name') === name ? 'on' : '';
    });
    if (m.scale) {
      el.innerHTML = '<div class="bar" style="background:' + m.scale.gradient + '"></div>' +
        '<div class="ticks">' + m.scale.ticks.map(function (t) {
          return '<span style="left:' + (100 * t[0]) + '%">' + t[1] + '</span>'; }).join('') +
        '</div><div class="lab">' + m.scale.label + '</div>';
    } else {
      el.innerHTML = '<div class="lab">' + name + ': one colour per participant</div>';
    }
    document.getElementById('gdesc').textContent = m.desc;
  }
  Object.keys(GROUP).forEach(function (name) {
    var b = document.createElement('button');
    b.textContent = name; b.setAttribute('data-name', name);
    b.onclick = function () { send({setMap: name}); legend(name); };
    chips.appendChild(b);
  });
  document.querySelectorAll('.views button').forEach(function (b) {
    b.onclick = function () { send({view: JSON.parse(b.getAttribute('data-view'))}); };
  });
  window.addEventListener('message', function (e) { if (e.data && e.data.trMap) legend(e.data.trMap); });
  legend(Object.keys(GROUP)[0]);

  // scale bar for the gallery
  document.getElementById('mubar').style.background = MU.gradient;
  document.getElementById('muticks').innerHTML = MU.ticks.map(function (t) {
    return '<span style="left:' + (100 * t[0]) + '%">' + t[1] + '</span>'; }).join('');

  var g = document.getElementById('gallery');
  SUBJECTS.forEach(function (s) {
    var a = document.createElement('a');
    a.className = 'card'; a.href = 'sub-' + s.id + '/index.html';
    var pct = Math.round(100 * (s.ips - s.vertex) / s.vertex);
    a.innerHTML = '<img loading="lazy" src="img/sub-' + s.id + '.png" alt="Preferred-numerosity map of sub-' + s.id + '">' +
      '<div class="meta"><b>sub-' + s.id + '</b><span>amplitude <span class="' + (pct < 0 ? 'dn' : 'up') + '">' +
      (pct > 0 ? '+' : '') + pct + '%</span></span></div>';
    g.appendChild(a);
  });

  // paired amplitudes, one line per participant
  var W = 560, H = 430, ML = 56, MR = 24, MT = 18, MB = 44;
  var xs = [ML + 110, W - MR - 110];
  var ymax = Math.ceil(Math.max.apply(null, SUBJECTS.map(function (s) { return Math.max(s.vertex, s.ips); })) * 2) / 2;
  function y(v) { return MT + (1 - v / ymax) * (H - MT - MB); }
  var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Amplitude per participant, vertex versus IPS cTBS">';
  for (var t = 0; t <= ymax + 1e-9; t += 0.5) {
    svg += '<line x1="' + ML + '" x2="' + (W - MR) + '" y1="' + y(t) + '" y2="' + y(t) + '" stroke="#eceae4"/>' +
      '<text x="' + (ML - 10) + '" y="' + (y(t) + 4) + '" text-anchor="end" font-size="12" fill="#77777f">' + t.toFixed(1) + '</text>';
  }
  svg += '<text transform="translate(16,' + ((H - MB + MT) / 2) + ') rotate(-90)" text-anchor="middle" font-size="12.5" fill="#55555e">Amplitude (% signal change)</text>';
  svg += '<text x="' + xs[0] + '" y="' + (H - 16) + '" text-anchor="middle" font-size="14" font-weight="600" fill="' + '__VERTEX__' + '">Vertex cTBS</text>' +
    '<text x="' + xs[1] + '" y="' + (H - 16) + '" text-anchor="middle" font-size="14" font-weight="600" fill="' + '__IPS__' + '">IPS cTBS</text>';
  SUBJECTS.forEach(function (s, i) {
    var dn = s.ips < s.vertex;
    svg += '<g class="pp" data-i="' + i + '" style="cursor:pointer">' +
      '<line x1="' + xs[0] + '" x2="' + xs[1] + '" y1="' + y(s.vertex) + '" y2="' + y(s.ips) + '" stroke="' + (dn ? '#8c8c96' : '#c9c7c0') + '" stroke-width="1.4"/>' +
      '<line x1="' + xs[0] + '" x2="' + xs[1] + '" y1="' + y(s.vertex) + '" y2="' + y(s.ips) + '" stroke="transparent" stroke-width="9"/>' +
      '<circle cx="' + xs[0] + '" cy="' + y(s.vertex) + '" r="4.2" fill="__VERTEX__" fill-opacity=".85"/>' +
      '<circle cx="' + xs[1] + '" cy="' + y(s.ips) + '" r="4.2" fill="__IPS__" fill-opacity=".85"/></g>';
  });
  var med = function (k) { var v = SUBJECTS.map(function (s) { return s[k]; }).sort(function (a, b) { return a - b; });
    var n = v.length; return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2; };
  svg += '<line x1="' + (xs[0] - 34) + '" x2="' + (xs[0] - 12) + '" y1="' + y(med('vertex')) + '" y2="' + y(med('vertex')) + '" stroke="#18181b" stroke-width="3"/>' +
    '<line x1="' + (xs[1] + 12) + '" x2="' + (xs[1] + 34) + '" y1="' + y(med('ips')) + '" y2="' + y(med('ips')) + '" stroke="#18181b" stroke-width="3"/>' +
    '<text x="' + (xs[0] - 40) + '" y="' + (y(med('vertex')) + 4) + '" text-anchor="end" font-size="12" fill="#18181b">median</text>' +
    '<text x="' + (xs[1] + 40) + '" y="' + (y(med('ips')) + 4) + '" font-size="12" fill="#18181b">median</text>';
  svg += '</svg><div class="tip"></div>';
  var box = document.getElementById('paired');
  box.innerHTML = svg;
  var tip = box.querySelector('.tip');
  box.querySelectorAll('g.pp').forEach(function (el) {
    var s = SUBJECTS[+el.getAttribute('data-i')];
    el.addEventListener('mouseenter', function () {
      el.querySelector('line').setAttribute('stroke', '#18181b');
      el.querySelector('line').setAttribute('stroke-width', '2.6');
      tip.style.display = 'block';
      tip.textContent = 'sub-' + s.id + ':  ' + s.vertex.toFixed(2) + ' → ' + s.ips.toFixed(2);
    });
    el.addEventListener('mousemove', function (e) {
      var r = box.getBoundingClientRect();
      tip.style.left = (e.clientX - r.left + 12) + 'px'; tip.style.top = (e.clientY - r.top - 30) + 'px';
    });
    el.addEventListener('mouseleave', function () {
      var dn = s.ips < s.vertex;
      el.querySelector('line').setAttribute('stroke', dn ? '#8c8c96' : '#c9c7c0');
      el.querySelector('line').setAttribute('stroke-width', '1.4');
      tip.style.display = 'none';
    });
    el.addEventListener('click', function () { window.location.href = 'sub-' + s.id + '/index.html'; });
  });
})();
</script>
</body>
</html>
"""


def write_index(out_dir, table, summary):
    subjects = [dict(id=f'{s:02d}', vertex=round(float(r.amp_vertex), 4),
                     ips=round(float(r.amp_ips), 4), fit=round(float(r.fit), 5))
                for s, r in table.iterrows()]
    group = {name: dict(scale=scale.spec() if scale else None, desc=desc)
             for name, (scale, desc) in GROUP_DESC.items()}
    html = INDEX
    for k, v in {'__TITLE__': PAPER['title'], '__AUTHORS__': PAPER['authors'],
                 '__AFFIL__': PAPER['affil'], '__CODE__': PAPER['code'],
                 '__CODE_SHORT__': PAPER['code'].replace('https://github.com/', ''),
                 '__AMPV__': f'{summary["vertex"]:.2f}', '__AMPI__': f'{summary["ips"]:.2f}',
                 '__DF__': str(summary['n'] - 1), '__T__': f'{summary["t"]:.2f}',
                 '__P__': f'{summary["p1"]:.3f}', '__N__': str(summary['n']),
                 '__NDEC__': str(summary['n_decrease']),
                 '__GROUP__': json.dumps(group), '__SUBJECTS__': json.dumps(subjects),
                 '__MUSCALE__': json.dumps(MU.spec()),
                 '__VIEW3D__': json.dumps(GROUP_VIEW), '__VIEWFLAT__': json.dumps(FLAT_VIEW),
                 '__VERTEX__': VERTEX_GREEN, '__IPS__': IPS_RED}.items():
        html = html.replace(k, v)
    fn = Path(out_dir) / 'index.html'
    fn.write_text(html)
    print(f'wrote {fn}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--root', default=str(BIDS / 'derivatives' / 'surface_viewer'))
    p.add_argument('--subjects', nargs='*', default=None)
    p.add_argument('--skip', nargs='*', default=[],
                   choices=['subjects', 'group', 'thumbs', 'index'])
    p.add_argument('--reinject', action='store_true',
                   help='refresh the injected panels only; keep the pycortex bundles')
    a = p.parse_args()

    tms_keys = yaml.safe_load((REPO / 'tms_risk' / 'data' / 'tms_keys.yml').read_text())
    table, summary = participant_table(tms_keys)
    MU.set_cdf(*pooled_mu_cdf(a.root), MU_TICKS)
    order = [f'{s:02d}' for s in table.index]          # best-fitting first
    subjects = [f'{int(s):02d}' for s in a.subjects] if a.subjects else order

    group_ds = None
    if 'group' not in a.skip:
        group_ds = build_group(a.out_dir, a.root, a.reinject)
    if 'subjects' not in a.skip:
        for sid in subjects:
            build_subject(sid, a.out_dir, a.root, table, order, a.reinject)
    if 'thumbs' not in a.skip and not a.reinject:
        build_thumbnails(a.out_dir, a.root, subjects, group_ds)
    if 'index' not in a.skip:
        write_index(a.out_dir, table, summary)


if __name__ == '__main__':
    main()
