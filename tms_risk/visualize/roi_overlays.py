"""Real pycortex ROIs (paths in ``overlays.svg``) contoured from vertex masks.

Pycortex draws the ROIs of a subject's ``overlays.svg`` over every map, in the
WebGL viewer and on flatmaps, toggleable and labelled. They are normally traced
by hand in Inkscape; here they are contoured from a vertex mask in flatmap space
instead, which needs the subject's flat surfaces but no Inkscape. Adapted from
abstract_values/visualize/roi_overlays.py; the three silent traps are in the
pycortex skill:

* the paths must sit in ``rois > shapes > <g inkscape:label=NAME>``; a labelled
  path directly in ``rois`` loads without error and yields zero ROIs
* never load the overlay through ``cortex.db.get_overlay()`` with default
  arguments while writing -- it can rewrite the file from its own tree
* ``make_static`` copies the overlay cached next to the ``.ctm``; after writing,
  clear ``<filestore>/<subject>/cache`` (see ``clear_cache``)
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

SVG_NS = 'http://www.w3.org/2000/svg'
INK_NS = 'http://www.inkscape.org/namespaces/inkscape'


def _svg_shape(svgfile):
    root = ET.parse(svgfile).getroot()
    return float(root.get('width')), float(root.get('height'))


def _flat_to_svg(cx, svgshape):
    """Flat vertex coordinates in the SVG's pixel space, as pycortex maps them."""
    import cortex
    pts, _ = cortex.db.get_surf(cx, 'flat', merge=True, nudge=True)
    c = pts[:, :2].astype(float).copy()
    c -= c.min(0)
    c /= c.max(0)
    return c * np.asarray(svgshape)


def _contour_paths(mask, svg_xy, svgshape, hemi_sizes, grid=1200, smooth=2.5, min_frac=0.25):
    """Closed SVG paths around `mask`, per hemisphere, in flatmap space.

    Rasterised by nearest vertex, blurred a little (the 0.5 level of a blurred
    binary mask stays where the boundary is, but loses the staircase) and
    contoured. y is flipped: SVG grows downward, flat y upward -- a missing flip
    still parses and looks plausible, only mirrored.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    w, h = svgshape
    ny = max(8, int(round(grid * h / w)))
    gx, gy = np.meshgrid(np.linspace(0, w, grid), np.linspace(0, h, ny))
    paths, start = [], 0
    for n in hemi_sizes:
        sl = slice(start, start + n)
        start += n
        if not mask[sl].any():
            continue
        # contour each hemisphere alone, or regions touching the medial edge
        # bridge across the gap between the two flatmaps
        tree = cKDTree(svg_xy[sl])
        dist, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
        z = mask[sl][idx].astype(float)
        z[dist > 0.01 * w] = 0                     # outside this hemisphere's sheet
        z = gaussian_filter(z.reshape(gy.shape), smooth)
        fig = plt.figure()
        cs = plt.contour(gx, gy, z, levels=[0.5])
        plt.close(fig)
        segs = [s for s in cs.allsegs[0] if len(s) >= 12]      # specks, not regions
        # pycortex puts a label on every path, so fragments of a ragged mask each
        # get one; keep the pieces that carry at least `min_frac` of the largest
        area = [0.5 * abs(np.dot(s[:, 0], np.roll(s[:, 1], 1)) - np.dot(s[:, 1], np.roll(s[:, 0], 1)))
                for s in segs]
        for seg, a in zip(segs, area):
            if a < min_frac * max(area):
                continue
            seg = np.column_stack([seg[:, 0], h - seg[:, 1]])
            paths.append('M ' + ' L '.join(f'{x:.2f},{y:.2f}' for x, y in seg) + ' Z')
    return paths


def write_rois(cx, rois, prune=True, svgfile=None, circles=None):
    """Write {label: (vertex mask, colour)} as ROIs into `cx`'s overlays.svg.

    `prune` removes every other ROI; only use it on subjects whose overlay this
    pipeline alone writes. For a shared subject such as fsaverage, pass
    `svgfile`: a copy of its overlay to write into instead (for make_static's
    `overlay_file`).

    `circles` ({label: (vertex mask, colour)}) are drawn as a circle around the
    mask rather than contoured: a few-mm disc is only a handful of raster cells,
    and the blur-and-contour of `_contour_paths` shrinks it to a sliver or to
    nothing where the flatmap is compressed.
    """
    import cortex
    if svgfile is None:
        svgfile = Path(cortex.database.default_filestore) / cx / 'overlays.svg'
        if not svgfile.exists():
            cortex.db.get_overlay(cx)              # fresh subject: creates it, no prompt
    svgshape = _svg_shape(svgfile)
    svg_xy = _flat_to_svg(cx, svgshape)
    hemi_sizes = [len(p) for p, _ in cortex.db.get_surf(cx, 'flat')]

    ET.register_namespace('', SVG_NS)
    ET.register_namespace('inkscape', INK_NS)
    tree = ET.parse(svgfile)
    root = tree.getroot()

    def layer(parent, name):
        for g in parent.findall(f'{{{SVG_NS}}}g'):
            if g.get(f'{{{INK_NS}}}label') == name:
                return g
        return None

    rlayer = layer(root, 'rois')
    if rlayer is None:
        raise RuntimeError(f"no 'rois' layer in {svgfile}")
    shapes = layer(rlayer, 'shapes')
    if shapes is None:
        shapes = ET.SubElement(rlayer, f'{{{SVG_NS}}}g')
        shapes.set(f'{{{INK_NS}}}label', 'shapes')
        shapes.set(f'{{{INK_NS}}}groupmode', 'layer')
    for child in list(rlayer):
        if child.tag == f'{{{SVG_NS}}}path':
            rlayer.remove(child)
    circles = circles or {}
    for child in list(shapes):
        if prune or child.get(f'{{{INK_NS}}}label') in {**rois, **circles}:
            shapes.remove(child)

    for label, (mask, colour) in circles.items():
        pts = svg_xy[np.asarray(mask, bool)]
        c = pts.mean(0)
        r = max(3.0, float(np.max(np.linalg.norm(pts - c, axis=1))))
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ring = np.column_stack([c[0] + r * np.cos(t), svgshape[1] - (c[1] + r * np.sin(t))])
        g = ET.SubElement(shapes, f'{{{SVG_NS}}}g')
        g.set(f'{{{INK_NS}}}label', label)
        g.set('id', 'roi_' + label.replace(' ', '_'))
        el = ET.SubElement(g, f'{{{SVG_NS}}}path')
        el.set('d', 'M ' + ' L '.join(f'{x:.2f},{y:.2f}' for x, y in ring) + ' Z')
        el.set('id', f'roi_{label.replace(" ", "_")}_0')
        el.set('style', f'fill:none;stroke:{colour};stroke-width:2')
        print(f'  {cx}: ROI {label}, circle of {r:.1f} px')

    for label, (mask, colour) in rois.items():
        g = ET.SubElement(shapes, f'{{{SVG_NS}}}g')
        g.set(f'{{{INK_NS}}}label', label)
        g.set('id', 'roi_' + label.replace(' ', '_'))
        paths = _contour_paths(np.asarray(mask, bool), svg_xy, svgshape, hemi_sizes)
        for i, d in enumerate(paths):
            el = ET.SubElement(g, f'{{{SVG_NS}}}path')
            el.set('d', d)
            el.set('id', f'roi_{label.replace(" ", "_")}_{i}')
            el.set('style', f'fill:none;stroke:{colour};stroke-width:2')
        print(f'  {cx}: ROI {label}, {len(paths)} path(s)')
    tree.write(svgfile, encoding='utf-8', xml_declaration=True)
    return svgfile


def roi_masks_back(cx, labels):
    """Read ROIs back as boolean vertex masks -- the check that the SVG is right."""
    import cortex
    ov = cortex.db.get_overlay(cx, modify_svg_file=False)
    n = sum(len(p) for p, _ in cortex.db.get_surf(cx, 'flat'))
    out = {}
    for lab in labels:
        m = np.zeros(n, bool)
        m[np.asarray(ov.rois.get_mask(lab), dtype=int)] = True   # [] comes back as float
        out[lab] = m
    return out


def dice(a, b):
    return 2 * np.sum(a & b) / max(1, a.sum() + b.sum())


def clear_cache(cx):
    """Force make_static to rebuild the .ctm with the current overlays.svg."""
    import cortex
    cache = Path(cortex.database.default_filestore) / cx / 'cache'
    for f in cache.glob('*'):
        if f.is_file():
            f.unlink()
