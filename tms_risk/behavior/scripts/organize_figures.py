"""Sort notes/figures/ into one folder per plot type.

The model label is already in every filename (`fig5.flexible2nf.pdf`), so grouping by
plot type is what lets you compare models at a glance -- all the posterior predictive
checks in one place, all the noise functions in another. Grouping by model would
scatter each plot type across a dozen folders.

`paper/` is the exception: it holds a copy of whichever variant the manuscript
currently uses, so there is one obvious place to look for "the current Figure 5".

Idempotent -- safe to re-run after generating new figures.

    python -m tms_risk.behavior.scripts.organize_figures            # dry run
    python -m tms_risk.behavior.scripts.organize_figures --apply
"""
import argparse
import shutil
from pathlib import Path

# (folder, [filename prefixes]) -- first match wins, so order matters
GROUPS = [
    ('ppc',            ['ppc_fig3a']),
    ('noise',          ['noise_winner', 'noise_variants', 'noise_functions',
                        'noise_decomposition', 'fig4bc_style', 'localized_noise']),
    ('decision_space', ['fig5', 'decision_space']),
    ('percepts',       ['percept_distortion', 'sfig_s1_']),
    ('mechanism',      ['why_risky_second', 'order_asymmetry', 'pmc_channels',
                        'pmc_explained', 'pmc_mechanism', 'noise_amplitude_link']),
    ('parameters',     ['pmc_parameters', 'ddm_flexible_null_forest']),
    ('model_compare',  ['ssm_']),
    ('imaging',        ['cvr2_', 'm1_', 'm2_', 'prf_params', 'spherical_',
                        'decoder_collapse', 'figure2']),
]

# what the manuscript currently uses -> copied into paper/
PAPER = [
    'figure2a.pdf', 'figure2b.pdf', 'figure2c.pdf', 'figure2_legend.pdf',
    'ppc_fig3a.flexible2nf.pdf',
    'noise_winner.flexible2nf_perception.pdf',
    'fig5.flexible2nf.pdf',
    'why_risky_second.flexible2nf.pdf',
    'percept_distortion.flexible2nf.pdf',
]
SKIP_DIRS = {'paper', 'archive_published_ecc6454', 'archive_superseded'}


def destination(name):
    for folder, prefixes in GROUPS:
        if any(name.startswith(p) for p in prefixes):
            return folder
    return None


def main(fig_dir, apply):
    root = Path(fig_dir)
    moves, unmatched = [], []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            if p.name not in SKIP_DIRS and p.name not in dict(GROUPS):
                unmatched.append(p.name + '/')
            continue
        if p.suffix.lower() not in {'.pdf', '.png', '.svg'}:
            continue
        d = destination(p.name)
        (moves.append((p, root / d / p.name)) if d else unmatched.append(p.name))

    print(f'{len(moves)} files to sort, {len(unmatched)} unmatched')
    by_folder = {}
    for _, dst in moves:
        by_folder[dst.parent.name] = by_folder.get(dst.parent.name, 0) + 1
    for folder, n in sorted(by_folder.items()):
        print(f'  {folder:<16} {n:3d}')
    if unmatched:
        print('  unmatched: ' + ', '.join(unmatched[:12])
              + (' ...' if len(unmatched) > 12 else ''))

    if not apply:
        print('\ndry run -- pass --apply to move')
        return

    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    paper = root / 'paper'
    paper.mkdir(exist_ok=True)
    n_paper = 0
    for name in PAPER:
        d = destination(name)
        src = (root / d / name) if d else (root / name)
        if src.exists():
            shutil.copy(src, paper / name)
            n_paper += 1
        else:
            print(f'  paper/: MISSING {name}')
    print(f'\nmoved {len(moves)} files; paper/ holds {n_paper}/{len(PAPER)} current figures')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fig_dir', default='/Users/gdehol/git/tms_risk/notes/figures')
    parser.add_argument('--apply', action='store_true')
    a = parser.parse_args()
    main(a.fig_dir, a.apply)
