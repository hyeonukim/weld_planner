"""A/B test the seam detector on the local photo sets.

For each photo set, runs the full reconstruction pipeline ONCE (GPU recommended),
captures the workpiece point cloud handed to the seam detector, and runs both the
previous detector (legacy_detector.py) and the current one on that same cloud.

    python compare_photo_sets.py                 # all default sets
    python compare_photo_sets.py --sets tu t     # only some
    python compare_photo_sets.py --from-saved    # re-run detectors only (no models)

Writes compare_output/<set>/{reconstruction.ply, weld_result.json, geometry.npz}
plus compare_output/summary.csv and compare_output/summary.png.
"""
import argparse, csv, json, queue, sys
from pathlib import Path
import numpy as np

DEFAULT_SETS = {
    'g':       ['g1.jpg', 'g2.jpg', 'g3.jpg'],
    'groove':  ['groove1.jpg', 'groove2.jpg', 'groove3.jpg'],
    'l':       ['l1.jpg', 'l2.jpg', 'l3.jpg'],
    't-joint': ['t-joint_1.jpg', 't-joint_2.jpg', 't-joint_3.jpg'],
    't':       ['t1.jpg', 't2.jpg', 't3.jpg'],
    'tu':      ['tu1.jpg', 'tu2.jpg', 'tu3.jpg'],
}


def reconstruct(name, images, out):
    """Run the maintained pipeline; return the detector's captured inputs."""
    import server
    captured = {}
    current = server._detect_seam_geometric

    def capture(workpiece_pts, scene_diag, table_n=None, table_d=None, joint_mode='auto'):
        captured.update(part=np.asarray(workpiece_pts, float), diag=float(scene_diag),
                        table_n=None if table_n is None else np.asarray(table_n, float),
                        table_d=None if table_d is None else float(table_d))
        return current(workpiece_pts, scene_diag, table_n=table_n,
                       table_d=table_d, joint_mode=joint_mode)

    server._detect_seam_geometric = capture
    try:
        server.JOBS[name] = {'queue': queue.Queue(), 'workdir': str(out),
                             'intrinsics': None, 'hfov_deg': None, 'joint_mode': 'auto'}
        server.run_pipeline(name, [str(p) for p in images], str(out))
        job = server.JOBS[name]
        if job.get('status') != 'done':
            msgs = []
            while not job['queue'].empty():
                msgs.append(job['queue'].get())
            raise RuntimeError(f'pipeline failed for {name}: {msgs[-3:]}')
    finally:
        server._detect_seam_geometric = current
    if not captured:
        raise RuntimeError(f'{name}: too few workpiece points; detector never ran')
    np.savez(out / 'geometry.npz', part=captured['part'], diag=captured['diag'],
             table_n=captured['table_n'] if captured['table_n'] is not None else np.full(3, np.nan),
             table_d=captured['table_d'] if captured['table_d'] is not None else np.nan)


def detect_both(out):
    import server
    from legacy_detector import detect_seam_legacy
    server.np = np
    g = np.load(out / 'geometry.npz')
    tn = None if np.isnan(g['table_n']).any() else g['table_n']
    td = None if np.isnan(float(g['table_d'])) else float(g['table_d'])
    res = json.loads((out / 'weld_result.json').read_text(encoding='utf-8'))
    scale = res.get('scale_factor')
    rows = {}
    for tag, fn in [('old', detect_seam_legacy), ('new', server._detect_seam_geometric)]:
        _, _, method, path = fn(g['part'], float(g['diag']), table_n=tn, table_d=td)
        length = None if path is None else float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        rows[tag] = dict(method=method, path=path, length_units=length,
                         length_mm=None if length is None or not scale else length * scale * 1000)
    return g['part'], rows, scale


def plot(results, path_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(results)
    cols = min(3, n)
    rws = (n + cols - 1) // cols
    fig, axs = plt.subplots(rws, cols, figsize=(5.4 * cols, 5.4 * rws), squeeze=False)
    for ax in axs.ravel()[n:]:
        ax.axis('off')
    for ax, (name, part, rows, scale) in zip(axs.ravel(), results):
        # View straight down the most informative direction: drop the PCA axis
        # with least spread, so the seam and both faces stay visible.
        c = part.mean(axis=0)
        _, _, vt = np.linalg.svd(part - c, full_matrices=False)
        k = 1000 * scale if scale else 1.0
        proj = lambda X: (X - c) @ vt[:2].T * k
        P = proj(part)
        ax.scatter(P[:, 0], P[:, 1], s=.4, c='#c8c8c8', rasterized=True)
        for tag, col, ls in [('old', '#eb6834', '--'), ('new', '#2a78d6', '-')]:
            r = rows[tag]
            if r['path'] is None:
                ax.plot([], [], ls, c=col, lw=2, label=f"{'before' if tag == 'old' else 'after'}: none")
                continue
            Q = proj(r['path'])
            L = f"{r['length_mm']:.0f} mm" if r['length_mm'] else f"{r['length_units']:.3f} u"
            ax.plot(Q[:, 0], Q[:, 1], ls, c=col, lw=2.2,
                    label=f"{'before' if tag == 'old' else 'after'}: {L}, {r['method'].split(':')[0]}")
            ax.plot(Q[[0, -1], 0], Q[[0, -1], 1], 'o', c=col, ms=6, mec='white', mew=1.5)
        ax.set_title(name, fontweight='bold')
        lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
        pad = .15 * (hi - lo).max()
        ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='lower center', fontsize=8, frameon=False, bbox_to_anchor=(.5, -.2))
    fig.suptitle('Seam detection: before (orange, dashed)\nvs after (blue, solid)',
                 fontweight='bold', fontsize=11, x=.02, ha='left')
    plt.tight_layout()
    plt.savefig(path_png, dpi=110)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sets', nargs='+', default=list(DEFAULT_SETS))
    ap.add_argument('--output', type=Path, default=Path('compare_output'))
    ap.add_argument('--from-saved', action='store_true',
                    help='skip reconstruction; reuse <output>/<set>/geometry.npz')
    a = ap.parse_args()
    results, table = [], []
    for name in a.sets:
        images = [Path(p) for p in DEFAULT_SETS.get(name, [])]
        out = a.output / name
        out.mkdir(parents=True, exist_ok=True)
        print(f'=== {name} ===', flush=True)
        try:
            if not a.from_saved:
                missing = [str(p) for p in images if not p.is_file()]
                if missing:
                    raise RuntimeError(f'missing images: {missing}')
                reconstruct(name, [p.resolve() for p in images], out.resolve())
            part, rows, scale = detect_both(out)
        except Exception as e:                      # keep going with other sets
            print(f'  FAILED: {e}', flush=True)
            table.append({'set': name, 'error': str(e)})
            continue
        results.append((name, part, rows, scale))
        row = {'set': name, 'points': len(part), 'scale_m_per_unit': scale}
        for tag in ('old', 'new'):
            r = rows[tag]
            row[f'{tag}_method'] = r['method']
            row[f'{tag}_waypoints'] = 0 if r['path'] is None else len(r['path'])
            row[f'{tag}_length_mm'] = None if r['length_mm'] is None else round(r['length_mm'], 1)
            print(f"  {tag}: {r['method']}  length_mm={row[f'{tag}_length_mm']}", flush=True)
        table.append(row)
    keys = ['set', 'points', 'scale_m_per_unit', 'old_method', 'old_waypoints', 'old_length_mm',
            'new_method', 'new_waypoints', 'new_length_mm', 'error']
    with open(a.output / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in table:
            w.writerow({k: r.get(k) for k in keys})
    if results:
        plot(results, a.output / 'summary.png')
    print(f'Wrote {a.output / "summary.csv"} and {a.output / "summary.png"}')
