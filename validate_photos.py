"""Run the browser pipeline on local photos and save a reviewable result."""
import argparse
import queue
from pathlib import Path
from server import JOBS, run_pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('validation_output'))
    parser.add_argument('--joint-mode', choices=['auto', 'base'], default='auto')
    args = parser.parse_args()
    if not 2 <= len(args.images) <= 6 or not all(p.is_file() for p in args.images):
        parser.error('Provide 2–6 existing image files')
    args.output.mkdir(parents=True, exist_ok=True)
    JOBS['validation'] = {'queue': queue.Queue(), 'workdir': str(args.output.resolve()),
                          'intrinsics': None, 'hfov_deg': None,
                          'joint_mode': args.joint_mode}
    run_pipeline('validation', [str(p.resolve()) for p in args.images], str(args.output.resolve()))
    job = JOBS['validation']
    while not job['queue'].empty():
        message = job['queue'].get()
        if message and not message.startswith('event: done'):
            print(message)
    if job.get('status') != 'done':
        raise SystemExit(1)
    result = job['result']
    print(result['detection_method'])
    print(f"Waypoints: {result['waypoint_count']}; length (mm): {result['seam_length_mm']}")
    print(f'Preview with: python preview_saved.py "{args.output}"')
