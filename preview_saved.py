"""Open an exported reconstruction and result in the ROS-free browser viewer."""
import argparse
import json
from pathlib import Path
from server import app, JOBS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path,
                        help='Directory containing reconstruction.ply and weld_result.json')
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not (directory / 'reconstruction.ply').is_file():
        parser.error('reconstruction.ply is missing')
    result = json.loads((directory / 'weld_result.json').read_text(encoding='utf-8'))
    JOBS['saved'] = {'status': 'done', 'workdir': str(directory), 'result': result}
    print(f'Preview: http://127.0.0.1:{args.port}/?job=saved', flush=True)
    app.run(host='127.0.0.1', port=args.port, threaded=True)
