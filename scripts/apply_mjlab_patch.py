"""Install the observation-history patch into the active Python environment."""

import argparse
from importlib.metadata import distribution
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check without changing the environment')
    args = parser.parse_args()
    package = distribution('mjlab')
    if package.version != '1.2.0':
        raise SystemExit(f'This patch requires mjlab 1.2.0, found {package.version}')
    relative = Path('mjlab/managers/observation_manager.py')
    source = Path(__file__).resolve().parents[1] / 'mjlab_patch' / relative
    target = Path(package.locate_file(relative))
    if target.read_bytes() == source.read_bytes():
        print(f'mjlab observation patch is installed: {target}')
        return
    if args.check:
        raise SystemExit('Patch missing. Run: python scripts/apply_mjlab_patch.py')
    backup = target.with_name(target.name + '.amp_mjlab.bak')
    if not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(source, target)
    print(f'Installed mjlab observation patch: {target}')
    print(f'Original file backup: {backup}')


if __name__ == '__main__':
    main()
