"""Rebuild the Hunter130 baseline from an external checkout of the original G1 repository."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

# Tiny IK systems are faster with one BLAS thread; this only affects this tool.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import mujoco
import numpy as np

from rsl_rl.utils.motion_validation import validate_motion
from src import SRC_PATH
from src.motion_retarget.hunter130 import HUNTER_XML, RetargetCfg, Retargeter, states_from_qpos

DEFAULT_OUTPUT = SRC_PATH / 'assets/motions/hunter130/amp_baseline'
VERSION = 1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(retargeter, qpos, contacts, targets, stats, output):
    model, data = retargeter.model, mujoco.MjData(retargeter.model)
    points, min_floor, self_depth = [], [], []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        points.append([retargeter.point(data, body, local) for body, local, _, _, _ in retargeter.points])
        self_depth.append(min([0.] + [c.dist for c in data.contact]))
        retargeter.data.qpos[:] = q
        mujoco.mj_kinematics(model, retargeter.data)
        min_floor.append(retargeter.floor_points()[:, 2].min())
    points = np.array(points)
    errors = np.linalg.norm(points - targets, axis=-1)
    speeds = np.linalg.norm(np.gradient(points[:, :2, :2], .02, axis=0), axis=-1)
    # Exclude stance boundary frames when measuring sliding during a contact.
    stable_contact = contacts[:, :2].copy()
    stable_contact[1:] &= contacts[:-1, :2]
    stable_contact[:-1] &= contacts[1:, :2]
    stance_speeds = speeds[stable_contact]
    assert stance_speeds.size, 'No stable foot contacts found'
    return {
        'frames': len(qpos), 'fps': 50., 'duration_s': (len(qpos) - 1) / 50,
        'root_foot_scale': retargeter.scale,
        'foot_position_error_mean_m': float(errors[:, :2].mean()),
        'foot_position_error_max_m': float(errors[:, :2].max()),
        'hand_position_error_mean_m': float(errors[:, [5, 8]].mean()),
        'hand_position_error_max_m': float(errors[:, [5, 8]].max()),
        'stance_slip_median_m_s': float(np.median(stance_speeds)),
        'stance_slip_p95_m_s': float(np.quantile(stance_speeds, .95)),
        'stance_slip_max_m_s': float(np.max(stance_speeds)),
        'min_collision_height_m': float(min(min_floor)),
        'max_floor_lift_m': float(stats['floor_lift'].max()),
        'max_self_penetration_m': float(-min(self_depth)),
        'max_joint_speed_rad_s': float(np.abs(output['joint_vel']).max()),
        'max_joint_step_rad': float(np.abs(np.diff(qpos[:, 7:], axis=0)).max()),
        'solver_mean_evaluations': float(stats['nfev'].mean()),
        'solver_budget_fraction': float(np.mean(stats['nfev'] >= retargeter.cfg.max_nfev)),
    }


def build_dataset(source_repo, output=DEFAULT_OUTPUT, cfg=None, clips=None):
    source_repo = Path(source_repo).expanduser().resolve()
    source = source_repo / 'src/assets/motions/g1/amp'
    source_xml = source_repo / 'src/assets/robots/unitree_g1/xmls/g1.xml'
    output = Path(output).expanduser().resolve()
    if not source_xml.is_file() or not source.is_dir():
        raise FileNotFoundError(
            f'{source_repo} must contain the original G1 model and AMP motions; '
            'see the source revision and checkout commands in README_zh.md'
        )
    cfg = cfg or RetargetCfg()
    files = sorted(source.glob('WalkandRun/*.npz')) + sorted(source.glob('Recovery/*.npz'))
    if clips:
        files = [p for p in files if p.stem in clips]
        if output == DEFAULT_OUTPUT.resolve():
            raise ValueError('Partial runs must use --output outside the default baseline directory')
        if {p.stem for p in files} != set(clips):
            raise ValueError('Some --clips names were not found in the source dataset')
    elif (sum(p.parent.name == 'WalkandRun' for p in files), sum(p.parent.name == 'Recovery' for p in files)) != (17, 1):
        raise ValueError('Expected the original 17 locomotion and 1 recovery clips')
    if not files:
        raise ValueError(f'No source NPZ files in {source}')
    relative_files = {str(p.relative_to(source)) for p in files}
    existing = {str(p.relative_to(output)) for p in output.rglob('*.npz')}
    if existing - relative_files:
        raise ValueError(f'Destination contains unrelated NPZ clips: {sorted(existing - relative_files)}')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Source and output dataset trees must be separate')
    retargeter = Retargeter(cfg, source_xml=source_xml)
    provenance = {
        'retarget_version': VERSION,
        'source_model_sha256': sha256(source_xml), 'target_model_sha256': sha256(HUNTER_XML),
        'retarget_code_sha256': sha256(Path(__file__).parents[1] / 'src/motion_retarget/hunter130.py'),
        'settings': asdict(cfg),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Publish only after every clip passes validation; an interrupted solve keeps
    # the current dataset intact. Staging is on the destination filesystem.
    with tempfile.TemporaryDirectory(prefix='hunter130-retarget-', dir=output.parent) as temporary:
        stage = Path(temporary)
        summaries = []
        for index, path in enumerate(files):
            relative = path.relative_to(source)
            with np.load(path) as archive:
                motion = dict(archive)
            validate_motion(motion, path, retargeter.source_body_names, retargeter.source_joint_names, 50)
            print(f'[{index+1}/{len(files)}] {relative}: {len(motion["joint_pos"])} frames', flush=True)
            qpos, contacts, targets, stats = retargeter.retarget(motion)
            result = states_from_qpos(retargeter.model, qpos, 50)
            validate_motion(result, relative, retargeter.body_names, retargeter.joint_names, 50, True)
            if not (np.all(qpos[:, 7:] >= retargeter.lower - 1e-7) and np.all(qpos[:, 7:] <= retargeter.upper + 1e-7)):
                raise ValueError(f'{relative}: joint limits violated')
            summary = summarize(retargeter, qpos, contacts, targets, stats, result)
            if summary['min_collision_height_m'] < -1e-6 or summary['max_joint_speed_rad_s'] > cfg.max_joint_speed + 1e-3:
                raise ValueError(f'{relative}: floor/velocity validation failed: {summary}')
            if summary['max_self_penetration_m'] > .015:
                raise ValueError(f'{relative}: self penetration exceeds 15 mm: {summary}')
            result.update(
                source=np.array(str(path.relative_to(SRC_PATH.parent)) if path.is_relative_to(SRC_PATH.parent) else str(path)),
                source_sha256=np.array(sha256(path)), input_fps=np.array(50., dtype=np.float32),
                retarget_metadata=np.array(json.dumps(provenance, sort_keys=True)),
                foot_contact=contacts[:, :2].astype(np.uint8),
            )
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **result)
            summary.update(file=str(relative), source_sha256=sha256(path), sha256=sha256(target))
            summaries.append(summary)
            print(f'  foot mean={summary["foot_position_error_mean_m"]*1000:.1f} mm, '
                  f'stance slip p95={summary["stance_slip_p95_m_s"]:.3f} m/s, '
                  f'self penetration={summary["max_self_penetration_m"]*1000:.1f} mm', flush=True)
        manifest = dict(provenance, source=str(source), fps=50.,
                        joint_names=retargeter.joint_names, body_names=retargeter.body_names,
                        clips=summaries, total_frames=sum(s['frames'] for s in summaries))
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                target = output / path.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                path.replace(target)
    print(f'Prepared {len(summaries)} clips / {manifest["total_frames"]} frames in {time.monotonic()-started:.1f}s: {output}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo', type=Path, required=True,
                        help='External original G1 checkout containing its model, meshes and NPZ motions')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--clips', nargs='+', help='Optional clip stems for a separate diagnostic output')
    parser.add_argument('--max-nfev', type=int, default=RetargetCfg.max_nfev)
    args = parser.parse_args()
    build_dataset(args.source_repo, args.output, RetargetCfg(max_nfev=args.max_nfev), args.clips)
