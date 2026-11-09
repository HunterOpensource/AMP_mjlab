"""Baseline integrity checks; optional source checks use HUNTER130_G1_SOURCE."""

import hashlib
import json
import os
from pathlib import Path
import unittest

import mujoco
import numpy as np

from rsl_rl.utils.motion_validation import validate_motion
from src.motion_retarget.hunter130 import HUNTER_XML, Retargeter
from src.tasks.amp_loco.config.hunter130.constants import MOTION_DIR


class BaselineTests(unittest.TestCase):
    def test_bundled_baseline_integrity_and_constraints(self):
        manifest = json.loads((MOTION_DIR / 'manifest.json').read_text())
        expected = {entry['file'] for entry in manifest['clips']}
        actual = {str(p.relative_to(MOTION_DIR)) for p in MOTION_DIR.rglob('*.npz')}
        self.assertEqual(actual, expected)
        self.assertEqual(len(manifest['clips']), 18)
        self.assertEqual(sum(p.startswith('WalkandRun/') for p in actual), 17)
        self.assertEqual(sum(p.startswith('Recovery/') for p in actual), 1)
        self.assertEqual(manifest['total_frames'], 9992)
        self.assertEqual(sum(c['frames'] for c in manifest['clips']), 9992)
        self.assertEqual(manifest['target_model_sha256'], hashlib.sha256(HUNTER_XML.read_bytes()).hexdigest())
        model = mujoco.MjModel.from_xml_path(str(HUNTER_XML))
        joints = [model.joint(i).name for i in range(1, model.njnt)]
        bodies = [model.body(i).name for i in range(1, model.nbody)]
        self.assertEqual(joints, manifest['joint_names'])
        self.assertEqual(bodies, manifest['body_names'])
        limits = model.jnt_range[1:]
        center = limits.mean(axis=1)
        extent = np.diff(limits, axis=1).ravel() * manifest['settings']['soft_limit_factor'] / 2
        for entry in manifest['clips']:
            with self.subTest(clip=entry['file']):
                path = MOTION_DIR / entry['file']
                self.assertEqual(entry['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
                with np.load(path) as data:
                    validate_motion(data, path, bodies, joints, 50, True)
                    self.assertEqual(data['joint_pos'].shape, (entry['frames'], 25))
                    self.assertEqual(data['body_pos_w'].shape, (entry['frames'], 26, 3))
                    self.assertEqual(data['source_sha256'].item(), entry['source_sha256'])
                    provenance = json.loads(data['retarget_metadata'].item())
                    # The hash identifies generation-time code, retained at the
                    # manifest's retarget_code_revision, before source-path API changes.
                    for key in ('retarget_version', 'source_model_sha256', 'target_model_sha256',
                                'retarget_code_sha256', 'settings'):
                        self.assertEqual(provenance[key], manifest[key])
                    self.assertTrue(np.all(data['joint_pos'] >= center - extent - 2e-7))
                    self.assertTrue(np.all(data['joint_pos'] <= center + extent + 2e-7))
                    self.assertLessEqual(np.abs(np.diff(data['joint_pos'], axis=0)).max(), .400001)
                    np.testing.assert_allclose(np.gradient(data['joint_pos'], .02, axis=0),
                                               data['joint_vel'], atol=2e-5)
                self.assertGreaterEqual(entry['min_collision_height_m'], 0.)
                self.assertLess(entry['max_self_penetration_m'], .015)


@unittest.skipUnless(os.environ.get('HUNTER130_G1_SOURCE'),
                     'Set HUNTER130_G1_SOURCE to an external original G1 checkout for regeneration checks')
class RetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ['HUNTER130_G1_SOURCE']).expanduser().resolve()
        cls.source = root / 'src/assets/motions/g1/amp'
        cls.source_xml = root / 'src/assets/robots/unitree_g1/xmls/g1.xml'

    def test_source_identity_timing_and_root_orientation(self):
        manifest = json.loads((MOTION_DIR / 'manifest.json').read_text())
        self.assertEqual(manifest['source_model_sha256'], hashlib.sha256(self.source_xml.read_bytes()).hexdigest())
        self.assertEqual({str(p.relative_to(self.source)) for p in self.source.rglob('*.npz')},
                         {entry['file'] for entry in manifest['clips']})
        for entry in manifest['clips']:
            with self.subTest(clip=entry['file']):
                source_path = self.source / entry['file']
                self.assertEqual(entry['source_sha256'], hashlib.sha256(source_path.read_bytes()).hexdigest())
                with np.load(source_path) as src, np.load(MOTION_DIR / entry['file']) as dst:
                    self.assertEqual(len(src['joint_pos']), len(dst['joint_pos']))
                    self.assertEqual(float(src['fps'].item()), float(dst['fps'].item()))
                    dot = (src['body_quat_w'][:, 0] * dst['body_quat_w'][:, 0]).sum(axis=-1)
                    np.testing.assert_allclose(np.abs(dot), 1., atol=2e-6)

    def test_ik_jacobian_including_contact_avoidance(self):
        with np.load(self.source / 'Recovery/fallAndGetUp1_subject1.npz') as archive:
            data = {k: (v[420:424] if v.ndim > 1 else v) for k in archive.files for v in [archive[k]]}
        r = Retargeter(source_xml=self.source_xml)
        qpos, _, _, _ = r.retarget(data)
        x = np.r_[qpos[-1, :3], qpos[-1, 7:]]
        _, analytical = r._evaluate(x)
        numerical = np.empty_like(analytical)
        for i in range(len(x)):
            step = np.zeros_like(x)
            step[i] = 1e-6
            numerical[:, i] = (r._evaluate(x + step)[0] - r._evaluate(x - step)[0]) / 2e-6
        np.testing.assert_allclose(analytical, numerical, atol=2e-4, rtol=2e-4)


if __name__ == '__main__':
    unittest.main()
