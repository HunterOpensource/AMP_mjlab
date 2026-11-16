"""Hunter180 model, configured PD dynamics, reference data and GPU integration."""

import hashlib
import json
import os
from pathlib import Path
import unittest

import mjlab  # Initialize task plugins before importing their RL dependencies.
import mujoco
import numpy as np
import torch
from mjlab.entity import Entity

from rsl_rl.utils.motion_loader import AMPLoader
from rsl_rl.utils.motion_validation import validate_motion
from src.assets.robots.hunter180 import HUNTER180_XML, get_action_scale, get_hunter180_robot_cfg, load_motor_parameters
from src.motion_retarget.hunter180 import Hunter180Retargeter, SOURCE_MOTIONS
from src.tasks.amp_loco.config.hunter180.constants import AMP_ANCHOR_NAME, AMP_BODY_NAMES, MOTION_DIR


class Hunter180Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(HUNTER180_XML))
        cls.joints = [cls.model.joint(i).name for i in range(1, cls.model.njnt)]
        cls.bodies = [cls.model.body(i).name for i in range(1, cls.model.nbody)]

    def test_pd_dynamics_and_torque_limits(self):
        entity = Entity(get_hunter180_robot_cfg())
        model = entity.spec.compile()
        self.assertEqual((model.nq, model.nv, model.nu), (30, 29, 23))
        self.assertEqual(list(entity.joint_names), self.joints)
        motors = load_motor_parameters()
        data = mujoco.MjData(model)
        expected_force, effort_limits = [], []
        for i in range(model.nu):
            joint = int(model.actuator_trnid[i, 0])
            name = model.joint(joint).name
            dof = int(model.jnt_dofadr[joint])
            motor = motors[name]
            self.assertEqual(model.dof_armature[dof], motor["armature"])
            self.assertEqual(model.actuator_gainprm[i, 0], motor["stiffness"])
            self.assertEqual(model.actuator_biasprm[i, 2], -motor["damping"])
            np.testing.assert_allclose(model.actuator_forcerange[i], [-motor["effort"], motor["effort"]])
            np.testing.assert_allclose(model.jnt_actfrcrange[joint], [-motor["effort"], motor["effort"]])
            self.assertEqual(get_action_scale()[name], motor["action_scale"])
            data.qvel[dof] = 0.2
            data.ctrl[i] = data.qpos[model.jnt_qposadr[joint]] + 0.05
            expected_force.append(np.clip(0.05 * motor["stiffness"] - 0.2 * motor["damping"],
                                          -motor["effort"], motor["effort"]))
            effort_limits.append(motor["effort"])
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.actuator_force, expected_force, atol=1e-10)
        for sign in (-1, 1):
            data.ctrl[:] = sign * 1e6
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.actuator_force, sign * np.array(effort_limits))

    def test_registered_tasks_and_play_configs(self):
        import src.tasks
        from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

        for terrain in ("Flat", "Rough"):
            task = f"Hunter180-AMP-{terrain}"
            self.assertIn(task, list_tasks())
            rl = load_rl_cfg(task)
            self.assertEqual(len(rl.min_normalized_std), 23)
            self.assertEqual(Path(rl.amp_motion_files), MOTION_DIR)
            self.assertTrue(set(rl.amp_body_names) <= set(self.bodies))
            for play in (False, True):
                cfg = load_env_cfg(task, play=play)
                loader = cfg.events["init_motion_loader"].params
                self.assertEqual(Path(loader["motion_dir"]), MOTION_DIR / "WalkandRun")
                self.assertEqual(Path(loader["recovery_dir"]), MOTION_DIR / "Recovery")
                self.assertEqual(loader["delay_reset_env_ratio"], 1.0 if play else 0.4)
                self.assertEqual("command_vel" in cfg.curriculum, not play)
                self.assertEqual(cfg.terminations["bad_base_height"].params["minimum_height"], 0.5)

    def test_bundled_motion_integrity_and_fk(self):
        manifest = json.loads((MOTION_DIR / "manifest.json").read_text())
        self.assertEqual(len(manifest["clips"]), 18)
        self.assertEqual(manifest["source_robot"], "hunter130")
        self.assertEqual(manifest["target_robot"], "hunter180")
        self.assertEqual(manifest["total_frames"], 9992)
        self.assertEqual(hashlib.sha256(HUNTER180_XML.read_bytes()).hexdigest(), manifest["target_model_sha256"])
        total = 0
        data = mujoco.MjData(self.model)
        limits = self.model.jnt_range[1:]
        center = limits.mean(axis=1)
        half = np.diff(limits, axis=1).ravel() * 0.9 / 2
        self.assertEqual({c["file"] for c in manifest["clips"]},
                         {str(p.relative_to(SOURCE_MOTIONS)) for p in SOURCE_MOTIONS.rglob("*.npz")})
        for clip in manifest["clips"]:
            path = MOTION_DIR / clip["file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), clip["sha256"])
            with np.load(path) as archive:
                motion = dict(archive)
            validate_motion(motion, path, self.bodies, self.joints, 50, True)
            self.assertEqual(len(motion["joint_pos"]), clip["frames"])
            total += clip["frames"]
            source_path = SOURCE_MOTIONS / clip["file"]
            self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), clip["source_sha256"])
            with np.load(source_path) as source:
                self.assertEqual(motion["joint_pos"].shape, (len(source["joint_pos"]), 23))
                dots = np.sum(source["body_quat_w"][:, 0] * motion["body_quat_w"][:, 0], axis=-1)
                np.testing.assert_allclose(np.abs(dots), 1, atol=2e-6)
            self.assertTrue(np.all(motion["joint_pos"] >= center - half - 2e-7))
            self.assertTrue(np.all(motion["joint_pos"] <= center + half + 2e-7))
            np.testing.assert_allclose(np.gradient(motion["joint_pos"], .02, axis=0), motion["joint_vel"], atol=2e-5)
            self.assertLessEqual(np.abs(motion["joint_vel"]).max(), 20.001)
            self.assertGreaterEqual(clip["min_collision_height_m"], -1e-6)
            self.assertLessEqual(clip["max_self_penetration_m"], .015)
            for t in (0, len(motion["joint_pos"]) // 2, len(motion["joint_pos"]) - 1):
                data.qpos[:] = np.r_[motion["body_pos_w"][t, 0], motion["body_quat_w"][t, 0], motion["joint_pos"][t]]
                mujoco.mj_kinematics(self.model, data)
                np.testing.assert_allclose(data.xpos[1:], motion["body_pos_w"][t], atol=3e-6)
        self.assertEqual(total, manifest["total_frames"])

    def test_hunter130_to_hunter180_ik_jacobian(self):
        with np.load(SOURCE_MOTIONS / "Recovery/fallAndGetUp1_subject1.npz") as archive:
            motion = {k: (v[420:424] if v.ndim > 1 else v) for k in archive.files for v in [archive[k]]}
        retargeter = Hunter180Retargeter()
        qpos, _, _, _ = retargeter.retarget(motion)
        x = np.r_[qpos[-1, :3], qpos[-1, 7:]]
        # Exercise the floor penalty away from its non-differentiable boundary.
        x[2] -= 0.01
        _, analytical = retargeter._evaluate(x)
        numerical = np.empty_like(analytical)
        for i in range(len(x)):
            step = np.zeros_like(x)
            step[i] = 1e-6
            numerical[:, i] = (retargeter._evaluate(x + step)[0] - retargeter._evaluate(x - step)[0]) / 2e-6
        np.testing.assert_allclose(analytical, numerical, atol=2e-4, rtol=2e-4)

    @unittest.skipUnless(os.environ.get("HUNTER180_TEST_GPU") == "1" and torch.cuda.is_available(),
                         "Set HUNTER180_TEST_GPU=1 for CUDA observations, actions, resets and rough terrain")
    def test_gpu_amp_observations_actions_and_terrain(self):
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg
        import src.tasks

        loader = AMPLoader(str(MOTION_DIR), AMP_BODY_NAMES, AMP_ANCHOR_NAME, self.bodies,
                           "cuda:0", self.joints, 50)
        path = sorted(MOTION_DIR.rglob("*.npz"))[0]
        with np.load(path) as archive:
            motion = dict(archive)
        for terrain in ("Flat", "Rough"):
            cfg = load_env_cfg(f"Hunter180-AMP-{terrain}")
            cfg.scene.num_envs = 4
            cfg.episode_length_s = 0.08
            if terrain == "Rough":
                cfg.scene.terrain.terrain_generator.num_rows = 2
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.max_init_terrain_level = 0
            env = ManagerBasedRlEnv(cfg, device="cuda:0")
            try:
                env.reset()
                robot = env.scene["robot"]
                def batch(v):
                    return torch.as_tensor(v, device=env.device).float().expand(env.num_envs, -1).clone()
                robot.write_root_link_pose_to_sim(torch.cat((batch(motion["body_pos_w"][0, 0]) + env.scene.env_origins,
                                                             batch(motion["body_quat_w"][0, 0])), dim=-1))
                robot.write_root_link_velocity_to_sim(batch(np.r_[motion["body_lin_vel_w"][0, 0], motion["body_ang_vel_w"][0, 0]]))
                robot.write_joint_state_to_sim(batch(motion["joint_pos"][0]), batch(motion["joint_vel"][0]))
                env.sim.forward()
                env.scene.update(env.step_dt)
                obs = env.observation_manager.compute(update_history=True)["amp"]
                torch.testing.assert_close(obs, loader._features[0].expand_as(obs), atol=4e-4, rtol=4e-4)
                action = env.action_manager.get_term("joint_pos")
                raw = torch.full((env.num_envs, 23), 0.1, device=env.device)
                action.process_actions(raw)
                scales = torch.tensor([get_action_scale()[n] for n in action.target_names], device=env.device)
                torch.testing.assert_close(action._processed_actions, raw * scales + robot.data.default_joint_pos[:, action.target_ids])
                saw_timeout = False
                for _ in range(12):
                    observations, reward, _, truncated, _ = env.step(torch.zeros_like(raw))
                    self.assertTrue(torch.isfinite(reward).all())
                    self.assertTrue(all(torch.isfinite(v).all() for v in observations.values()))
                    saw_timeout |= bool(truncated.any())
                self.assertTrue(saw_timeout)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
