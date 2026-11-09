"""Run with: python -m unittest discover -s tests -v."""

from pathlib import Path
import os
import tempfile
import unittest

import mjlab  # Initialize task plugins before importing their RL dependencies.
import mujoco
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionActionCfg
from scipy.spatial.transform import Rotation

from rsl_rl.utils.motion_loader import AMPLoader
from rsl_rl.utils.motion_validation import validate_motion
from src.assets.robots.hunter130 import (
    HUNTER130_XML, get_action_scale, get_hunter130_robot_cfg, load_motor_parameters,
)
from src.tasks.amp_loco.config.hunter130.constants import AMP_ANCHOR_NAME, AMP_BODY_NAMES, MOTION_DIR


class Hunter130Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(HUNTER130_XML))
        cls.joints = [cls.model.joint(i).name for i in range(1, cls.model.njnt)]
        cls.bodies = [cls.model.body(i).name for i in range(1, cls.model.nbody)]

    def test_compiled_pd_matches_motor_json(self):
        entity = Entity(get_hunter130_robot_cfg())
        model = entity.spec.compile()
        motors = load_motor_parameters()
        self.assertEqual((model.nq, model.nv, model.nu), (32, 31, 25))
        self.assertEqual(list(entity.joint_names), self.joints)
        data = mujoco.MjData(model)
        expected = []
        for i in range(model.nu):
            joint = int(model.actuator_trnid[i, 0])
            name = model.joint(joint).name
            motor = motors[name]
            self.assertEqual(model.dof_armature[model.jnt_dofadr[joint]], motor["armature"])
            np.testing.assert_allclose(model.actuator_forcerange[i], [-motor["effort"], motor["effort"]])
            np.testing.assert_allclose(model.jnt_actfrcrange[joint], [-motor["effort"], motor["effort"]])
            self.assertEqual(get_action_scale()[name], motor["action_scale"])
            self.assertEqual(model.actuator_gainprm[i, 0], motor["stiffness"])
            self.assertEqual(model.actuator_biasprm[i, 2], -motor["damping"])
            data.qvel[model.jnt_dofadr[joint]] = 0.2
            data.ctrl[i] = data.qpos[model.jnt_qposadr[joint]] + 0.05
            expected.append(np.clip(0.05 * motor["stiffness"] - 0.2 * motor["damping"],
                                    -motor["effort"], motor["effort"]))
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.actuator_force, expected, atol=1e-10)

    def _motion(self):
        path = MOTION_DIR / "WalkandRun" / "walk_forward_loop_002__A022.npz"
        self.assertTrue(path.is_file(), "The baseline motions must be included in the checkout")
        with np.load(path) as archive:
            data = dict(archive)
        # A short segment is enough to verify FK, metadata and AMP features.
        for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            data[key] = data[key][100:104].copy()
        return data

    def test_registered_configs_use_bundled_hunter130_data(self):
        import src.tasks
        from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

        tasks = list_tasks()
        self.assertNotIn("Unitree-G1-AMP-Flat", tasks)
        self.assertNotIn("Unitree-G1-AMP-Rough", tasks)
        for task in ("Hunter130-AMP-Flat", "Hunter130-AMP-Rough"):
            with self.subTest(task=task):
                self.assertIn(task, tasks)
                env_cfg, rl_cfg = load_env_cfg(task), load_rl_cfg(task)
                self.assertEqual(Path(rl_cfg.amp_motion_files), MOTION_DIR)
                self.assertEqual(len(rl_cfg.min_normalized_std), 25)
                for play in (False, True):
                    task_cfg = load_env_cfg(task, play=play)
                    self.assertIs(type(task_cfg.actions["joint_pos"]), JointPositionActionCfg)
                    self.assertEqual(task_cfg.terminations["bad_base_height"].params["minimum_height"], 0.35)
                loader = env_cfg.events["init_motion_loader"].params
                self.assertEqual(Path(loader["motion_dir"]), MOTION_DIR / "WalkandRun")
                self.assertEqual(Path(loader["recovery_dir"]), MOTION_DIR / "Recovery")

    def test_motion_fk_and_vectorized_amp_features(self):
        motion = self._motion()
        validate_motion(motion, "test", self.bodies, self.joints, 50, True)
        data = mujoco.MjData(self.model)
        data.qpos[:3] = motion["body_pos_w"][0, 0]
        data.qpos[3:7] = motion["body_quat_w"][0, 0]
        data.qpos[7:] = motion["joint_pos"][0]
        data.qvel[:3] = motion["body_lin_vel_w"][0, 0]
        root_rot = Rotation.from_quat(data.qpos[[4, 5, 6, 3]])
        data.qvel[3:6] = root_rot.inv().apply(motion["body_ang_vel_w"][0, 0])
        data.qvel[6:] = motion["joint_vel"][0]
        mujoco.mj_forward(self.model, data)
        np.testing.assert_allclose(data.xpos[1:], motion["body_pos_w"][0], atol=3e-6)
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(Path(tmp) / "motion.npz", **motion)
            loader = AMPLoader(tmp, AMP_BODY_NAMES, AMP_ANCHOR_NAME, self.bodies,
                               "cpu", self.joints, 50)
        indexes = [self.bodies.index(name) for name in AMP_BODY_NAMES]
        anchor = self.bodies.index(AMP_ANCHOR_NAME)
        q = motion["body_quat_w"][0]
        rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]])
        pos = rotations[anchor].inv().apply(motion["body_pos_w"][0, indexes] - motion["body_pos_w"][0, anchor])
        ori = (rotations[anchor].inv() * rotations[indexes]).as_matrix()[..., :2]
        lin = rotations[indexes].inv().apply(motion["body_lin_vel_w"][0, indexes])
        ang = rotations[indexes].inv().apply(motion["body_ang_vel_w"][0, indexes])
        expected = np.concatenate([value.ravel() for value in (pos, ori, lin, ang)])
        np.testing.assert_allclose(loader._features[0], expected, atol=2e-5)
        for i, body in enumerate(indexes):
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, body + 1, velocity, 0)
            np.testing.assert_allclose(velocity[:3], motion["body_ang_vel_w"][0, body], atol=2e-5)
            # mj_objectVelocity uses the inertial COM, shift it to the link origin.
            link_vel = velocity[3:] + np.cross(velocity[:3], data.xpos[body + 1] - data.xipos[body + 1])
            np.testing.assert_allclose(link_vel, motion["body_lin_vel_w"][0, body], atol=2e-5)

    def test_incompatible_motion_rejected(self):
        for changed in ("joint_names", "body_names", "fps", "joint_vel", "body_quat_w"):
            with self.subTest(changed=changed):
                motion = self._motion()
                if changed.endswith("names"):
                    motion[changed] = motion[changed][::-1]
                elif changed == "fps":
                    motion[changed] = np.array(30.)
                elif changed == "joint_vel":
                    motion[changed][0, 0] = np.nan
                else:
                    motion[changed][0, 0] = 0
                with self.assertRaises(ValueError):
                    validate_motion(motion, "bad.npz", self.bodies, self.joints, 50, True)

    def test_sampler_cycles_clips_and_repeats_terminal_frame(self):
        motion = self._motion()
        with tempfile.TemporaryDirectory() as tmp:
            # Encode clip and frame IDs in root-local angular velocity.
            motion["body_quat_w"][:] = [1, 0, 0, 0]
            for clip in range(18):
                motion["body_ang_vel_w"][:] = (clip * 100 + np.arange(4))[:, None, None]
                np.savez(Path(tmp) / f"{clip:02d}.npz", **motion)
            loader = AMPLoader(tmp, ["pelvis"], "pelvis", self.bodies, "cpu", self.joints, 50)
        torch.manual_seed(42)
        batch_clips = []
        for current, following in loader.feed_forward_generator(20, 4096):
            clip_ids = torch.unique(current[:, -1].long() // 100)
            self.assertEqual(len(clip_ids), 1)
            batch_clips.append(int(clip_ids[0]))
            frames = current[:, -1].long() % 100
            self.assertEqual(set(frames.tolist()), {0, 1, 2, 3})
            # A sampled final frame repeats itself instead of crossing clips.
            expected = current[:, -1] + (frames < 3).float()
            torch.testing.assert_close(following[:, -1], expected)
        self.assertEqual(batch_clips, list(range(18)) + [0, 1])
        # Each PPO update creates a new generator starting from the first clip.
        first, _ = next(loader.feed_forward_generator(1, 128))
        self.assertTrue(torch.all(first[:, -1] < 100))

    @unittest.skipUnless(os.environ.get("HUNTER130_TEST_GPU") == "1" and torch.cuda.is_available(),
                         "Set HUNTER130_TEST_GPU=1 to check GPU observations, actions and resets")
    def test_gpu_amp_observation_and_action_mapping(self):
        from mjlab.envs import ManagerBasedRlEnv
        from src.tasks.amp_loco.config.hunter130.env_cfgs import hunter130_amp_flat_env_cfg

        motion = self._motion()
        cfg = hunter130_amp_flat_env_cfg()
        cfg.scene.num_envs = 8
        cfg.episode_length_s = 0.08
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
        try:
            env.reset()
            robot = env.scene["robot"]
            ids = torch.arange(env.num_envs, device=env.device)
            def batch(array):
                return torch.as_tensor(array, device=env.device).float().expand(env.num_envs, -1).clone()

            root_pos = batch(motion["body_pos_w"][0, 0]) + env.scene.env_origins
            root_quat = batch(motion["body_quat_w"][0, 0])
            robot.write_root_link_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids=ids)
            robot.write_root_link_velocity_to_sim(batch(np.concatenate((motion["body_lin_vel_w"][0, 0],
                                                                        motion["body_ang_vel_w"][0, 0]))), env_ids=ids)
            robot.write_joint_state_to_sim(batch(motion["joint_pos"][0]), batch(motion["joint_vel"][0]), env_ids=ids)
            env.sim.forward()
            env.scene.update(env.step_dt)
            observation = env.observation_manager.compute(update_history=True)["amp"]
            with tempfile.TemporaryDirectory() as tmp:
                np.savez(Path(tmp) / "motion.npz", **motion)
                loader = AMPLoader(tmp, AMP_BODY_NAMES, AMP_ANCHOR_NAME, self.bodies,
                                   str(env.device), self.joints, 50)
            torch.testing.assert_close(observation, loader._features[0].expand_as(observation), atol=3e-4, rtol=3e-4)
            action = env.action_manager.get_term("joint_pos")
            scales = torch.tensor([get_action_scale()[n] for n in action.target_names], device=env.device)
            for value in (0.1, 100., -100.):
                raw = torch.full((env.num_envs, 25), value, device=env.device)
                action.process_actions(raw)
                expected = raw * scales + robot.data.default_joint_pos[:, action.target_ids]
                torch.testing.assert_close(action._processed_actions, expected)
                torch.testing.assert_close(action.raw_action, raw)
            saw_timeout = False
            for _ in range(12):
                observations, reward, _, truncated, _ = env.step(torch.zeros(env.num_envs, 25, device=env.device))
                saw_timeout |= bool(truncated.any())
                self.assertTrue(torch.isfinite(reward).all())
                self.assertTrue(all(torch.isfinite(v).all() for v in observations.values()))
            self.assertTrue(saw_timeout)
            self.assertEqual(int(env.termination_manager._delay_env_mask.sum()), 3)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
