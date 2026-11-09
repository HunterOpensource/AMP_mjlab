from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .motion_validation import validate_motion


class AMPLoader:
    def __init__(self, motion_file: str, body_names: Sequence[str],
                 anchor_name: str, all_body_names: Sequence[str],
                 device: str = "cuda:0", joint_names: Sequence[str] | None = None,
                 expected_fps: float | None = None):
        """Load body-local AMP features; NPZ body order must match the robot.

        Files with name metadata are checked explicitly. Older G1 files without
        metadata remain supported, with shape and sampling-rate validation.
        """
        import mjlab.utils.lab_api.math as math_utils

        path = Path(motion_file)
        motion_files = [path] if path.is_file() else sorted(path.rglob("*.npz"))
        if not motion_files:
            raise FileNotFoundError(f"No AMP .npz files in {path}. Prepare motion data first.")
        all_names = list(all_body_names)
        self._body_indexes = [all_names.index(n) for n in body_names]
        self._anchor_indexes = all_names.index(anchor_name)
        self._num_bodies = len(body_names)
        self.motion_names = [p.stem for p in motion_files]
        self._body_pos_b_list = []
        self._body_quat_b_list = []
        self._body_ori_b_list = []
        self._body_lin_vel_b_list = []
        self._body_ang_vel_b_list = []
        features = []
        for index, motion_path in enumerate(motion_files):
            with np.load(motion_path) as archive:
                data = dict(archive)
            fps = validate_motion(data, motion_path, all_body_names, joint_names,
                                  expected_fps if index == 0 else self.fps)
            if index == 0:
                self.fps = fps
            # Only transfer the tracked bodies and anchor, not all joints/links.
            def tensor(key, indexes):
                return torch.as_tensor(data[key][:, indexes], dtype=torch.float32, device=device)

            pos_w = tensor("body_pos_w", self._body_indexes)
            quat_w = tensor("body_quat_w", self._body_indexes)
            anchor_pos = tensor("body_pos_w", [self._anchor_indexes]).expand_as(pos_w)
            anchor_quat = tensor("body_quat_w", [self._anchor_indexes]).expand_as(quat_w)
            pos_b, quat_b = math_utils.subtract_frame_transforms(
                anchor_pos, anchor_quat, pos_w, quat_w)
            ori_b = math_utils.matrix_from_quat(quat_b)[..., :, :2].flatten(-2)
            lin_b = math_utils.quat_apply_inverse(quat_w, tensor("body_lin_vel_w", self._body_indexes))
            ang_b = math_utils.quat_apply_inverse(quat_w, tensor("body_ang_vel_w", self._body_indexes))
            self._body_pos_b_list.append(pos_b)
            self._body_quat_b_list.append(quat_b)
            self._body_ori_b_list.append(ori_b)
            self._body_lin_vel_b_list.append(lin_b)
            self._body_ang_vel_b_list.append(ang_b)
            features.append(torch.cat([v.flatten(1) for v in (pos_b, ori_b, lin_b, ang_b)], dim=-1))
            print(f"[AMP] {index + 1}/{len(motion_files)} {motion_path.stem}: {pos_b.shape[0]} frames")

        self._features = torch.cat(features)
        lengths = torch.tensor([f.shape[0] for f in features], device=device)
        self._starts = lengths.cumsum(0) - lengths
        self.time_step_total = self._body_pos_b_list[0].shape[0]
        self.motion_total_time = self.time_step_total / self.fps
        self._body_pos_b = self._body_pos_b_list[0]
        self._body_quat_b = self._body_quat_b_list[0]
        self._body_ori_b = self._body_ori_b_list[0]
        self._body_lin_vel_b = self._body_lin_vel_b_list[0]
        self._body_ang_vel_b = self._body_ang_vel_b_list[0]

    @property
    def observation_dim(self) -> int:
        return 15 * self._num_bodies

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Match the baseline: cycle clips per batch and repeat the final frame."""
        device = self._features.device
        for batch_idx in range(num_mini_batch):
            clip = batch_idx % len(self.motion_names)
            num_frames = self._body_pos_b_list[clip].shape[0]
            offsets = torch.randint(0, num_frames, (mini_batch_size,), device=device)
            next_offsets = (offsets + 1).clamp(max=num_frames - 1)
            start = self._starts[clip]
            yield self._features[start + offsets], self._features[start + next_offsets]
