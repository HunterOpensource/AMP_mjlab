"""Validate reference motion layout before using it for resets or AMP rewards."""

import numpy as np


def validate_motion(data, path, body_names=None, joint_names=None, expected_fps=None,
                    require_metadata=False):
    def fail(message):
        raise ValueError(f"{path}: {message}")

    keys = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w")
    for key in keys:
        if key not in data:
            fail(f"missing {key}")
    fps_array = np.asarray(data["fps"])
    if fps_array.size != 1:
        fail("fps must be a scalar or single-element array")
    fps = float(fps_array.item())
    if not np.isfinite(fps) or fps <= 0:
        fail("fps must be finite and positive")
    if expected_fps is not None and not np.isclose(fps, expected_fps):
        fail(f"expected {expected_fps:g} Hz, got {fps:g} Hz; resample the motion first")
    if data["joint_pos"].ndim != 2 or data["body_pos_w"].ndim != 3:
        fail("joint_pos/body_pos_w must have shape (T,J)/(T,B,3)")
    frames, joints = data["joint_pos"].shape
    bodies = data["body_pos_w"].shape[1]
    if frames < 2:
        fail("at least two frames are needed for AMP transitions")
    shapes = {
        "joint_pos": (frames, joints), "joint_vel": (frames, joints),
        "body_pos_w": (frames, bodies, 3), "body_quat_w": (frames, bodies, 4),
        "body_lin_vel_w": (frames, bodies, 3), "body_ang_vel_w": (frames, bodies, 3),
    }
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            fail(f"{key} must be finite with shape {shape}, got {data[key].shape}")
    if not np.allclose(np.linalg.norm(data["body_quat_w"], axis=-1), 1, atol=1e-3):
        fail("body_quat_w must contain unit quaternions (wxyz)")
    for key, expected, count in (("body_names", body_names, bodies),
                                  ("joint_names", joint_names, joints)):
        if expected is not None and len(expected) != count:
            fail(f"{key} count: expected {len(expected)}, got {count}")
        if key not in data:
            if require_metadata:
                fail(f"missing {key} metadata")
        else:
            names = data[key].tolist()
            if len(names) != count or len(set(names)) != count:
                fail(f"invalid {key} metadata")
            if expected is not None and names != list(expected):
                fail(f"{key} order differs from the robot model")
    return fps
