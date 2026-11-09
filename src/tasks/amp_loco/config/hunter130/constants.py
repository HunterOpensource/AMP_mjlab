"""Shared reference-data and discriminator layout for Hunter130."""

from src import SRC_PATH

# Retargeted copies of this repository's original 17 locomotion + 1 recovery clips.
MOTION_DIR = SRC_PATH / "assets" / "motions" / "hunter130" / "amp_baseline"
AMP_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_roll_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_roll_link",
)
AMP_ANCHOR_NAME = "torso_link"
