"""Hunter180 motion paths and discriminator layout (23 joints, no wrists)."""

from src import SRC_PATH

# All 17 locomotion + 1 recovery clips retargeted from the bundled Hunter130 baseline.
MOTION_DIR = SRC_PATH / "assets/motions/hunter180/amp_baseline"
AMP_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "left_shoulder_roll_link", "left_elbow_link",
    "right_shoulder_roll_link", "right_elbow_link",
)
AMP_ANCHOR_NAME = "torso_link"
