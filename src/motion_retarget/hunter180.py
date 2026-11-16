"""Hunter130 AMP -> Hunter180 with bounded whole-body inverse kinematics."""

import numpy as np

from src import SRC_PATH
from .hunter130 import HUNTER_XML, RetargetCfg, Retargeter

HUNTER180_XML = SRC_PATH / "assets/robots/hunter180/assets/hunter180.xml"
SOURCE_MOTIONS = SRC_PATH / "assets/motions/hunter130/amp_baseline"


class Hunter180Retargeter(Retargeter):
    def __init__(self, cfg: RetargetCfg | None = None):
        # Hunter180 ends at the elbow housing: no wrist/hand body is present.
        # Use the housing's +x endpoint (75.2 mm), not an unsupported virtual hand.
        source_points = {side: (f"{side}_wrist_roll_link", (0.20, 0, 0)) for side in ("left", "right")}
        target_points = {side: (f"{side}_elbow_link", (0.0752, 0, 0)) for side in ("left", "right")}
        super().__init__(
            cfg or RetargetCfg(root_adjustment=0.15, max_nfev=40),
            source_xml=HUNTER_XML, target_xml=HUNTER180_XML,
            source_palm_points=source_points, target_palm_points=target_points,
            joint_position_offsets={f"{side}_elbow_joint": -np.pi / 2 for side in ("left", "right")},
        )
        # Both hand endpoints use their local +x direction, whereas the target
        # elbow has an additional fixed 90-degree rotation in the URDF.
        for i, point in enumerate(self.points):
            if point[-1] == "palm":
                self.rotation_corrections[i] = np.eye(3)
