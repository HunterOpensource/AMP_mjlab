"""Hunter130's 25-joint model and named motor parameters."""

import json
import math
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

ASSET_ROOT = Path(__file__).resolve().parent
HUNTER130_XML = ASSET_ROOT / "assets" / "hunter130.xml"


def load_motor_parameters() -> dict[str, dict[str, float]]:
    motors = json.loads((ASSET_ROOT / "assets" / "motors.json").read_text())
    for name, motor in motors.items():
        for key in ("stiffness", "damping", "armature", "effort", "action_scale"):
            value = motor[key]
            valid = value >= 0 if key in ("damping", "armature") else value > 0
            if not math.isfinite(value) or not valid:
                raise ValueError(f"motors.json: invalid {name}.{key}: {value}")
    return motors


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(HUNTER130_XML))
    motors = load_motor_parameters()
    names = [j.name for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
    if len(names) != 25 or set(names) != set(motors):
        raise ValueError("motors.json must match all 25 physical Hunter130 joints")
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for name in names:
        joint = spec.joint(name)
        effort = motors[name]["effort"]
        joint.actfrcrange = [-effort, effort]
        joint.actfrclimited = True
    return spec


def get_action_scale() -> dict[str, float]:
    return {name: motor["action_scale"] for name, motor in load_motor_parameters().items()}


def get_hunter130_robot_cfg() -> EntityCfg:
    motors = load_motor_parameters()
    # XML joint order is also the policy action order and NPZ joint order.
    spec = get_spec()
    names = [j.name for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, 0.68),
            joint_pos={
                ".*_hip_pitch_joint": -0.1,
                ".*_knee_joint": 0.3,
                ".*_ankle_pitch_joint": -0.2,
                ".*_shoulder_pitch_joint": 0.2,
                "left_shoulder_roll_joint": 0.18,
                "right_shoulder_roll_joint": -0.18,
                ".*_elbow_joint": 0.6,
            },
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(name,),
                    stiffness=motors[name]["stiffness"],
                    damping=motors[name]["damping"],
                    armature=motors[name]["armature"],
                    effort_limit=motors[name]["effort"],
                )
                for name in names
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
        collisions=(CollisionCfg(
            geom_names_expr=(".*_collision",),
            contype=1,
            conaffinity=1,
            condim={".*_foot_collision": 3, ".*_collision": 1},
            priority={".*_foot_collision": 1, ".*": 0},
            friction={".*_foot_collision": (0.6,)},
        ),),
    )
