"""Position PD with drive overspeed protection from joint maxSpd (rad/s)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.actuator import Actuator, ActuatorCmd
from mjlab.utils.spec import create_position_actuator

if TYPE_CHECKING:
    from mjlab.entity import Entity


@dataclass(kw_only=True)
class VelocityLimitedPositionActuatorCfg(BuiltinPositionActuatorCfg):
    velocity_limit: float
    """Symmetric drive speed threshold, in rad/s for Hunter180 joints."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.velocity_limit) or self.velocity_limit <= 0:
            raise ValueError("velocity_limit must be finite and positive")
        if not math.isfinite(self.stiffness) or self.stiffness <= 0:
            raise ValueError("stiffness must be finite and positive")

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> VelocityLimitedPositionActuator:
        return VelocityLimitedPositionActuator(self, entity, target_ids, target_names)


class VelocityLimitedPositionActuator(Actuator[VelocityLimitedPositionActuatorCfg]):
    """Keep native position PD and suppress torque that drives further overspeed.

    Inheriting Actuator directly keeps compute() out of mjlab's built-in actuator
    fast path, so protection is recomputed from joint state each physics step.
    This does not hard-clamp qvel: external forces can still cause overspeed,
    and the original braking torque is always allowed.
    """

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        for target_name in target_names:
            actuator = create_position_actuator(
                spec,
                target_name,
                stiffness=self.cfg.stiffness,
                damping=self.cfg.damping,
                effort_limit=self.cfg.effort_limit,
                armature=self.cfg.armature,
                frictionloss=self.cfg.frictionloss,
                transmission_type=self.cfg.transmission_type,
            )
            self._mjs_actuators.append(actuator)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        # Native PD: torque = Kp * (ctrl - pos) - Kd * vel. At the speed
        # threshold, clip ctrl at the zero-torque target in the drive direction.
        zero_torque_target = cmd.pos + (self.cfg.damping / self.cfg.stiffness) * cmd.vel
        target = torch.where(
            cmd.vel >= self.cfg.velocity_limit,
            torch.minimum(cmd.position_target, zero_torque_target),
            cmd.position_target,
        )
        return torch.where(
            cmd.vel <= -self.cfg.velocity_limit,
            torch.maximum(target, zero_torque_target),
            target,
        )
