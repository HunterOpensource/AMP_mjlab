"""AMP runner parameters shared by the robot task configurations."""

from dataclasses import dataclass, field

from mjlab.rl import RslRlOnPolicyRunnerCfg


@dataclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Extended runner config with AMP-specific parameters."""

    amp_reward_coef: float = 0.1
    amp_motion_files: str = ""
    amp_num_preload_transitions: int = 200000
    amp_task_reward_lerp: float = 0.75
    amp_discr_hidden_dims: list[int] = field(default_factory=lambda: [1024, 512, 256])
    min_normalized_std: list[float] = field(default_factory=lambda: [0.05] * 25)
    amp_body_names: tuple = ()
    amp_anchor_name: str = ""
