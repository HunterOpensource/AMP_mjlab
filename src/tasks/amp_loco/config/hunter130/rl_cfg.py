"""Self-contained AMP/PPO configuration for Hunter130's 25-joint policy."""

from dataclasses import dataclass, field

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .constants import AMP_ANCHOR_NAME, AMP_BODY_NAMES, MOTION_DIR


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


def hunter130_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    return RslRlAmpRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            class_name="AMPPPO",
        ),
        experiment_name="hunter130_amp_baseline",
        logger="tensorboard",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=100001,
        amp_reward_coef=0.1,
        amp_motion_files=str(MOTION_DIR),
        amp_num_preload_transitions=200000,
        amp_task_reward_lerp=0.75,
        amp_discr_hidden_dims=[1024, 512, 256],
        min_normalized_std=[0.05] * 25,
        amp_body_names=AMP_BODY_NAMES,
        amp_anchor_name=AMP_ANCHOR_NAME,
    )
