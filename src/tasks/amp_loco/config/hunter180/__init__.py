from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_loco.rl import AMPOnPolicyRunner
from .env_cfgs import hunter180_amp_flat_env_cfg, hunter180_amp_rough_env_cfg
from .rl_cfg import hunter180_amp_ppo_runner_cfg

for terrain, factory in (("Flat", hunter180_amp_flat_env_cfg), ("Rough", hunter180_amp_rough_env_cfg)):
    register_mjlab_task(
        task_id=f"Hunter180-AMP-{terrain}",
        env_cfg=factory(),
        play_env_cfg=factory(play=True),
        rl_cfg=hunter180_amp_ppo_runner_cfg(),
        runner_cls=AMPOnPolicyRunner,
    )
