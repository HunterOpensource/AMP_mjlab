from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_loco.rl import AMPOnPolicyRunner
from .env_cfgs import hunter130_amp_flat_env_cfg, hunter130_amp_rough_env_cfg
from .rl_cfg import hunter130_amp_ppo_runner_cfg

for terrain, factory in (("Flat", hunter130_amp_flat_env_cfg), ("Rough", hunter130_amp_rough_env_cfg)):
    register_mjlab_task(
        task_id=f"Hunter130-AMP-{terrain}",
        env_cfg=factory(),
        play_env_cfg=factory(play=True),
        rl_cfg=hunter130_amp_ppo_runner_cfg(),
        runner_cls=AMPOnPolicyRunner,
    )
