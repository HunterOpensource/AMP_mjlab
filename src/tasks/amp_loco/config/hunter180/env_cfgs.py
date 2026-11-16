"""Hunter180 AMP locomotion and fall-recovery environments (50 Hz)."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg, mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.terrains import HfPyramidSlopedTerrainCfg, HfRandomUniformTerrainCfg, HfWaveTerrainCfg

from src.assets.robots.hunter180 import get_action_scale, get_hunter180_robot_cfg
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg
from .constants import AMP_ANCHOR_NAME, AMP_BODY_NAMES, MOTION_DIR


def hunter180_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_amp_env_cfg()
    cfg.scene.entities = {"robot": get_hunter180_robot_cfg()}
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 256
    cfg.sim.nconmax = 64
    cfg.sim.njmax = 640

    for sensor in cfg.scene.sensors or ():
        if isinstance(sensor, RayCastSensorCfg):
            sensor.frame.name = "pelvis"
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1, history_length=4,
        ),
    )
    cfg.actions["joint_pos"] = JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",),
        scale=get_action_scale(), use_default_offset=True,
    )
    cfg.viewer.body_name = AMP_ANCHOR_NAME
    cfg.commands["twist"].viz.z_offset = 1.0
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision", "right_foot_collision",
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = (AMP_ANCHOR_NAME,)
    cfg.events["init_motion_loader"].params.update(
        motion_dir=str(MOTION_DIR / "WalkandRun"),
        recovery_dir=str(MOTION_DIR / "Recovery"),
        delay_reset_env_ratio=0.4,
        max_delay_steps=250,
    )
    cfg.events["reset_from_motion"].params["motion_dir"] = str(MOTION_DIR / "WalkandRun")

    for name in ("track_anchor_linear_velocity", "track_anchor_angular_velocity"):
        cfg.rewards[name].params["anchor_cfg"].body_names = (AMP_ANCHOR_NAME,)
    cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = ("pelvis",)
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = ("left_foot", "right_foot")
    # Hunter180 has a roughly 0.94 m standing pelvis height.
    cfg.terminations["bad_base_height"].params["minimum_height"] = 0.5
    for group in ("critic", "amp"):
        for term in cfg.observations[group].terms.values():
            if "body_cfg" in term.params:
                term.params["body_cfg"].body_names = AMP_BODY_NAMES
                term.params["body_cfg"].preserve_order = True
                term.params["anchor_cfg"].body_names = (AMP_ANCHOR_NAME,)

    assert cfg.scene.terrain is not None
    assert cfg.scene.terrain.terrain_generator is not None
    # A fallen torso can intersect >50 triangles on a 0.1 m heightfield,
    # exceeding MuJoCo Warp's fixed per-pair contact limit. A 0.2 m grid
    # bounds this footprint while retaining slopes, waves and roughness.
    # Copy nested terrain configs before customizing Hunter180 terrain.
    cfg.scene.terrain.terrain_generator = deepcopy(cfg.scene.terrain.terrain_generator)
    for terrain in cfg.scene.terrain.terrain_generator.sub_terrains.values():
        if isinstance(terrain, (HfPyramidSlopedTerrainCfg, HfRandomUniformTerrainCfg, HfWaveTerrainCfg)):
            terrain.horizontal_scale = 0.2
    cfg.scene.terrain.terrain_generator.curriculum = True
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(func=envs_mdp.randomize_terrain, mode="reset", params={})
        cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0
        generator = cfg.scene.terrain.terrain_generator
        generator.curriculum = False
        generator.num_cols = generator.num_rows = 5
        generator.border_width = 10.0
    return cfg


def hunter180_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = hunter180_amp_rough_env_cfg(play=play)
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan")
    for group in ("actor", "critic"):
        del cfg.observations[group].terms["height_scan"]
    cfg.curriculum.pop("terrain_levels", None)
    return cfg
