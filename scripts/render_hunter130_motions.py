"""Audit and render each Hunter130 reference clip as a two-view, real-time MP4."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

from rsl_rl.utils.motion_validation import validate_motion

ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = ROOT / "src/assets/robots/hunter130/assets/hunter130.xml"
MOTIONS = ROOT / "src/assets/motions/hunter130/amp_baseline"
FRAME_SIZE = (1280, 720)
PANEL_SIZE = (640, 608)


def render_model():
    """Add a visual floor/lights in memory; do not change the training model."""
    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((ROBOT_XML.parent / compiler.get("meshdir")).resolve()))
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="review_grid", type="2d", builtin="checker",
                  rgb1="0.24 0.28 0.33", rgb2="0.18 0.21 0.25", width="512", height="512")
    ET.SubElement(asset, "material", name="review_floor", texture="review_grid", texrepeat="4 4",
                  texuniform="true", reflectance="0.05")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="review_ground", type="plane", size="0 0 0.1",
                  material="review_floor", contype="0", conaffinity="0", group="0")
    ET.SubElement(world, "light", name="review_key", pos="-1 -3 5", dir="0 0 -1",
                  directional="true", diffuse="0.7 0.7 0.7", ambient="0.25 0.25 0.25")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(PANEL_SIZE[0]), offheight=str(PANEL_SIZE[1]))
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    ET.SubElement(visual, "headlight", diffuse="0.4 0.4 0.4", ambient="0.3 0.3 0.3")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    # Foot colors make left/right motion identifiable from either view.
    model.geom_rgba[model.geom("left_ankle_roll_link_visual").id] = [0.20, 0.60, 0.95, 1]
    model.geom_rgba[model.geom("right_ankle_roll_link_visual").id] = [0.95, 0.48, 0.18, 1]
    return model


class Audit:
    def __init__(self, model):
        self.model = model
        self.geoms = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
        self.meshes = {}
        for g in self.geoms:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(model.geom_dataid[g])
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[start:start + count].astype(float)
                self.meshes[mesh] = vertices[ConvexHull(vertices).vertices]

    def clearance(self, data):
        lowest = np.inf
        for g in self.geoms:
            direction = data.geom_xmat[g].reshape(3, 3)[2]
            kind, size = self.model.geom_type[g], self.model.geom_size[g]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                extent = np.abs(direction) @ size
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent = size[0] + size[1] * abs(direction[2])
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent = size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                extent = -np.min(self.meshes[int(self.model.geom_dataid[g])] @ direction)
            else:
                raise ValueError(f"Unsupported collision shape: {self.model.geom(g).name}")
            lowest = min(lowest, float(data.geom_xpos[g, 2] - extent))
        penetration = max([0.] + [-float(c.dist) for c in data.contact])
        return lowest, penetration


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def record_clip(path, output, motion_dir, model, renderer, audit, manifest_entry, max_frames=None):
    relative = path.relative_to(motion_dir)
    with np.load(path, allow_pickle=False) as archive:
        motion = dict(archive)
    joints = [model.joint(i).name for i in range(1, model.njnt)]
    bodies = [model.body(i).name for i in range(1, model.nbody)]
    fps = validate_motion(motion, path, bodies, joints, 50, True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest_entry and digest != manifest_entry["sha256"]:
        raise ValueError(f"Motion checksum mismatch: {path}")
    frames = len(motion["joint_pos"])
    if max_frames is not None:
        frames = min(frames, max_frames)
    target = output / relative.with_suffix(".mp4")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".partial.mp4")
    data = mujoco.MjData(model)
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    options.sitegroup[:] = 0
    cameras = []
    for angle in (135, 45):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth, camera.elevation, camera.distance = angle, -18, 2.6
        cameras.append(camera)
    header = Image.new("RGB", FRAME_SIZE, (20, 26, 35))
    draw = ImageDraw.Draw(header)
    draw.text((18, 8), f"Hunter130 / {relative.parent.name} / {path.stem}", font=font(21), fill="white")
    draw.text((18, 40), "Reference motion | 1x speed | 50 Hz | left foot: blue / right foot: orange",
              font=font(17), fill=(190, 205, 220))
    small_font = font(17)
    writer = imageio_ffmpeg.write_frames(str(partial), FRAME_SIZE, fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=7, macro_block_size=1,
                                         ffmpeg_log_level="error",
                                         output_params=["-preset", "veryfast", "-threads", "2", "-movflags", "+faststart"])
    writer.send(None)
    limits = model.jnt_range[1:]
    violation = np.maximum(limits[:, 0] - motion["joint_pos"], motion["joint_pos"] - limits[:, 1]).max()
    report = {"file": str(relative), "sha256": digest, "frames": frames, "fps": fps,
              "video": str(target.relative_to(output)), "video_duration_s": frames / fps,
              "min_collision_height_m": float("inf"), "max_self_penetration_m": 0.,
              "max_body_position_error_m": 0., "max_body_quaternion_error": 0.,
              "max_body_velocity_error": 0., "max_joint_limit_violation_rad": max(0., float(violation)),
              "max_joint_speed_rad_s": float(np.abs(motion["joint_vel"]).max())}
    preview_frames = sorted(set(np.linspace(0, frames - 1, 8, dtype=int).tolist()))
    previews = []
    velocity = np.zeros(6)
    started = time.monotonic()
    try:
        for frame in range(frames):
            data.qpos[:3] = motion["body_pos_w"][frame, 0]
            data.qpos[3:7] = motion["body_quat_w"][frame, 0]
            data.qpos[7:] = motion["joint_pos"][frame]
            data.qvel[:3] = motion["body_lin_vel_w"][frame, 0]
            conjugate = data.qpos[3:7].copy()
            conjugate[1:] *= -1
            mujoco.mju_rotVecQuat(data.qvel[3:6], motion["body_ang_vel_w"][frame, 0], conjugate)
            data.qvel[6:] = motion["joint_vel"][frame]
            mujoco.mj_forward(model, data)
            clearance, penetration = audit.clearance(data)
            report["min_collision_height_m"] = min(report["min_collision_height_m"], clearance)
            if penetration > report["max_self_penetration_m"]:
                report["max_self_penetration_m"] = penetration
                report["max_penetration_frame"] = frame
            position_error = np.abs(data.xpos[1:] - motion["body_pos_w"][frame]).max()
            report["max_body_position_error_m"] = max(report["max_body_position_error_m"], float(position_error))
            quats = motion["body_quat_w"][frame]
            signs = np.where((quats * data.xquat[1:]).sum(axis=-1, keepdims=True) < 0, -1, 1)
            quat_error = np.abs(data.xquat[1:] - signs * quats).max()
            report["max_body_quaternion_error"] = max(report["max_body_quaternion_error"], float(quat_error))
            for body in range(1, model.nbody):
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
                error = max(np.abs(velocity[:3] - motion["body_ang_vel_w"][frame, body - 1]).max(),
                            np.abs(velocity[3:] - motion["body_lin_vel_w"][frame, body - 1]).max())
                report["max_body_velocity_error"] = max(report["max_body_velocity_error"], float(error))
            canvas = header.copy()
            for index, camera in enumerate(cameras):
                camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.60]
                renderer.update_scene(data, camera=camera, scene_option=options)
                canvas.paste(Image.fromarray(renderer.render()), (index * PANEL_SIZE[0], 72))
            overlay = ImageDraw.Draw(canvas)
            overlay.text((16, 82), "World view A", font=small_font, fill="white", stroke_width=1, stroke_fill="black")
            overlay.text((656, 82), "World view B (+90 deg)", font=small_font, fill="white", stroke_width=1, stroke_fill="black")
            overlay.text((16, 690), f"{frame / fps:6.2f} s | frame {frame + 1}/{frames} | "
                         f"clearance {clearance * 1000:5.1f} mm | self-overlap {penetration * 1000:4.1f} mm",
                         font=small_font, fill=(245, 188, 93) if penetration > .003 else "white")
            writer.send(np.asarray(canvas))
            if frame in preview_frames:
                preview = canvas.resize((640, 360))
                previews.append(preview)
            if frame == frames // 2:
                canvas.save(target.with_suffix(".jpg"), quality=90)
            if frame and frame % 250 == 0:
                print(f"  {relative}: {frame}/{frames} frames", flush=True)
    finally:
        writer.close()
    partial.replace(target)
    strip = Image.new("RGB", (1280, 360 * ((len(previews) + 1) // 2)), (20, 26, 35))
    for index, preview in enumerate(previews):
        strip.paste(preview, ((index % 2) * 640, (index // 2) * 360))
    strip.save(target.with_name(target.stem + "_timeline.jpg"), quality=90)
    actual_frames, duration = imageio_ffmpeg.count_frames_and_secs(str(target))
    if actual_frames != frames or abs(duration - frames / fps) > .04:
        raise ValueError(f"Video validation failed: {target}: {actual_frames} frames / {duration} s")
    report["checks"] = {
        "joint_limits": report["max_joint_limit_violation_rad"] <= 1e-6,
        "forward_kinematics_positions": report["max_body_position_error_m"] <= 5e-6,
        "forward_kinematics_rotations": report["max_body_quaternion_error"] <= 5e-6,
        "link_velocities": report["max_body_velocity_error"] <= 1e-4,
        "above_ground": report["min_collision_height_m"] >= -1e-6,
    }
    if "max_penetration_frame" in report:
        report["max_penetration_time_s"] = report["max_penetration_frame"] / fps
    report["render_seconds"] = time.monotonic() - started
    return report


def write_gallery(output, reports):
    cards = []
    for item in reports:
        video = html.escape(item["video"], quote=True)
        poster = html.escape(str(Path(item["video"]).with_suffix(".jpg")), quote=True)
        timeline = html.escape(str(Path(item["video"]).with_suffix("")) + "_timeline.jpg", quote=True)
        overlap_link = ""
        if "max_penetration_time_s" in item:
            timestamp = item["max_penetration_time_s"]
            overlap_link = (f' · <a href="{video}#t={timestamp:.2f}">'
                            f'最大自穿透位置：{timestamp:.2f} 秒</a>')
        cards.append(f'<article><h2>{html.escape(item["file"])}</h2>'
                     f'<video controls preload="none" poster="{poster}" src="{video}"></video>'
                     f'<p>{item["frames"]} 帧 · {item["video_duration_s"]:.2f} 秒 · 50 Hz · '
                     f'最大自穿透 {item["max_self_penetration_m"] * 1000:.2f} mm</p>'
                     f'<p><a href="{video}">打开视频</a> · <a href="{timeline}">查看 8 帧时间线</a>'
                     f'{overlap_link}</p></article>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Hunter130 重定向动作检查</title><style>'
            'body{margin:32px auto;max-width:1500px;padding:0 20px;font-family:system-ui;background:#101720;color:#e9eef5}'
            'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:24px}'
            'article{background:#1c2734;padding:18px;border-radius:12px}h2{font-size:17px;overflow-wrap:anywhere}'
            'video{width:100%}a{color:#85c8ff}p{line-height:1.7}'
            '@media(max-width:600px){main{grid-template-columns:1fr}}</style>'
            '<h1>Hunter130 重定向动作检查</h1>'
            '<p>逐帧回放 NPZ 参考姿态：原速、双视角，蓝色左脚、橙色右脚。相机跟随根部平移，视角方向保持固定。'
            '此处不运行策略或动力学步进；视频用于检查参考动作，不能证明机器人能动态执行。</p>'
            f'<p>{len(reports)} 段 · {sum(c["frames"] for c in reports):,} 帧 · '
            f'总时长 {sum(c["video_duration_s"] for c in reports):.2f} 秒 · '
            '<a href="audit.json">完整检查数据</a></p><main>' + ''.join(cards) + '</main></html>')
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "audit.json").write_text(json.dumps({"model_sha256": hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest(),
                                                  "clips": reports}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, default=MOTIONS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "logs/motion_review" / datetime.now().strftime("hunter130_%Y%m%d_%H%M%S"))
    parser.add_argument("--clips", nargs="+", help="Optional clip stems; default renders all NPZ files")
    parser.add_argument("--max-frames", type=int, help="Optional preview limit; omit for complete clips")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    files = sorted(args.motion_dir.rglob("*.npz"))
    if args.clips:
        files = [p for p in files if p.stem in args.clips]
        if {p.stem for p in files} != set(args.clips):
            parser.error("Some requested clips are missing")
    if not files:
        parser.error("No NPZ motions found")
    manifest_path = args.motion_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries = {c["file"]: c for c in manifest.get("clips", [])}
    args.output.mkdir(parents=True, exist_ok=True)
    model = render_model()
    audit = Audit(model)
    reports = []
    print(f"Output: {args.output.resolve()}", flush=True)
    with mujoco.Renderer(model, height=PANEL_SIZE[1], width=PANEL_SIZE[0]) as renderer:
        for index, path in enumerate(files):
            print(f"[{index + 1}/{len(files)}] {path.relative_to(args.motion_dir)}", flush=True)
            report = record_clip(path, args.output, args.motion_dir, model, renderer, audit,
                                 entries.get(str(path.relative_to(args.motion_dir))), args.max_frames)
            reports.append(report)
            write_gallery(args.output, reports)
            print(f"  Saved {report['video']}: {report['frames']} frames, "
                  f"self-overlap {report['max_self_penetration_m'] * 1000:.2f} mm", flush=True)
    print(f"Completed {len(reports)} videos: {args.output.resolve() / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
