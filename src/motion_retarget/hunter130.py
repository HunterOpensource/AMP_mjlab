"""G1 AMP NPZ -> Hunter130, using bounded whole-body inverse kinematics.

Timing is unchanged. Root/foot trajectories use the neutral sole-to-pelvis height
ratio. Arm targets follow source bone directions with target bone lengths. The
root orientation is preserved; root translation and all 25 joints are optimized.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from src import SRC_PATH

G1_XML = SRC_PATH / "assets/robots/unitree_g1/xmls/g1.xml"
HUNTER_XML = SRC_PATH / "assets/robots/hunter130/assets/hunter130.xml"


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[..., [1, 2, 3, 0]])


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


@dataclass
class RetargetCfg:
    soft_limit_factor: float = 0.9
    max_nfev: int = 30
    root_adjustment: float = 0.08
    contact_height: float = 0.035
    contact_speed: float = 0.4
    floor_clearance: float = 0.002
    max_joint_speed: float = 20.0
    collision_margin: float = 0.003
    # Virtual Hunter palm measured from the wrist-roll origin, along its x axis.
    palm_offset: tuple[float, float, float] = (0.20, 0., 0.)


class Retargeter:
    def __init__(self, cfg: RetargetCfg | None = None):
        self.cfg = cfg or RetargetCfg()
        self.source = mujoco.MjModel.from_xml_path(str(G1_XML))
        self.model = mujoco.MjModel.from_xml_path(str(HUNTER_XML))
        self.sd = mujoco.MjData(self.source)
        self.data = mujoco.MjData(self.model)
        self.joint_names = [self.model.joint(i).name for i in range(1, self.model.njnt)]
        self.body_names = [self.model.body(i).name for i in range(1, self.model.nbody)]
        self.source_joint_names = [self.source.joint(i).name for i in range(1, self.source.njnt)]
        self.source_body_names = [self.source.body(i).name for i in range(1, self.source.nbody)]
        self.source_joint_ids = [self.source_joint_names.index(n) for n in self.joint_names]
        self.columns = np.r_[np.arange(3), np.arange(6, self.model.nv)]
        limits = self.model.jnt_range[1:]
        center = limits.mean(axis=1)
        extent = (limits[:, 1] - limits[:, 0]) * self.cfg.soft_limit_factor / 2
        self.lower, self.upper = center - extent, center + extent
        mujoco.mj_kinematics(self.source, self.sd)
        mujoco.mj_kinematics(self.model, self.data)
        self.points = []
        # (target body ID, local point, source body ID, source local point, role)
        for side in ('left', 'right'):
            fs = self.source.site(f'{side}_foot')
            ft = self.model.site(f'{side}_foot')
            self.points.append((int(ft.bodyid[0]), ft.pos.copy(), int(fs.bodyid[0]), fs.pos.copy(), 'foot'))
        self.points.append((self.model.body('torso_link').id, np.zeros(3),
                            self.source.body('torso_link').id, np.zeros(3), 'torso'))
        for side in ('left', 'right'):
            for link, role in (('knee_link', 'knee'), ('elbow_link', 'elbow')):
                self.points.append((self.model.body(f'{side}_{link}').id, np.zeros(3),
                                    self.source.body(f'{side}_{link}').id, np.zeros(3), role))
            palm = self.source.site(f'{side}_palm')
            self.points.append((self.model.body(f'{side}_wrist_roll_link').id,
                                np.array(self.cfg.palm_offset), int(palm.bodyid[0]), palm.pos.copy(), 'palm'))
        self.rest_source = np.array([self.point(self.sd, sb, sl) for _, _, sb, sl, _ in self.points])
        self.rest_target = np.array([self.point(self.data, tb, tl) for tb, tl, _, _, _ in self.points])
        self.rest_root_source = self.sd.xpos[1].copy()
        self.rest_root_target = self.data.xpos[1].copy()
        source_height = self.rest_root_source[2] - self.rest_source[:2, 2].mean()
        target_height = self.rest_root_target[2] - self.rest_target[:2, 2].mean()
        self.scale = float(target_height / source_height)
        self.rotation_corrections = [self.sd.xmat[sb].reshape(3, 3).T @ self.data.xmat[tb].reshape(3, 3)
                                     for tb, _, sb, _, _ in self.points]
        self.geoms = np.flatnonzero(self.model.geom_contype)
        self.mesh_hulls = {}
        for geom in self.geoms:
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(self.model.geom_dataid[geom])
                adr, num = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
                verts = self.model.mesh_vert[adr:adr+num].astype(float)
                self.mesh_hulls[geom] = verts[ConvexHull(verts).vertices]
        self.jp = np.zeros((3, self.model.nv))
        self.jr = np.zeros((3, self.model.nv))
        excluded = set(self.model.exclude_signature.tolist())
        self.collision_pairs = []
        for i, g1 in enumerate(self.geoms):
            for g2 in self.geoms[i + 1:]:
                b1, b2 = sorted((int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])))
                if b1 == b2 or self.model.body_parentid[b2] == b1 or (b1 << 16) + b2 in excluded:
                    continue
                self.collision_pairs.append((g1, g2))
        self.collision_pairs = np.array(self.collision_pairs)
        self.pair_radius = self.model.geom_rbound[self.collision_pairs].sum(axis=1) + self.cfg.collision_margin
        self.j_other = np.zeros_like(self.jp)
        self.fromto = np.zeros(6)

    @staticmethod
    def point(data, body, local):
        return data.xpos[body] + data.xmat[body].reshape(3, 3) @ local

    def floor_points(self):
        """Exact downward support points of the model's convex collision geoms."""
        result = []
        for g in self.geoms:
            mat = self.data.geom_xmat[g].reshape(3, 3)
            size = self.model.geom_size[g]
            kind = self.model.geom_type[g]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                local = -np.sign(mat[2]) * size
                p = self.data.geom_xpos[g] + mat @ local
            elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE):
                p = self.data.geom_xpos[g].copy()
                if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    p -= np.sign(mat[2, 2]) * size[1] * mat[:, 2]
                p[2] -= size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                verts = self.mesh_hulls[g]
                p = self.data.geom_xpos[g] + mat @ verts[np.argmin(verts @ mat[2])]
            else:
                raise ValueError(f'Unsupported collision geom: {self.model.geom(g).name}')
            result.append(p)
        return np.array(result)

    def prepare_targets(self, motion):
        frames = len(motion['joint_pos'])
        source_points = np.empty((frames, len(self.points), 3))
        source_rotations = np.empty((frames, len(self.points), 3, 3))
        for t in range(frames):
            self.sd.qpos[:3] = motion['body_pos_w'][t, 0]
            self.sd.qpos[3:7] = motion['body_quat_w'][t, 0]
            self.sd.qpos[7:] = motion['joint_pos'][t]
            mujoco.mj_kinematics(self.source, self.sd)
            # Legacy G1 NPZ has no body names: verify its body layout against FK.
            if not np.allclose(self.sd.xpos[1:], motion['body_pos_w'][t], atol=2e-5):
                raise ValueError(f'Source NPZ body order/FK mismatch at frame {t}')
            for i, (_, _, sb, sl, _) in enumerate(self.points):
                source_points[t, i] = self.point(self.sd, sb, sl)
                source_rotations[t, i] = self.sd.xmat[sb].reshape(3, 3)
        fps = float(np.asarray(motion['fps']).item())
        root = motion['body_pos_w'][:, 0].astype(float).copy() * self.scale
        root_rot = rotation(motion['body_quat_w'][:, 0]).as_matrix()
        targets = source_points * self.scale
        # Match torso displacement about the neutral waist, without scaling its
        # much taller pelvis-to-waist offset by the G1 offset ratio.
        source_rel = source_points[:, 2] - motion['body_pos_w'][:, 0]
        torso_offset = self.rest_target[2] - self.rest_root_target - self.scale * (self.rest_source[2] - self.rest_root_source)
        targets[:, 2] = root + self.scale * source_rel + np.einsum('tij,j->ti', root_rot, torso_offset)
        for start, side in ((3, 'left'), (6, 'right')):
            shoulder_s = self.source.body(f'{side}_shoulder_roll_link').id
            shoulder_t = self.model.body(f'{side}_shoulder_roll_link').id
            self.sd.qpos[:] = self.source.qpos0
            mujoco.mj_kinematics(self.source, self.sd)
            shoulder_s_rel = self.sd.xpos[shoulder_s] - self.rest_source[2]
            # data remains at the target rest pose here.
            shoulder_t_rel = self.data.xpos[shoulder_t] - self.rest_target[2]
            upper_s = np.linalg.norm(self.rest_source[start + 1] - self.sd.xpos[shoulder_s])
            upper_t = np.linalg.norm(self.rest_target[start + 1] - self.data.xpos[shoulder_t])
            fore_s = np.linalg.norm(self.rest_source[start + 2] - self.rest_source[start + 1])
            fore_t = np.linalg.norm(self.rest_target[start + 2] - self.rest_target[start + 1])
            shoulder_s_w = source_points[:, 2] + np.einsum('tij,j->ti', source_rotations[:, 2], shoulder_s_rel)
            shoulder_t_w = targets[:, 2] + np.einsum('tij,j->ti', source_rotations[:, 2], shoulder_t_rel)
            targets[:, start + 1] = shoulder_t_w + (source_points[:, start + 1] - shoulder_s_w) * (upper_t / upper_s)
            targets[:, start + 2] = targets[:, start + 1] + (source_points[:, start + 2] - source_points[:, start + 1]) * (fore_t / fore_s)
        desired_rot = source_rotations @ np.array(self.rotation_corrections)
        # Correct small source ground offsets once per clip, keeping aerial phases.
        ground = float(np.quantile(targets[:, :2, 2], .05))
        root[:, 2] -= ground
        targets[..., 2] -= ground
        velocity = np.gradient(source_points, 1 / fps, axis=0)
        contact = np.zeros((frames, len(self.points)), dtype=bool)
        for i, (_, _, _, _, role) in enumerate(self.points):
            if role not in ('foot', 'palm'):
                continue
            z = source_points[:, i, 2]
            contact[:, i] = (z < self.cfg.contact_height) & (np.linalg.norm(velocity[:, i], axis=1) < self.cfg.contact_speed)
            # Keep the target stationary during each sustained source contact.
            edges = np.diff(np.r_[False, contact[:, i], False].astype(int))
            for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if end - begin < 3:
                    contact[begin:end, i] = False
                    continue
                targets[begin:end, i, :2] = np.median(targets[begin:end, i, :2], axis=0)
                if role == 'foot':
                    targets[begin:end, i, 2] = self.cfg.floor_clearance
        return root, targets, desired_rot, contact

    def _evaluate(self, x, with_jac=True):
        if self._cached_x is not None and np.array_equal(x, self._cached_x):
            return self._cached
        self.data.qpos[:3] = x[:3]
        self.data.qpos[7:] = x[3:]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        residual, jacobian = [], []
        for i, (body, local, _, _, role) in enumerate(self.points):
            p = self.point(self.data, body, local)
            mujoco.mj_jac(self.model, self.data, self.jp, self.jr, p, body)
            jp, jr = self.jp[:, self.columns], self.jr[:, self.columns]
            wp = {'foot': 80., 'torso': 10., 'knee': 8., 'elbow': 8., 'palm': 25.}[role]
            if self._contact[i]:
                wp = 150. if role == 'foot' else 80.
            residual.append(wp * (p - self._targets[i]))
            jacobian.append(wp * jp)
            wr = {'foot': 10., 'torso': 12., 'knee': 0., 'elbow': 0., 'palm': 1.5}[role]
            if wr:
                mat = self.data.xmat[body].reshape(3, 3)
                residual.append(wr * (mat - self._rotations[i]).T.ravel())
                jacobian.append(wr * np.vstack([-skew(mat[:, k]) @ jr for k in range(3)]))
        floor = self.floor_points()
        # Keep the complete collision model above ground, including torso/hands
        # during recovery, not only the two ankle origins.
        for g, point in zip(self.geoms, floor):
            penetration = min(0., point[2] - self.cfg.floor_clearance)
            residual.append(np.array([500. * penetration]))
            row = np.zeros((1, len(x)))
            if penetration < 0:
                mujoco.mj_jac(self.model, self.data, self.jp, None, point, int(self.model.geom_bodyid[g]))
                row[0] = 500. * self.jp[2, self.columns]
            jacobian.append(row)
        distances = np.zeros(len(self.collision_pairs))
        collision_jac = np.zeros((len(distances), len(x)))
        centers = self.data.geom_xpos[self.collision_pairs]
        nearby = np.linalg.norm(centers[:, 1] - centers[:, 0], axis=1) < self.pair_radius
        for index in np.flatnonzero(nearby):
            g1, g2 = self.collision_pairs[index]
            dist = mujoco.mj_geomDistance(self.model, self.data, int(g1), int(g2), self.cfg.collision_margin, self.fromto)
            if dist >= self.cfg.collision_margin:
                continue
            distances[index] = 500. * (dist - self.cfg.collision_margin)
            if abs(dist) > 1e-10:
                normal = (self.fromto[3:] - self.fromto[:3]) / dist
                mujoco.mj_jac(self.model, self.data, self.jp, None, self.fromto[:3], int(self.model.geom_bodyid[g1]))
                mujoco.mj_jac(self.model, self.data, self.j_other, None, self.fromto[3:], int(self.model.geom_bodyid[g2]))
                collision_jac[index] = 500. * normal @ (self.j_other - self.jp)[:, self.columns]
        residual.append(distances)
        jacobian.append(collision_jac)
        residual.extend((40. * (x[:3] - self._root), .6 * (x[3:] - self._prior), 1.0 * (x - self._smooth)))
        jacobian.extend((40. * np.eye(len(x))[:3], .6 * np.eye(len(x))[3:], np.eye(len(x))))
        self._cached_x = x.copy()
        self._cached = np.concatenate(residual), np.vstack(jacobian)
        return self._cached

    def retarget(self, motion, progress=None):
        # Each clip starts from model rest state; no dependence on clip order.
        self.data.qpos[:] = self.model.qpos0
        mujoco.mj_kinematics(self.model, self.data)
        root, targets, rotations, contacts = self.prepare_targets(motion)
        frames = len(root)
        qpos = np.empty((frames, self.model.nq))
        prior = motion['joint_pos'][:, self.source_joint_ids].astype(float).clip(self.lower, self.upper)
        stats = {'nfev': [], 'position_error': [], 'min_collision_z': [], 'root_adjustment': []}
        previous = np.r_[root[0], prior[0]]
        fps = float(np.asarray(motion['fps']).item())
        for t in range(frames):
            self.data.qpos[3:7] = motion['body_quat_w'][t, 0]
            self._root, self._targets, self._rotations, self._contact = root[t], targets[t], rotations[t], contacts[t]
            self._prior = prior[t]
            delta = np.zeros_like(previous) if t == 0 else np.r_[root[t] - root[t-1], prior[t] - prior[t-1]]
            self._smooth = previous + delta
            lower = np.r_[root[t] - self.cfg.root_adjustment, self.lower]
            upper = np.r_[root[t] + self.cfg.root_adjustment, self.upper]
            if t:
                step = self.cfg.max_joint_speed / fps
                lower[3:] = np.maximum(lower[3:], previous[3:] - step)
                upper[3:] = np.minimum(upper[3:], previous[3:] + step)
            x0 = self._smooth.clip(lower + 1e-9, upper - 1e-9)
            self._cached_x = None
            result = least_squares(lambda x: self._evaluate(x)[0], x0,
                                   jac=lambda x: self._evaluate(x)[1], bounds=(lower, upper),
                                   max_nfev=self.cfg.max_nfev, ftol=1e-5, xtol=1e-6, gtol=1e-5)
            self._evaluate(result.x)
            qpos[t] = self.data.qpos
            previous = result.x
            achieved = np.array([self.point(self.data, b, local) for b, local, _, _, _ in self.points])
            stats['nfev'].append(result.nfev)
            stats['position_error'].append(np.linalg.norm(achieved - targets[t], axis=-1))
            stats['min_collision_z'].append(self.floor_points()[:, 2].min())
            stats['root_adjustment'].append(np.linalg.norm(result.x[:3] - root[t]))
            if progress is not None and (t + 1) % 100 == 0:
                progress(t + 1, frames)
        # Remove the residual penetration left by soft IK penalties. A smooth
        # upper envelope avoids frame-to-frame jitter of the vertical offset.
        lift = np.maximum(0., self.cfg.floor_clearance - np.array(stats['min_collision_z']))
        lift = np.maximum(lift, gaussian_filter1d(maximum_filter1d(lift, size=7), sigma=1))
        qpos[:, 2] += lift
        stats['floor_lift'] = lift
        stats['min_collision_z'] = np.array(stats['min_collision_z']) + lift
        return qpos, contacts, targets, {key: np.asarray(value) for key, value in stats.items()}


def states_from_qpos(model, qpos, fps):
    """Differentiate final poses and run target FK; free-joint omega is local."""
    frames = len(qpos)
    dt = 1 / fps
    quats = qpos[:, 3:7].copy()
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    for t in range(1, frames):
        if np.dot(quats[t - 1], quats[t]) < 0:
            quats[t] *= -1
    qpos = qpos.copy()
    qpos[:, 3:7] = quats
    qvel = np.empty((frames, model.nv))
    qvel[:, :3] = np.gradient(qpos[:, :3], dt, axis=0)
    rotations = rotation(quats)
    omega = np.empty((frames, 3))
    omega[1:-1] = (rotations[2:] * rotations[:-2].inv()).as_rotvec() / (2 * dt)
    omega[0] = (rotations[1] * rotations[0].inv()).as_rotvec() / dt
    omega[-1] = (rotations[-1] * rotations[-2].inv()).as_rotvec() / dt
    qvel[:, 3:6] = rotations.inv().apply(omega)
    qvel[:, 6:] = np.gradient(qpos[:, 7:], dt, axis=0)
    nb = model.nbody - 1
    result = {'fps': np.array(fps, dtype=np.float32), 'joint_pos': qpos[:, 7:].astype(np.float32),
              'joint_vel': qvel[:, 6:].astype(np.float32)}
    for key, dim in (('body_pos_w', 3), ('body_quat_w', 4), ('body_lin_vel_w', 3), ('body_ang_vel_w', 3)):
        result[key] = np.empty((frames, nb, dim), dtype=np.float32)
    data = mujoco.MjData(model)
    for t in range(frames):
        data.qpos[:] = qpos[t]
        data.qvel[:] = qvel[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        result['body_pos_w'][t] = data.xpos[1:]
        result['body_quat_w'][t] = data.xquat[1:]
        result['body_ang_vel_w'][t] = data.cvel[1:, :3]
        result['body_lin_vel_w'][t] = data.cvel[1:, 3:] - np.cross(data.cvel[1:, :3], data.subtree_com[1] - data.xpos[1:])
    result['joint_names'] = np.array([model.joint(i).name for i in range(1, model.njnt)])
    result['body_names'] = np.array([model.body(i).name for i in range(1, model.nbody)])
    return result
