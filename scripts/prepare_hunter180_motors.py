"""Estimate Hunter180 PD gains from Hunter130 in a separate diagnostic directory.

Training uses the user-specified table in assets/motors.json. These inertia-based
estimates are optional diagnostics and must not overwrite that training table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ROOT / "src/assets/robots"
OUTPUT = ROBOTS / "hunter180/assets"


def joint_inertias(xml: Path) -> dict[str, float]:
    """Diagonal joint-space inertia with the base/other joints held fixed.

    Remove rotor armature before measuring the rigid-body load. Scaling Kp,
    Kd and armature by the same ratio preserves the reference linearized
    natural frequency and damping ratio for each individual joint.
    """
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.dof_armature[:] = 0.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    matrix = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, matrix)
    return {
        model.joint(i).name: float(matrix[model.jnt_dofadr[i], model.jnt_dofadr[i]])
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    }


def calculate_motor_parameters():
    reference_xml = ROBOTS / "hunter130/assets/hunter130.xml"
    reference_json = ROBOTS / "hunter130/assets/motors.json"
    target_xml, target_urdf = OUTPUT / "hunter180.xml", OUTPUT / "hunter180.urdf"
    reference = json.loads(reference_json.read_text())
    source_inertia, target_inertia = joint_inertias(reference_xml), joint_inertias(target_xml)
    limits = {
        joint.attrib["name"]: float(joint.find("limit").attrib["effort"])
        for joint in ET.parse(target_urdf).getroot().findall("joint")
        if joint.attrib["type"] != "fixed"
    }
    motors, calculations = {}, {}
    for name, inertia in target_inertia.items():
        ratio = inertia / source_inertia[name]
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"Invalid inertia ratio for {name}: {ratio}")
        motor = reference[name]
        motors[name] = {
            key: motor[key] * ratio for key in ("stiffness", "damping", "armature")
        }
        motors[name].update(effort=limits[name], action_scale=motor["action_scale"])
        calculations[name] = {
            "hunter130_inertia_kg_m2": source_inertia[name],
            "hunter180_inertia_kg_m2": inertia,
            "inertia_ratio": ratio,
        }
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "method": "Kp180/Kp130 = Kd180/Kd130 = armature180/armature130 = M180[j,j]/M130[j,j]",
        "reference_pose": "All hinge positions and velocities zero; rotor armature excluded from M",
        "action_scale": "Same as the corresponding Hunter130 joint (radians per policy action)",
        "effort": "Hunter180 URDF hardware limit; not scaled from Hunter130",
        "limitation": "Initial simulation gains; coupling, transmission changes and hardware response need tuning",
        "input_sha256": {
            str(path.relative_to(ROOT)): sha(path)
            for path in (reference_xml, reference_json, target_xml, target_urdf)
        },
        "joints": calculations,
    }
    return motors, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Separate diagnostic directory, e.g. logs/hunter180_motor_estimates")
    args = parser.parse_args()
    if args.output.expanduser().resolve() == OUTPUT.resolve():
        parser.error("Use a separate --output directory; the training motor table is user-specified")
    args.output = args.output.expanduser()
    motors, report = calculate_motor_parameters()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("motors.json", motors), ("motor_derivation.json", report)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(f"Derived {len(motors)} Hunter180 motors from Hunter130: {args.output}")
