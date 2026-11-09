# AMP_mjlab

[中文 README](README_zh.md)

Hunter130 AMP training based on [ccrpRepo/AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab),
using mjlab and the included rsl_rl AMP fork. One 25-joint policy learns locomotion,
velocity tracking and fall recovery, with ONNX export support.

## Installation

Requires Linux, Python 3.12 and a working NVIDIA GPU / driver setup.

```bash
git clone https://github.com/HunterOpensource/AMP_mjlab.git
cd AMP_mjlab
# Skip creation if the mjlab environment already exists.
conda create -n mjlab python=3.12 -y
conda activate mjlab
python -m pip install -e .
python -m pip install -e ./rsl_rl
python scripts/apply_mjlab_patch.py
python scripts/apply_mjlab_patch.py --check
```

Install in this order. The local AMP fork reports version `2.3.1`, while mjlab
requires `rsl-rl-lib==5.0.1`, so pip may report a dependency conflict when installing
the fork. This is a known integration limitation; training requires the local AMP runner.

**The mjlab patch is required** for `history_ordering="time"` in the training
configuration. The helper locates mjlab 1.2.0 in the active Python environment,
backs up the original observation manager and installs the patch. After
reinstalling mjlab, reinstall the local rsl_rl fork and reapply the patch.
The patch only supports mjlab 1.2.0.

## Training

Run from the repository root:

```bash
python scripts/list_envs.py --keyword Hunter130
python scripts/train.py Hunter130-AMP-Flat \
  --env.scene.num-envs=4096 --agent.run-name=hunter130_baseline
```

For rough terrain:

```bash
python scripts/train.py Hunter130-AMP-Rough \
  --env.scene.num-envs=4096 --agent.run-name=hunter130_rough
```

Reduce `--env.scene.num-envs` if GPU memory is limited. Logs, checkpoints and
exported policies are saved to `logs/rsl_rl/hunter130_amp_baseline/<run_dir>/`.

Replay a trained policy with the matching Flat / Rough task:

```bash
python scripts/play.py Hunter130-AMP-Flat \
  --checkpoint-file logs/rsl_rl/hunter130_amp_baseline/<run_dir>/model_<iter>.pt
```

## Model and motion data

- `src/assets/robots/hunter130/`: 25-joint MuJoCo model, 26 STL meshes, motor parameters and actuator configuration.
- `src/assets/motions/hunter130/amp_baseline/WalkandRun/`: 17 locomotion clips.
- `src/assets/motions/hunter130/amp_baseline/Recovery/`: 1 fall-recovery clip.
- `src/assets/motions/hunter130/amp_baseline/manifest.json`: provenance, model/motion hashes, retarget settings and quality statistics.

All 18 clips are bundled: 9,992 frames at 50 Hz. Both reset initialization and the
AMP discriminator use this baseline. Training requires no CSV conversion,
retargeting or external SuperMimic data. Delayed resets give fallen environments
a recovery window so one policy can learn locomotion and getting up.

AMP sampling, joint targets and the fall-height threshold match the successfully
trained revision `8a29d4f`: batches cycle through sorted clips, sampling frames
within one clip per batch and repeating the last frame for terminal pairs.
Standard `JointPositionActionCfg` computes targets as the default pose plus scaled
actions. Root height below `0.35 m` triggers the height termination condition;
recovery environments still use delayed resets.

The G1 model, original CSV/NPZ data and this repository's G1 task registrations
have been removed from the current tree. They remain available in Git history
and the upstream repository. Other robot tasks bundled with the mjlab dependency
are unaffected. The optional SuperMimic/LAFAN set at
`src/assets/motions/hunter130/amp/` remains ignored and is not used by default.
Logs, checkpoints, caches and intermediate conversions do not need to be committed.

The six scripts cover training, policy playback, task listing, patch installation,
baseline retargeting and motion review. The old standalone velocity task has been
removed; AMP's velocity-command and terrain curricula live in
`src/tasks/amp_loco/mdp/curriculums.py`.

## Record each reference motion

```bash
conda activate mjlab
python -m pip install Pillow imageio-ffmpeg
python scripts/render_hunter130_motions.py --output logs/motion_review/hunter130
```

Open `index.html` in the output directory to review all 18 clips as dual-view MP4s
with timeline images. Videos are 1280×720 at 50 fps and original speed; the left
foot is blue and the right foot orange. Per-frame audit results are summarized in
`audit.json`. Playback sets reference poses without running a policy or stepping
the dynamics. The output lives under Git-ignored `logs/`.

Use `--clips <clip_stem>` to select clips, or `--max-frames 100` for a short preview.
If EGL selects Mesa software rendering, prefix the command with
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
to select NVIDIA, provided that driver file exists on your system.

## Optional: reproduce retargeting

The original G1 motions come from upstream revision
`6c7a2947fccc973e4af8e6d90e550400f1b6fcfc`. Bounded whole-body inverse kinematics
with contact constraints maps them to Hunter130, then recomputes target poses
and velocities. Clip names, frame counts, sampling rate and root orientations are preserved.

Only regeneration requires an external checkout of the original G1 repository:

```bash
git clone https://github.com/ccrpRepo/AMP_mjlab.git ../AMP_mjlab_g1_source
git -C ../AMP_mjlab_g1_source checkout --detach 6c7a2947fccc973e4af8e6d90e550400f1b6fcfc
python scripts/retarget_g1_to_hunter130.py \
  --source-repo ../AMP_mjlab_g1_source \
  --output logs/retargeted_baseline
```

The manifest's `retarget_code_revision` locates the archived generator used for
the bundled data; `retarget_code_sha256` records its generation-time hash.
Later changes to the external source-path interface did not rewrite the NPZ files.

Retargeting is kinematic and does not guarantee dynamic feasibility in every
frame. Mean foot-target error is about 2.65 mm for locomotion and 6.31 mm for
recovery; maximum recovery self-penetration is about 7.1 mm.

## Validation

```bash
python -m unittest discover -s tests -v
```

Default checks cover the bundled model, motion integrity, AMP features and task
configuration without G1 files. With an external source checkout, enable source
provenance, IK Jacobian and GPU action/reset checks:

```bash
HUNTER130_G1_SOURCE=../AMP_mjlab_g1_source HUNTER130_TEST_GPU=1 \
  python -m unittest discover -s tests -v
```

## Acknowledgements

- [ccrpRepo/AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab): original AMP framework and baseline motions.
- [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab): open-source implementation and inspiration.
- [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab): reference for the rsl_rl AMP implementation.
