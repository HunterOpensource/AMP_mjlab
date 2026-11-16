# AMP_mjlab

[English README](README.md)

基于 [ccrpRepo/AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab) 的 Hunter130 / Hunter180 AMP 训练仓库，使用 mjlab + 本地 rsl_rl AMP 分支。
Hunter130 的 25 关节策略学习走跑、速度跟踪与跌倒恢复；Hunter180 的 23 关节策略使用独立模型及从 Hunter130 重定向的走跑、恢复参考集。两者均支持导出 ONNX。

## 安装

要求 Linux、Python 3.12 和可用的 NVIDIA GPU / 驱动。

```bash
git clone https://github.com/HunterOpensource/AMP_mjlab.git
cd AMP_mjlab
# 已有 mjlab 环境可跳过创建。
conda create -n mjlab python=3.12 -y
conda activate mjlab
python -m pip install -e .
python -m pip install -e ./rsl_rl
python scripts/apply_mjlab_patch.py
python scripts/apply_mjlab_patch.py --check
```

请按此顺序安装。仓库内的 AMP 版 `rsl_rl` 版本号为 `2.3.1`，而 mjlab 的依赖元数据要求
`rsl-rl-lib==5.0.1`，所以安装本地分支时 pip 可能提示依赖冲突。这是当前集成方式的已知限制；
训练需要本地分支提供的 AMP runner。

**mjlab 补丁是必需的**：训练配置使用 `history_ordering="time"`，按时间展开历史观测。
脚本会自动找到当前 Python 环境中的 mjlab 1.2.0，备份并替换观测管理器，不依赖固定的 conda 路径。
重新安装 mjlab 后，需要重新安装本地 `rsl_rl` 分支并应用补丁；补丁只支持 mjlab 1.2.0。

## 训练

### Hunter180 开发分支

开发分支为 `dev/hunter180`，使用本机 Hunter180 URDF 转换的模型（23 关节、约 64.35 kg），
保留其关节轴、限位、质量和惯量。模型、24 个 STL、URDF 和转换后的动作均随仓库提供，
正常训练不依赖外部模型或动作路径。

```bash
conda activate mjlab
python scripts/list_envs.py --keyword Hunter180
python scripts/train.py Hunter180-AMP-Flat \
  --env.scene.num-envs=4096 --agent.run-name=hunter180_baseline
```

复杂地形任务为 `Hunter180-AMP-Rough`。显存不足时减小并行环境数。
日志、检查点及 ONNX 位于 `logs/rsl_rl/hunter180_amp_baseline/<run_dir>/`。
首次验证可先运行短训练：

```bash
python scripts/train.py Hunter180-AMP-Flat \
  --env.scene.num-envs=64 --agent.max-iterations=3 \
  --agent.run-name=hunter180_smoke --enable-nan-guard=True
python scripts/play.py Hunter180-AMP-Flat \
  --checkpoint-file logs/rsl_rl/hunter180_amp_baseline/<run_dir>/model_<iter>.pt
```

电机参数使用用户指定的训练参数表，23 个关节的 `armature` 均为 `0.10 kg·m²`。
`effort` 为下表指定的训练扭矩限值；源 URDF 保留原始硬件规格，训练加载时会同步设置
关节和执行器的扭矩限制。左右同名关节使用相同数值。

| 关节（左右相同） | Kp (N·m/rad) | Kd (N·m·s/rad) | armature (kg·m²) | 扭矩限值 (N·m) | action_scale (rad) |
|---|---:|---:|---:|---:|---:|
| hip_pitch | 250 | 15 | 0.10 | ±400 | 0.3 |
| hip_roll | 400 | 30 | 0.10 | ±400 | 0.3 |
| hip_yaw | 200 | 10 | 0.10 | ±400 | 1.0 |
| knee | 400 | 30 | 0.10 | ±400 | 0.3 |
| ankle_pitch | 200 | 10 | 0.10 | ±200 | 0.7 |
| ankle_roll | 200 | 10 | 0.10 | ±100 | 0.3 |
| waist_yaw | 100 | 5 | 0.10 | ±300 | 1.0 |
| waist_roll | 150 | 7.5 | 0.10 | ±150 | 1.0 |
| waist_pitch | 150 | 7.5 | 0.10 | ±150 | 1.0 |
| shoulder_pitch | 100 | 5 | 0.10 | ±150 | 0.5 |
| shoulder_roll | 100 | 5 | 0.10 | ±150 | 0.5 |
| shoulder_yaw | 100 | 5 | 0.10 | ±150 | 0.5 |
| elbow | 50 | 5 | 0.10 | ±150 | 0.5 |

实际配置为 `src/assets/robots/hunter180/assets/motors.json`，来源与配置校验和记录在
同目录 `motor_parameters_source.json`。这组数值替代了最初按 Hunter130 惯量比缩放的估计值。
可选的惯量估算工具只用于对照，必须输出到独立目录：

```bash
python scripts/prepare_hunter180_motors.py --output logs/hunter180_motor_estimates
```

Hunter180 训练数据全部来自本仓库 `src/assets/motions/hunter130/amp_baseline/`，
通过带关节限位、接触和自碰撞约束的全身 IK 重新重定向到 Hunter180：
17 段走跑、1 段恢复，共 9,992 帧、50 Hz，保留片段名称、帧数、采样率及根部朝向。
按两款模型中立姿态的脚底到骨盆高度比缩放根部/足端轨迹，并按目标骨长适配手臂。
Hunter180 肘部的固定朝向比 Hunter130 多约 90°，关节先验对此作了修正。
当前 Hunter180 模型手臂终止于肘部壳体，没有腕/手连杆，因此末端目标采用实际壳体端点。
目标位姿经 IK 求解后，重新计算全部关节速度、连杆位姿与速度，不直接复制 Hunter130 的身体状态。
训练和回放保留跌倒恢复的延迟重置机制。

来源模型、每个源文件与目标文件的校验和、求解参数和质量统计保存在
`src/assets/motions/hunter180/amp_baseline/manifest.json`。
训练直接使用随仓库提供的 NPZ。需要重新生成或验证时：

```bash
python scripts/retarget_hunter130_to_hunter180.py
HUNTER180_TEST_GPU=1 python -m unittest discover -s tests -p test_hunter180.py -v
```

单片段调试须输出到独立目录，例如：

```bash
python scripts/retarget_hunter130_to_hunter180.py \
  --clips walk_forward_loop_002__A022 --output logs/hunter180_retarget_preview
```

重定向是运动学处理；短训练检查用于验证训练流程，不能证明策略已收敛或动作在实机上可执行。

### Hunter130

在仓库根目录执行：

```bash
python scripts/list_envs.py --keyword Hunter130
python scripts/train.py Hunter130-AMP-Flat \
  --env.scene.num-envs=4096 --agent.run-name=hunter130_baseline
```

复杂地形训练：

```bash
python scripts/train.py Hunter130-AMP-Rough \
  --env.scene.num-envs=4096 --agent.run-name=hunter130_rough
```

显存不足时减小 `--env.scene.num-envs`。日志、检查点和导出的策略保存在
`logs/rsl_rl/hunter130_amp_baseline/<run_dir>/`。

回放已训练的策略：

```bash
python scripts/play.py Hunter130-AMP-Flat \
  --checkpoint-file logs/rsl_rl/hunter130_amp_baseline/<run_dir>/model_<iter>.pt
```

回放任务应与训练时的 Flat / Rough 类型一致。

## 模型与训练数据

- `src/assets/robots/hunter130/`：25 关节 MuJoCo 模型、26 个 STL 网格、电机参数及执行器配置。
- `src/assets/motions/hunter130/amp_baseline/WalkandRun/`：17 段走跑动作。
- `src/assets/motions/hunter130/amp_baseline/Recovery/`：1 段跌倒恢复动作。
- `src/assets/motions/hunter130/amp_baseline/manifest.json`：来源、模型和动作校验和、重定向参数与质量统计。

18 段动作共 9,992 帧、50 Hz，随仓库直接提供。默认训练同时用它们初始化机器人状态、
提供 AMP 判别器参考；训练不需要重新转换 CSV、运行重定向或下载 SuperMimic 数据。
延迟重置机制给部分跌倒环境留出恢复窗口，使单个策略学习走跑和起身。

AMP 采样、关节目标处理和跌倒高度阈值对齐成功训练版本 `8a29d4f`：按文件排序逐批循环
选择一个片段，在片段内随机取帧；抽到末帧时下一帧仍为末帧。关节目标使用普通
`JointPositionActionCfg`，按默认姿态加动作缩放计算；根部高度低于 `0.35 m` 触发高度终止条件，
恢复环境仍适用延迟重置。

G1 模型、原始 CSV / NPZ 和本仓库的 G1 任务注册已从当前工作树移除。
原始资料仍保留在 Git 历史和上游仓库中；mjlab 依赖包自身包含的其他机器人任务不受此改动影响。
可选的 SuperMimic/LAFAN 对比集 `src/assets/motions/hunter130/amp/` 仍被 Git 忽略，默认训练不使用它。
日志、检查点、缓存和中间转换文件也不需要提交。

仓库保留训练、策略回放、环境列表、补丁安装、Hunter130 基准动作重定向和视频检查，
并提供 Hunter180 可选电机参数估算及 Hunter130 → Hunter180 动作重定向脚本。
旧的独立 velocity 任务已移除，AMP 使用的速度与地形课程学习函数位于
`src/tasks/amp_loco/mdp/curriculums.py`。

## 逐段录制参考动作

```bash
conda activate mjlab
python -m pip install Pillow imageio-ffmpeg
python scripts/render_hunter130_motions.py --output logs/motion_review/hunter130
```

用浏览器打开输出目录的 `index.html`，即可逐段查看全部 18 段动作的双视角 MP4
和时间线截图。视频为 1280×720、50 fps、原速，蓝色为左脚，橙色为右脚；
逐帧检查结果保存在 `audit.json`。这会直接回放参考姿态，不运行训练策略或动力学步进。
视频输出位于被 Git 忽略的 `logs/` 下。

可用 `--clips <片段名，不含扩展名>` 选择片段；`--max-frames 100` 仅用于短预览。
如果本机 EGL 选择了 Mesa 软件渲染，可在命令前添加
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
选择 NVIDIA 驱动，前提是该驱动文件存在。

## 可选：重现重定向

基准数据来自上游提交 `6c7a2947fccc973e4af8e6d90e550400f1b6fcfc` 中的 G1 动作，
通过带关节限位和接触约束的全身逆运动学转换为 Hunter130，并重新计算目标机器人的位姿、速度。
保留片段名称、帧数、采样率与根部朝向。

仅重新生成数据时，需要准备外部的原始 G1 仓库：

```bash
git clone https://github.com/ccrpRepo/AMP_mjlab.git ../AMP_mjlab_g1_source
git -C ../AMP_mjlab_g1_source checkout --detach 6c7a2947fccc973e4af8e6d90e550400f1b6fcfc
python scripts/retarget_g1_to_hunter130.py \
  --source-repo ../AMP_mjlab_g1_source \
  --output logs/retargeted_baseline
```

`manifest.json` 的 `retarget_code_revision` 指向已有基准数据生成代码的存档提交，
`retarget_code_sha256` 记录生成时的代码；之后的外部来源路径接口调整没有改写已有 NPZ。

重定向是运动学处理，不保证每帧动态可执行。已有基准集的平均足端目标误差约为
走跑 2.65 mm、恢复 6.31 mm，恢复动作最大自穿透约 7.1 mm。

## 检查

```bash
python -m unittest discover -s tests -v
```

默认校验随仓库提供的模型、动作完整性、AMP 特征及任务配置，不需要 G1 文件。
准备好外部来源后，可同时启用来源追溯、IK 雅可比和 GPU 动作 / 重置检查：

```bash
HUNTER130_G1_SOURCE=../AMP_mjlab_g1_source HUNTER130_TEST_GPU=1 \
  python -m unittest discover -s tests -v
```

## 致谢

- [ccrpRepo/AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab)：原始 AMP 训练框架与基准动作。
- [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)：开源实现与启发。
- [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)：本项目 rsl_rl AMP 实现的参考。
