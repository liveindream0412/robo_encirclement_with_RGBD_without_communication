📌 项目简介

本项目面向RoboMaster 多机器人协同围捕任务，在Isaac Sim 4.5.0仿真环境中，搭建了一套基于RGB-D 视觉感知的端到端多智能体决策框架。
本系统不依赖机器人间显性通信，三台参数共享的机器人仅依靠自身车载 RGB-D 相机的局部视觉观测，自主完成目标、队友状态的感知推理，自适应形成稳定、均匀的环形围捕队形，实现纯视觉驱动的无通信协同围捕。
项目采用行为克隆（BC）预训练 + 强化学习（RL）微调的两阶段训练范式：依托专家示范数据完成策略快速初始化，再通过多智能体强化学习优化协同队形稳定性、环境鲁棒性，兼顾训练收敛效率与最终协同控制性能，完美适配视觉驱动的多机器人协同任务场景。

✨ 核心特色

- 无通信协同机制：机器人之间无任何显性信息交互、无全局位姿广播，仅依靠本机局部视觉观测完成自主协同与队形博弈，贴近真实受限通信部署场景。
- RGB-D 端到端感知决策：融合彩色语义信息与深度几何信息，让策略同时具备目标语义识别、相对距离估算、空间位置推理能力，提升围捕控制精度。
- 多智能体共享策略：三台机器人共用同一套网络参数与决策模型，结构统一、部署轻量化，大幅提升样本利用率，便于拓展与工程落地。
- 模仿学习 + 强化学习融合训练：通过专家模仿学习解决从零探索的低效问题，依托强化学习微调优化协同几何约束，兼顾收敛速度与任务性能。
- 面向围捕几何约束的精细化奖励设计：摒弃单一的目标追踪奖励，围绕围捕队形、空间分布、安全约束多维度设计奖励函数，保障环形围捕任务的稳定性。

🎯 项目目标

本项目核心验证目标：在无车间通信、无全局定位信息的强约束条件下，多移动机器人仅依托局部 RGB-D 观测，能否实现稳定、可泛化、可部署的自主协同围捕。
围绕核心目标，系统重点实现以下能力：
- 基于视觉的目标识别与相对位姿估计
- 隐性队友感知与多机器人协同避碰
- 自适应围捕半径控制与周向角度均匀分布
- 三机器人等边环形队形的自主生成与稳态保持
- 共享策略在多智能体场景的泛化复用与稳定部署

🔧 方法概览

1. 仿真环境建模
本项目基于 Isaac Lab / Isaac Sim 4.5.0 构建三机一目标多智能体围捕仿真环境，为每台 RoboMaster 移动底盘挂载独立 RGB-D 相机，实现并行训练与独立局部观测。环境核心配置与约束如下：
- 三台同构 RoboMaster 移动机器人底盘 + 单个动态围捕目标
- 各机器人独立 RGB-D 视觉观测通道，无共享全局观测
- 场景初始位姿、朝向、目标位置随机化，提升策略泛化性
- 内置队形稳定性约束、机器人防撞、目标姿态约束等安全机制

2. 两阶段策略训练框架
采用模仿学习预训练 + 强化学习微调的分层训练方案，解决纯强化学习探索低效、难收敛的问题：
- 阶段一：行为克隆（BC）预训练
基于人工设计的专家策略采集高质量示范数据集，通过监督学习让智能体快速掌握目标靠近、姿态对准、基础队形生成等核心行为，为后续强化学习提供优质参数初始化。
- 阶段二：多智能体强化学习微调
以 BC 预训练权重为初始参数，基于 PPO 算法开展多智能体自博弈训练，优化机器人协同几何关系、队形均匀性、抗干扰能力，适配复杂随机场景。
注：经测试，强化学习微调后效果与纯行为克隆基本持平，BC 预训练策略已可实现稳定围捕。

3. 精细化奖励机制设计
本任务核心为视觉驱动的多智能体队形协同围捕，而非简单目标追踪，因此奖励函数围绕几何队形约束核心设计：
- 目标径向距离约束：维持合理围捕半径，避免过近碰撞、过远失效
- 周向角度约束：保证三台机器人围绕目标均匀分布
- 队形几何约束：优化三机相对位置，趋近等边三角形稳态队形
- 稳态保持奖励：队形稳定持续一段时间后给予累加奖励
- 姿态约束：机器人朝向始终对准围捕目标
- 安全惩罚：机器人间、机器人与目标碰撞惩罚
- 动作平滑惩罚：抑制无效自旋、剧烈动作波动，提升运动稳定性

📁 项目结构

├─ robomaster_encirclement_env.py          # 多智能体围捕环境核心定义

├─ robomaster_encirclement_env_cfg.py      # 环境、相机、奖励、任务参数配置

├─ train_bc.py                             # 行为克隆预训练脚本

├─ train_shared_rsl_rl.py                  # 共享策略强化学习微调脚本

├─ play_shared_rsl_rl.py                   # RL 策略推理与仿真演示脚本

├─ play_bc_actor.py                        # BC 预训练策略推理与演示脚本

├─ collect_expert_dataset.py               # 专家示范数据集采集脚本

├─ teacher.py                              # 专家策略/教师示范逻辑实现

├─ capture_static_camera_scenarios.py      # 静态场景视觉数据采样与记录

├─ shared_marl_vecenv_wrapper.py           # 多智能体共享Actor网络封装器

├─ agents/

│  └─ rsl_rl_ppo_cfg.py                    # PPO强化学习算法参数配置

└─ networks/
 
   ├─ actor_critic_rgbd_gru.py            # GRU时序版RGB-D Actor-Critic网络

   └─ actor_critic_rgbd_mlp.py            # MLP基础版RGB-D Actor-Critic网络

💡 技术亮点与研究价值
1. 强约束无通信协同方案
摒弃传统多机器人协同依赖的显性通信、全局定位、集中式控制方案，仅基于局部视觉观测实现隐性协同，适配带宽受限、无定位、强干扰的真实工程场景，具备极高的落地价值。
2. 共享策略轻量化部署
同构机器人参数共享，大幅降低训练成本、提升样本利用率，同时简化部署流程、降低硬件算力依赖，适配批量机器人集群落地场景。
3. RGB-D 融合感知优势
相较于纯视觉RGB方案，深度信息提供精准几何测距与空间位置线索，有效支撑围捕半径控制、队形规整、碰撞规避，大幅降低策略学习难度。
4. 广泛的研究与拓展场景
本框架可直接用于以下方向的科研验证与算法迭代：
- 多智能体强化学习（MARL）算法验证
- 无通信多机器人协同控制研究
- 视觉伺服、纯视觉机器人导航与编队控制
- 模仿学习与强化学习融合训练范式探索
- 仿真到实机的协同策略迁移学习
5. 可拓展性极强
基于现有框架，可快速迭代拓展更多复杂任务场景：
- 动态高速目标围捕与拦截任务
- 异构机器人集群协同围捕
- 复杂障碍物环境下的视觉编队协同
- 局部观测受限场景的鲁棒协作机制研究
⚙️ 环境复现与运行指南
运行环境：Isaac Sim 4.5.0 + Isaac Lab
1. 项目部署路径
将本项目所有代码放置于 Isaac Lab 任务目录下：
~/source/isaaclab_tasks/isaaclab_tasks/direct/robomaster_encirclement
2. 专家数据集采集
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/robomaster_encirclement/collect_expert_dataset.py \
  --task Isaac-Robomaster-Encirclement-RGBD-Direct-v0 \
  --num_envs 128 \
  --num_steps 2000 \
  --device cuda:0 \
  --output logs/robomaster_bc/expert_dataset_rgbd_aux_v1.pt
3. 行为克隆（BC）预训练
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/robomaster_encirclement/train_bc.py \
  --dataset logs/robomaster_bc/expert_dataset_rgbd_aux_v1.pt \
  --epochs 40 \
  --batch_size 512 \
  --lr 1e-4 \
  --aux_weight 0.5 \
  --device cuda:0 \
  --output logs/robomaster_bc/bc_actor_rgbd_aux_v1.pt
4. 策略推理与效果演示（含视频录制）
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/robomaster_encirclement/play_bc_actor.py \
  --task Isaac-Robomaster-Encirclement-RGBD-Direct-v0 \
  --checkpoint logs/robomaster_bc/bc_actor_rgbd_aux_v1.pt \
  --num_envs 1 \
  --device cuda:0 \
  --real-time \
  --video \
  --video_length 800 \
  --save_camera_video \
  --camera_video_length 800 \
  --camera_video_fps 30
补充说明：项目已完整实现强化学习微调训练脚本，经实测，RL 微调后策略性能、队形稳定性与纯行为克隆预训练效果基本一致，BC 策略已完全满足围捕任务需求。

https://github.com/user-attachments/assets/71d63565-8b69-4054-a997-7d3ed69a7904
