# 保住"真·泛化"(不只 LIBERO-130)· 设计思路 (2026-07-23)

## 0. 问题(用户提的更深目标)
现在的 dual-KL 把"泛化"定义成 held-out LIBERO 套件——同域、同物体、同机器人,太窄。真正想保的是 base VLA 的**广泛开放世界泛化**:对**未见物体 / 场景 / 纹理 / 指令表述 / 执行扰动**的零样本操作能力。两个未知:(A) 这种泛化怎么测,(B) 怎么设计算法保住它。

## 1. 先厘清:VLA 的"真泛化"是什么、怎么测(文献)
不是 VQA(VLM 冻结,用户已否);是 **action policy 的 OOD 操作泛化**,按轴度量:
- **Semantics**:未见物体/容器、指令改写、干扰物
- **Vision**:未见纹理/背景、图像噪声
- **Execution**:随机初始位姿、episode 中途挪动物体

**现成 benchmark**(按此三轴设计):
- **Colosseum V2**(arXiv:2605.27759,"Benchmarking Generalization for VLA"):每轴 hold-out(9 新物体/16 新容器/5 新纹理/16 干扰背景)。
- **VLABench**(ICCV'25):seen(技能)vs unseen(泛化)物体两档。
- **SimplerEnv**:上面那篇的评测台(3 轴)。
- **我们现有可跑的**:**LIBERO-PRO**(object-appearance=未见物体、swap=未见位置、lan=改写指令)——正好覆盖 Semantics+部分 Execution,**在我们 infra 内**(Colosseum/SimplerEnv 需新 sim,ManiSkill/Vulkan 仍卡)。

→ **测试方案**:主用 **LIBERO-PRO 三扰动**当"真 OOD 泛化"(比 held-out 套件真);若要更强 claim,后续接 Colosseum V2 / SimplerEnv。

## 2. 关键洞见:广泛泛化"活在视觉表征里",且能 data-free 地锚住
**"Don't Blind Your VLA"(arXiv:2510.25616)**:微调 VLA 时,把中层 transformer 视觉特征(layer 16)**对齐到一个冻结的通用视觉基础模型**(C-RADIOv3;DINOv2/SigLIP 同类)——patch-wise 余弦相似度辅助损失。
- **data-free**:只用冻结 teacher 的预算特征,不加任何机器人数据;轻量。
- **为什么解决"广泛锚哪来"**:通用视觉模型对**任意图像**都给好特征 → 是天然的**广泛、on-manifold 监督**,不需要广泛动作 rollout。
- **结果**:OOD 三轴全升(Semantic 0.61 vs SFT 0.49 / Vision 0.83 vs 0.74),ImageNet 线性探针 82% vs 77%(视觉泛化真被保住);**冻结 encoder 反而崩(0.03-0.05)**——说明"冻住"不行,要"对齐"。

## 3. 我们的设计:双通道锚(动作 + 视觉),全 data-free
把当前 dual-KL(动作分布锚)**加一条视觉表征锚**:
```
L = L_task(reverse-KL to teacher)                    ← 学新任务
  + λ_a · g(H_B)·KL(π_B ‖ π_θ)  (on rollout states)   ← 保动作行为(现有 dual-KL)
  + λ_v · (1 − cos(f_θ^(l), f_teacher^(l)))            ← 保广泛视觉泛化(新)
```
- **f_teacher**:冻结通用视觉模型(C-RADIO/DINOv2)的 patch 特征;或退一步用 **base VLA 自己冻结的视觉特征**(更省事,但泛化上限=base;用外部通用模型上限更高)。
- **在哪些图上对齐**:学生 rollout 的图像即可(每张图 teacher 都能给广泛特征)——**这就是 data-free 广泛锚的关键**:图像来自窄任务,但视觉 teacher 的监督是广泛的。
- **两条锚互补**:动作锚保"会做的任务不退化",视觉锚保"对未见物体/场景的感知泛化不退化"(OOD 的根)。

## 4. 为什么够 ICLR(新意 + 可证伪发现)
- **新机制**:据我们所知,没人把"动作分布锚(保任务)"+"通用视觉表征锚(保广泛 OOD 泛化)"在**持续蒸馏 / data-free** 下合起来;"Don't Blind"只做 SFT+视觉对齐、无持续学习/动作锚;我们的 dual-KL 只做动作、泛化窄。
- **非平凡发现(待验证)**:*要保住真·OOD 泛化,必须锚在广泛泛化真正所在的地方=视觉表征(经通用视觉 teacher,data-free),只锚动作分布不够。* 用 LIBERO-PRO OOD 三轴证明:视觉锚 > 动作锚 > 无锚。
- **框成训练方法 + recipe**,不是新算法。必须打败:WiSE-FT/RETAIN、reverse-LiNeS、纯 dual-KL(动作锚)、"Don't Blind"式纯视觉对齐(无持续学习)。

## 5. 第一个实验(最小、可行、在现有 infra)
- base = 130 通才;distill object(同 dual-KL 设置)。
- arm:{plain-OPD, dual-KL(动作锚), **visual-anchor(视觉锚)**, **both(双锚)**}。
- 视觉 teacher 先用 **base VLA 自己冻结的视觉特征**(零新依赖,验证机制);过了再换外部 C-RADIO/DINOv2(泛化上限更高)。
- 测:object(学新)+ **LIBERO-PRO object/swap/lan(真 OOD 泛化)** + held-out 套件(广度),150 次(3-seed)。
- **通过门**:视觉锚 or 双锚在 **PRO 三轴**上的保留率 > 纯动作 dual-KL 和 plain-OPD;object 不掉。
- infra:OpenVLA-OFT 视觉 backbone(SigLIP+DINOv2)在 LoRA 下会动 → 视觉锚有意义;在 `fsdp_actor_worker` 取中层 patch 特征 + 冻结 teacher 特征算 cos 损失(env 开关默认关)。

## 6. 第一个实验结果(4-way, 50-trial 单 seed, success_once)—— 2026-07-23
base 参考:object 0.64 / PRO-object 0.38 / swap 0.68 / lan 0.70 / spatial 0.66 / goal 0.40 / long 0.72。

| 轴 | plain-OPD | dual-KL(动作锚) | visual(视觉锚) | **both(双通道)** |
|---|---|---|---|---|
| object 学新 | 0.86 | **0.92** | **0.92** | 0.86 |
| PRO-object 未见外观 | 0.40 | **0.52** | 0.44 | 0.48 |
| PRO-swap 未见位置 | 0.82 | 0.88 | **0.94** | **0.94** |
| PRO-lan 改写指令 | 0.84 | 0.90 | 0.90 | **0.96** |
| spatial (held-out) | 0.56 | 0.64 | 0.64 | **0.70** |
| goal (held-out) | 0.50 | 0.46 | 0.48 | **0.52** |
| long (held-out) | 0.66 | 0.70 | 0.66 | **0.74** |
| **PRO 三轴均** | 0.687 | 0.767 | 0.760 | **0.793** |
| **held-out 三套均** | 0.573 | 0.600 | 0.593 | **0.653** |

**读数(诚实,50-trial 单 seed 有噪声):**
1. **双通道(both)= 最广泛化保留**:PRO 均 0.793 + held-out 均 0.653 双料最高,7 轴赢/并列 5 轴(swap/lan/spatial/goal/long)→ 支持"保广泛泛化需两条锚一起"机制。
2. **代价**:both 学 object 0.86(=plain,< dual-KL 0.92),加视觉锚掉 ~6 分获取(待 150 次确认是否噪声)。
3. **唯一例外 = PRO-object 未见外观**:纯动作锚 dual-KL 0.52 反而最好,视觉锚没帮上 → 因为视觉 teacher 是 **base 自己冻结特征(对未见外观本就弱)**;要赢这轴须换外部 **DINOv2/C-RADIO**。

**下一步二选一**:(A) 150 次(3-seed)坐实 both>dual-KL 的广度优势;(B) 换外部 DINOv2 视觉 teacher 攻"未见外观"轴。
**代码状态**:双通道锚已在 `fsdp_actor_worker.py`(base forward + action/visual 双锚)+ `openvla_oft_action_model.py`(mid_features 抽取)+ config `visual_anchor_lambda/layer`,默认全 0 = plain-OPD 逐字节等价。

## 关键文献
- OOD 泛化机制:**Don't Blind Your VLA**(2510.25616,视觉表征对齐,data-free)← 核心。
- 泛化 benchmark:**Colosseum V2**(2605.27759)、**VLABench**(ICCV'25,2412.18194)、SimplerEnv。
- 保 VLM 知识对照:**VLM2VLA / Actions-as-Language**(OpenReview sFO9d6XSlf)、**FiberTune**(2606.08653,保视觉残差)。
- 我们的基础:dual-KL 动作锚(`project_dualkl_anchor_result`,已验证 Pareto 打过 RETAIN)。
