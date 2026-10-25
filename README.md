# ComfyUI: MOSS 提示词权重插件 (DiT Attention V-Scaling 深度加权)

专为 **ComfyUI** 深度定制的 **MOSS-Image** 专用提示词权重加权插件。
采用经架构验证的 **DiT Attention V-Scaling（值向量缩放）** 核心算法，彻底解决 Qwen3.5 文本编码器与 MOSS DiT 42 层 RMSNorm 归一化对传统 CLIP 权重的数学消除问题，并完美支持 Danbooru 角色转义括号与负向权重语法。

---

## 🌟 核心特性与工作原理

1. **彻底攻克 RMSNorm 权重消除难题**:
   - MOSS-Image 采用 Qwen3.5 文本编码器与 42 层 DiT 架构，其密集分布的 RMSNorm 归一化层会在前向传播中强行归一化 Token 嵌入模长，导致传统 WebUI/ComfyUI 权重语法完全失效。
   - 本插件绕过 Token 级缩放，在 42 层 DiT 注意力机制中直接对文本 **Value（值向量）矩阵** 执行精准调制，赋予加权词条真正强效的特征塑形能力。

2. **权重乘法放大机制（仅在权重大于 1 时生效）**:
   - **生效计算公式**：
     $$\text{实际生效权重} = \begin{cases} w \times \text{权重放大倍数}, & \text{当 } w > 1.0 \\ w, & \text{当 } w \le 1.0 \end{cases}$$
   - **默认放大倍数 = 5.0**：
     - 用户输入 `(huge breasts:1.1)` $\rightarrow$ 实际按 **$5.5$** 权重生效（温和轻度强化）
     - 用户输入 `(huge breasts:1.2)` $\rightarrow$ 实际按 **$6.0$** 权重生效（显著塑形效果）
     - 用户输入 `(huge breasts:1.5)` $\rightarrow$ 实际按 **$7.5$** 权重生效（强力特征突破）
     - 用户输入 `(huge breasts:2.0)` $\rightarrow$ 实际按 **$10.0$** 权重生效（夸张比例突破）
   - **保护 $\le 1.0$ 的权重不受影响**：
     - 当提示词权重 $\le 1$ 时（如普通词 1.0、Danbooru 角色名 `saori \(blue archive\)`、减权 `[tag:0.8]`、负向抑制 `(ugly:-1.0)`），保持原值，不乘放大倍数。

3. **Query 分离防崩机制 (Query Separation)**:
   - 图像 Query $Q_{\text{img}}$ 读取按倍数放大后的 $V$ 矩阵，实现对画面特征的强力控制；
   - 文本 Query $Q_{\text{txt}}$ 严格读取未放大的原始文本值，确保文本自注意力在 42 层 DiT 中保持结构稳定，绝不黑图、绝不烧色。

4. **纯负向权重单通道加速**:
   - 对于 `(ugly:-1.0)` 等纯符号反转权重，自动回退到单遍标准 Attention 计算，完全跳过 Query 拆分开销，生成速度与无权重时完全一致。

5. **全架构双引擎兼容**:
   - 完美兼容 ComfyUI 原生 `comfy.ldm.moss_image.model` 以及社区 `custom_nodes/moss-image` 加载器；
   - 无论是 fp8、int8、bf16 还是 fp32 精度模型均可稳定运行。

---

## 🚀 节点列表与连接方式

### 1. `🎯 MOSS Text Encode (V-Scaling)`
* **作用**：替代或串联原有的 `CLIPTextEncode`，支持解析提示词权重并生成携带 V-Scaling 注意力掩码的 `CONDITIONING`。
* **输入**：
  * `clip`: 连接到 Loader 输出的 `CLIP`。
  * `text`: 提示词文本（支持多行与动态语法）。
  * `weight_multiplier`: 权重放大倍数，默认 `5.0`（范围 1.0 ~ 50.0）。
  * `emphasis_mode`: 权重模式，默认 `Original`（解析括号权重）；选 `Ignore` 则将所有括号视为纯字面文本。
* **输出**：
  * `conditioning`: 输出连接到 KSampler 的 `positive` 或 `negative`。

### 2. `🎯 Apply MOSS Prompt Weights (V-Scaling)`
* **作用**：对 `MODEL` 打上 V-Scaling 注意力前向补丁，使其在采样时自动识别 Conditioning 中的权重掩码并启用 Query 分离注意力。
* **输入**：
  * `model`: 连接到 Loader 输出的 `MODEL`。
* **输出**：
  * `model`: 输出连接到 KSampler 的 `model`。

### 3. `🎯 MOSS Conditioning Weights (V-Scaling)`
* **作用**：为已有的 `CONDITIONING` 注入或覆盖 V-Scaling 提示词权重掩码。
* **输入**：
  * `conditioning`: 原有条件。
  * `clip`: `CLIP` 模型。
  * `text`: 提示词文本。
  * `weight_multiplier`: 放大倍数。
* **输出**：
  * `conditioning`: 注入权重掩码后的新条件。

---

## 🔌 经典工作流连接示例 (Workflow Wiring)

```text
[MOSS Loader / Checkpoint Loader]
  ├── MODEL ────► [Apply MOSS Prompt Weights] ────► KSampler (model)
  ├── CLIP  ────► [MOSS Text Encode (Positive)] ─► KSampler (positive)
  │          └──► [MOSS Text Encode (Negative)] ─► KSampler (negative)
  └── VAE   ─────────────────────────────────────► VAE Decode (vae)
```

---

## 📊 语法示例与权重对照表（默认放大倍数: 5）

| 提示词写法 | 原始解析权重 | 实际生效权重 ($w \times 5$) | 对应机制 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `1girl, solo` | `1.0` | **`1.0`** | 单通道标准 Attention | 零额外开销 |
| `saori \(blue archive\)` | `1.0` | **`1.0`** | **转义保护** | 字符字面值，不触发放大 |
| `[bad hands:0.8]` | `0.8` | **`0.8`** | **保留减权** | 权重 $\le 1$，保持原值，不乘放大倍数 |
| `(ugly:-1.0)` | `-1.0` | **`-1.0`** | **负向抑制** | 权重 $\le 1$，保持原值，走快速单通道 |
| `(huge breasts:1.1)` | `1.1` | **`5.5`** | **线性放大** | 温和强化 |
| `(huge breasts:1.2)` | `1.2` | **`6.0`** | **线性放大** | 显著塑形效果 |
| `(huge breasts:1.5)` | `1.5` | **`7.5`** | **线性放大** | 强力特征突破 |
| `(huge breasts:2.0)` | `2.0` | **`10.0`** | **线性放大** | 夸张体型突破 |

---

## 📁 目录结构

```text
ComfyUI-MOSS-Prompt-Weights/
├── __init__.py                # 节点注册映射与包导出入口
├── nodes.py                   # ComfyUI 节点定义 (MOSSTextEncode, ApplyMOSSPromptWeights 等)
├── v_scaling.py               # DiT Attention V-Scaling 引擎、Query 分离与权重计算
├── text_processing.py         # 提示词括号权重解析、Danbooru 角色转义与 JSON 布局支持
├── pyproject.toml             # ComfyUI Manager 注册元数据
├── README.md                  # 中英双语完整文档与工作流说明
└── example_workflows/
    └── moss_v_scaling_workflow.json  # 示例一键导入工作流
```
