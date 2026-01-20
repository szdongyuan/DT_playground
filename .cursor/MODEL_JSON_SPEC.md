# 模型架构 JSON 生成规范

> **用途**: 本文档定义模型构建器的 `.model.json` 文件格式规范，用于指导 AI 生成符合平台要求的模型架构文件。

## 1. JSON 整体结构

```json
{
  "version": "1.0",
  "name": "模型名称",
  "description": "模型描述",
  "created_at": "2025-12-26T10:00:00.000000",
  "modified_at": "2025-12-26T10:00:00.000000",
  "compile_config": {
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "loss": "mse",
    "metrics": ["mae"]
  },
  "layers": [...],
  "connections": [...]
}
```

### 1.1 元数据字段

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| version | string | ✅ | 固定为 `"1.0"` |
| name | string | ✅ | 模型名称（英文，下划线分隔）|
| description | string | ✅ | 模型中文描述 |
| created_at | string | ✅ | ISO 8601 时间戳 |
| modified_at | string | ✅ | ISO 8601 时间戳 |
| compile_config | object | ✅ | 编译配置（冗余字段，实际从 Output 层读取）|

### 1.2 编译配置 (compile_config)

| 字段 | 类型 | 可选值 | 默认值 |
|------|------|--------|--------|
| optimizer | string | `Adam`, `SGD`, `RMSprop`, `AdamW`, `Nadam` | `Adam` |
| learning_rate | float | 0.000001 ~ 1.0 | 0.001 |
| loss | string | 见下表 | `mse` |
| metrics | array | 见下表 | `["mae"]` |

**损失函数选项**:
- 回归: `mse`, `mae`, `huber`
- 二分类: `binary_crossentropy`
- 多分类: `categorical_crossentropy`, `sparse_categorical_crossentropy`

**评估指标选项**: `accuracy`, `mae`, `mse`, `rmse`, `precision`, `recall`, `auc`, `cosine_similarity`

---

## 2. 节点布局规范 ⭐重要

### 2.0 布局核心原则 🚨必读

> **⛔ 禁止将所有节点排成一条直线！** 这对用户非常不友好。
>
> **✅ 必须充分利用二维画布**，让网络结构一目了然。

**核心原则**:

1. **功能分组**: 将功能相关的节点放在一起（如 Conv+BN+ReLU 组成一个"块"）
2. **阶段分行**: 不同阶段（编码器/解码器、特征提取/分类头）放在不同行
3. **垂直分层**: 利用 Y 轴表达网络的深度或阶段变化
4. **紧凑布局**: 关联紧密的节点间距可以更小（如 120px），阶段之间间距更大

### 2.1 节点尺寸

- **节点宽度**: 约 120px
- **节点高度**: 约 60-80px
- **端口直径**: 约 12px

### 2.2 间距规范

| 布局方向 | 最小间距 | 推荐间距 | 说明 |
|---------|---------|---------|------|
| 水平 (X) | 120px | **150px** | 同一行节点之间的水平间隔 |
| 垂直 (Y) | 80px | **100px** | 不同行/阶段之间的垂直间隔 |
| 阶段间隔 (X) | 200px | **250px** | 不同功能阶段之间的水平间隔 |

### 2.3 布局策略

#### ⭐ 推荐策略: 功能分组 + 多行布局（默认）

**所有网络**（即使是简单的顺序模型）都应该采用分组多行布局：

```
布局思路:
1. 将 Conv+BN+Activation 视为一个"卷积块"，紧密排列
2. 池化层作为阶段分隔，可以换行或增加间距
3. 分类头（Dense+Dropout）放在新的一行
4. 每行最多 4-6 个节点

示例: 12层分类网络的二维布局

第1行 (y=0):    Input → [Conv→BN→ReLU] → Pool
第2行 (y=100):  [Conv→BN→ReLU] → Pool  
第3行 (y=200):  [Conv→BN→ReLU] → GAP
第4行 (y=300):  Dense → Dropout → Dense → Output
```

```
┌─────┐   ┌─────┐  ┌────┐  ┌────┐   ┌────┐
│Input│──→│Conv1│─→│BN1 │─→│ReLU│──→│Pool│
└─────┘   └─────┘  └────┘  └────┘   └────┘
(0,0)     (140,0)  (260,0) (380,0)  (500,0)
                                       │
    ┌──────────────────────────────────┘
    ▼
┌─────┐  ┌────┐  ┌────┐   ┌────┐
│Conv2│─→│BN2 │─→│ReLU│──→│Pool│
└─────┘  └────┘  └────┘   └────┘
(0,100)  (120,100)(240,100)(360,100)
                              │
    ┌─────────────────────────┘
    ▼
┌─────┐  ┌────┐  ┌────┐   ┌────┐
│Conv3│─→│BN3 │─→│ReLU│──→│GAP │
└─────┘  └────┘  └────┘   └────┘
(0,200)  (120,200)(240,200)(360,200)
                              │
    ┌─────────────────────────┘
    ▼
┌──────┐   ┌───────┐   ┌──────┐   ┌──────┐
│Dense1│──→│Dropout│──→│Dense2│──→│Output│
└──────┘   └───────┘   └──────┘   └──────┘
(0,300)    (150,300)   (300,300)  (450,300)
```

#### 策略2: S形/蛇形布局（超长网络）

当节点超过 15 个时，采用蛇形布局：

```
每行最多 5-6 个节点
奇数行从左到右，偶数行从右到左
行间距: 120px

第1行 (y=0):   Input → Conv1 → BN1 → ReLU → Pool1 ──┐
                                                     │
第2行 (y=120): Pool2 ← ReLU ← BN2 ← Conv2 ←─────────┘
    │
    └─→ 第3行继续...
```

#### 策略3: 分支布局（残差/跳跃连接）

对于有分支结构的网络（如 U-Net、ResNet）：

```
主干在中间 y=0
分支向上 y=-100, -200, ...
分支向下 y=100, 200, ...

          ┌─────┐
          │Skip1│ (y=-100)
          └─────┘
              ▲
┌─────┐   ┌──┴──┐   ┌─────┐   ┌─────┐
│Input│──→│Conv │──→│Conv │──→│Merge│──→ Output
└─────┘   └─────┘   └─────┘   └─────┘
(y=0)                           ▲
              ┌─────┐           │
              │Skip2│───────────┘ (y=100)
              └─────┘
```

### 2.4 自编码器布局（U形/V形）

对于自编码器/U-Net结构，**必须**采用 U 形或 V 形布局：

```
编码器向右下延伸:
x 递增, y 递增

解码器向右上延伸:
x 递增, y 递减

瓶颈层在最低点

        Input                              Output
        (0,0)                             (1200,0)
          │                                   ▲
          ▼                                   │
      Enc_Conv1                          Dec_Conv1
      (150,50)                           (1050,50)
          │                                   ▲
          ▼                                   │
      Enc_Conv2                          Dec_Conv2
      (300,100)                          (900,100)
          │                                   ▲
          ▼                                   │
      Enc_Conv3 ─────→ Bottleneck ────→ Dec_Conv3
      (450,150)        (600,200)        (750,150)

示例坐标:
编码器: (0, 0) → (150, 50) → (300, 100) → (450, 150)
瓶颈:   (600, 200)
解码器: (750, 150) → (900, 100) → (1050, 50) → (1200, 0)
```

### 2.5 坐标计算公式

```python
# 功能分组布局
def calculate_position(stage: int, index_in_stage: int, is_classifier_head: bool = False):
    """
    stage: 阶段编号 (0, 1, 2, ...)
    index_in_stage: 阶段内节点索引 (0, 1, 2, ...)
    """
    y = stage * 100  # 每个阶段一行
    
    if is_classifier_head:
        x = index_in_stage * 150  # 分类头间距大一些
    else:
        x = index_in_stage * 120  # 卷积块内紧凑
    
    return (x, y)

# 自编码器 U 形布局
def calculate_unet_position(layer_index: int, total_encoder: int, total_decoder: int):
    """计算 U-Net 布局坐标"""
    if layer_index < total_encoder:
        # 编码器：向右下
        x = layer_index * 150
        y = layer_index * 50
    elif layer_index == total_encoder:
        # 瓶颈层
        x = total_encoder * 150
        y = total_encoder * 50 + 50
    else:
        # 解码器：向右上
        dec_idx = layer_index - total_encoder - 1
        x = (total_encoder + 1 + dec_idx) * 150
        y = (total_decoder - dec_idx - 1) * 50
    return (x, y)
```

---

## 3. 层节点 (layers) 定义

### 3.1 层节点结构

```json
{
  "id": "唯一ID",
  "type": "层类型",
  "position": [x, y],
  "parameters": {...},
  "layer_name": "可选的Keras层名称",
  "trainable": true
}
```

| 字段 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| id | string | ✅ | - | 唯一标识符，8位十六进制或语义化命名 |
| type | string | ✅ | - | 层类型标识（见下表）|
| position | [float, float] | ✅ | - | 画布坐标 [x, y] |
| parameters | object | ✅ | - | 层参数 |
| layer_name | string | ❌ | `""` | Keras 层名称（可选）|
| trainable | bool | ❌ | `true` | 层是否可训练 |

### 3.2 可训练性规范 🔧

> **默认所有层都是可训练的** (`trainable: true`)

- 生成模型时，**不需要显式指定** `trainable` 字段（默认为 `true`）
- 只有在需要**冻结层**（如迁移学习场景）时，才设置 `trainable: false`
- Input 和 Output 层没有可训练参数，该字段对其无效

### 3.3 ID 命名规范

推荐使用**语义化 ID** 提高可读性：

```
格式: {阶段}_{层类型}{编号}

示例:
- "input_01"
- "enc_conv1", "enc_conv2", "enc_bn1"
- "dec_conv1", "dec_bn1"
- "dense_out", "output_01"
```

---

## 4. 可用层类型速查表

### 4.1 输入/输出层

| type | 显示名称 | 参数 |
|------|---------|------|
| `input` | Input 输入层 | `shape`: "(H, W)" 或 "(L, C)", `dtype`: "float32" |
| `output` | Output 输出层 | `activation` (none/linear/sigmoid/softmax/tanh/relu), `optimizer`, `learning_rate`, `loss`, `metrics` |

### 4.2 核心层

| type | 显示名称 | 主要参数 |
|------|---------|---------|
| `dense` | Dense 全连接 | `units`, `activation` (none/relu/sigmoid等), `use_bias`, `kernel_initializer` |
| `embedding` | Embedding 嵌入 | `input_dim`, `output_dim`, `mask_zero` |

### 4.3 卷积层

| type | 显示名称 | 主要参数 |
|------|---------|---------|
| `conv1d` | Conv1D | `filters`, `kernel_size`, `strides`, `padding`, `activation` |
| `conv2d` | Conv2D | `filters`, `kernel_size`: "(H,W)", `strides`, `padding` |
| `conv1d_transpose` | Conv1DTranspose | 同 conv1d（用于上采样）|
| `conv2d_transpose` | Conv2DTranspose | 同 conv2d（用于上采样）|
| `separable_conv1d` | SeparableConv1D | `filters`, `kernel_size`, `strides`, `padding` |

### 4.4 池化层

| type | 显示名称 | 主要参数 |
|------|---------|---------|
| `max_pooling1d` | MaxPooling1D | `pool_size`, `strides`, `padding` |
| `max_pooling2d` | MaxPooling2D | `pool_size`: "(H,W)", `strides`, `padding` |
| `avg_pooling1d` | AveragePooling1D | `pool_size`, `strides`, `padding` |
| `global_max_pooling1d` | GlobalMaxPooling1D | 无参数 |
| `global_avg_pooling1d` | GlobalAveragePooling1D | 无参数 |
| `global_max_pooling2d` | GlobalMaxPooling2D | 无参数 |
| `global_avg_pooling2d` | GlobalAveragePooling2D | 无参数 |

### 4.5 归一化层

| type | 显示名称 | 主要参数 |
|------|---------|---------|
| `batch_norm` | BatchNormalization | `momentum`, `epsilon`, `center`, `scale` |
| `layer_norm` | LayerNormalization | `epsilon`, `center`, `scale` |

### 4.6 正则化层

| type | 显示名称 | 主要参数 |
|------|---------|---------|
| `dropout` | Dropout | `rate`: 0.0~1.0 |
| `spatial_dropout1d` | SpatialDropout1D | `rate` |
| `spatial_dropout2d` | SpatialDropout2D | `rate` |
| `gaussian_noise` | GaussianNoise | `stddev` |
| `gaussian_dropout` | GaussianDropout | `rate` |

### 4.7 激活函数层

| type | 显示名称 | 主要参数 |
|------|---------|---------|
| `activation` | Activation | `activation`: relu, sigmoid, tanh, softmax, gelu, swish等 |
| `leaky_relu` | LeakyReLU | `alpha` |
| `prelu` | PReLU | 无参数（自动学习）|
| `elu` | ELU | `alpha` |
| `softmax` | Softmax | `axis` |

### 4.8 形状变换层

| type | 显示名称 | 主要参数 |
|------|---------|---------|
| `flatten` | Flatten | 无参数 |
| `reshape` | Reshape | `target_shape`: "(H, W)" |
| `permute` | Permute | `dims`: "(2, 1)" |
| `repeat_vector` | RepeatVector | `n` |
| `upsampling1d` | UpSampling1D | `size` |
| `upsampling2d` | UpSampling2D | `size`: "(H, W)" |

### 4.9 循环层

| type | 显示名称 | 主要参数 |
|------|---------|---------|
| `lstm` | LSTM | `units`, `return_sequences`, `bidirectional`, `dropout` |
| `gru` | GRU | `units`, `return_sequences`, `bidirectional`, `dropout` |
| `simple_rnn` | SimpleRNN | `units`, `return_sequences`, `dropout` |

### 4.10 注意力层

| type | 显示名称 | 主要参数 |
|------|---------|---------|
| `multi_head_attention` | MultiHeadAttention | `num_heads`, `key_dim`, `value_dim`, `dropout` |
| `transformer_encoder` | TransformerEncoder | `num_heads`, `key_dim`, `ff_dim`, `dropout_rate`, `activation` |
| `transformer_decoder` | TransformerDecoder | `num_heads`, `key_dim`, `ff_dim`, `dropout_rate` |
| `positional_encoding` | PositionalEncoding | `max_length`, `encoding_type`, `dropout_rate` |
| `attention` | Attention | `use_scale`, `score_mode`, `dropout` |
| `additive_attention` | AdditiveAttention | `use_scale`, `dropout` |

### 4.11 合并层

| type | 显示名称 | 主要参数 |
|------|---------|---------|
| `concatenate` | Concatenate | `axis` |
| `add` | Add | 无参数 |
| `multiply` | Multiply | 无参数 |
| `average` | Average | 无参数 |
| `maximum` | Maximum | 无参数 |
| `minimum` | Minimum | 无参数 |
| `dot` | Dot | `axes`, `normalize` |

---

## 5. 连接定义 (connections)

```json
{
  "source": "源层ID",
  "target": "目标层ID"
}
```

### 5.1 连接规则

1. **每个层只有一个输入端口和一个输出端口**（合并层除外）
2. **不能创建循环连接**
3. **连接顺序应与拓扑排序一致**

### 5.2 连接顺序规范

按拓扑顺序排列，便于阅读：

```json
"connections": [
  {"source": "input_01", "target": "conv1"},
  {"source": "conv1", "target": "bn1"},
  {"source": "bn1", "target": "act1"},
  {"source": "act1", "target": "pool1"},
  ...
]
```

---

## 6. 常用模型模板

### 6.1 音频降噪自编码器 (1D)

**输入**: `(144000, 1)` - 3秒@48kHz 单声道
**任务**: 回归（input → denoised output）

```
结构: Input → [Conv1D+BN+ReLU]×3 → [Conv1DT+BN+ReLU]×3 → Output

编码器 (下采样):
  Conv1D(32, k=15, s=4) → BN → ReLU  (144000→36000)
  Conv1D(64, k=9, s=4)  → BN → ReLU  (36000→9000)
  Conv1D(128, k=5, s=4) → BN → ReLU  (9000→2250)

解码器 (上采样):
  Conv1DT(64, k=5, s=4)  → BN → ReLU  (2250→9000)
  Conv1DT(32, k=9, s=4)  → BN → ReLU  (9000→36000)
  Conv1DT(1, k=15, s=4, act=tanh)    (36000→144000)

编译配置:
  loss: mse
  metrics: mae
```

### 6.2 MFCC 二分类器

**输入**: `(281, 40)` - MFCC特征 (时间帧, 系数)
**任务**: 二分类

```
结构: Input → [Conv1D+BN+ReLU+Pool]×3 → GAP → Dense+Dropout → Dense(sigmoid) → Output

特征提取:
  Conv1D(32, k=3) → BN → ReLU → MaxPool(2)  (281→140)
  Conv1D(64, k=3) → BN → ReLU → MaxPool(2)  (140→70)
  Conv1D(128, k=3) → BN → ReLU              (70)
  GlobalAveragePooling1D                     (128)

分类头:
  Dense(64, relu) → Dropout(0.5) → Dense(1, sigmoid)

编译配置:
  loss: binary_crossentropy
  metrics: accuracy
```

### 6.3 多分类 CNN

**输入**: `(时间帧, 特征维度)`
**任务**: 多分类 (N类)

```
结构: Input → CNN特征提取 → Flatten → Dense → Softmax → Output

输出层:
  Dense(N, activation=softmax)

编译配置:
  loss: sparse_categorical_crossentropy 或 categorical_crossentropy
  metrics: accuracy
```

### 6.4 Transformer 编码器

**输入**: `(序列长度, 特征维度)`
**任务**: 序列分类

```
结构: Input → PositionalEncoding → TransformerEncoder×N → GAP → Dense → Output

推荐参数:
  num_heads: 4-8
  key_dim: 64
  ff_dim: 128-256
  dropout_rate: 0.1
```

---

## 7. 生成 Checklist ✅

生成模型 JSON 前，请检查：

**布局规范** 🎯
- [ ] ⛔ **禁止单一直线布局**（所有节点 y=0 是错误的！）
- [ ] ✅ 采用**多行二维布局**，按功能阶段分行
- [ ] 关联紧密的节点放在一起（Conv+BN+ReLU 一组）
- [ ] 节点间距 ≥ 120px (同组) 或 ≥ 150px (跨组)
- [ ] 行间距 ≥ 100px

**结构规范**
- [ ] 所有层 ID 唯一且语义化
- [ ] Input 层 shape 与任务匹配
- [ ] Output 层包含正确的 loss 和 metrics
- [ ] connections 按拓扑顺序排列
- [ ] 无循环连接
- [ ] 参数值在有效范围内

**可训练性**
- [ ] 默认所有层可训练（无需显式设置 trainable）
- [ ] 仅在冻结层时设置 `trainable: false`

---

## 8. 示例：分类器完整 JSON（二维布局）

下面的示例展示了正确的**二维布局**方式，将网络分为三个阶段行：

```
布局示意图:

第1行 (y=0):   Input → Conv1 → BN1 → ReLU1 → Pool1
                 │
                 └───────────────────────────────┐
                                                 ▼
第2行 (y=100): Conv2 → BN2 → ReLU2 → GAP ───────┐
                                                 │
                 ┌───────────────────────────────┘
                 ▼
第3行 (y=200): Dense1 → Dropout → Dense2 → Output
```

```json
{
  "version": "1.0",
  "name": "mfcc_binary_classifier",
  "description": "MFCC特征二分类器 - 二维布局示例",
  "created_at": "2025-12-26T10:00:00.000000",
  "modified_at": "2025-12-26T10:00:00.000000",
  "compile_config": {
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "loss": "binary_crossentropy",
    "metrics": ["accuracy"]
  },
  "layers": [
    {
      "id": "input_01",
      "type": "input",
      "position": [0.0, 0.0],
      "parameters": {"shape": "(100, 40)", "dtype": "float32"}
    },
    {
      "id": "conv1",
      "type": "conv1d",
      "position": [150.0, 0.0],
      "parameters": {"filters": 32, "kernel_size": 3, "strides": 1, "padding": "same", "activation": "linear", "use_bias": true}
    },
    {
      "id": "bn1",
      "type": "batch_norm",
      "position": [280.0, 0.0],
      "parameters": {"momentum": 0.99, "epsilon": 0.001}
    },
    {
      "id": "relu1",
      "type": "activation",
      "position": [400.0, 0.0],
      "parameters": {"activation": "relu"}
    },
    {
      "id": "pool1",
      "type": "max_pooling1d",
      "position": [520.0, 0.0],
      "parameters": {"pool_size": 2, "strides": 2, "padding": "valid"}
    },
    {
      "id": "conv2",
      "type": "conv1d",
      "position": [0.0, 100.0],
      "parameters": {"filters": 64, "kernel_size": 3, "strides": 1, "padding": "same", "activation": "linear", "use_bias": true}
    },
    {
      "id": "bn2",
      "type": "batch_norm",
      "position": [130.0, 100.0],
      "parameters": {"momentum": 0.99, "epsilon": 0.001}
    },
    {
      "id": "relu2",
      "type": "activation",
      "position": [250.0, 100.0],
      "parameters": {"activation": "relu"}
    },
    {
      "id": "gap",
      "type": "global_avg_pooling1d",
      "position": [370.0, 100.0],
      "parameters": {}
    },
    {
      "id": "dense1",
      "type": "dense",
      "position": [0.0, 200.0],
      "parameters": {"units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": "glorot_uniform"}
    },
    {
      "id": "dropout1",
      "type": "dropout",
      "position": [150.0, 200.0],
      "parameters": {"rate": 0.5}
    },
    {
      "id": "dense_out",
      "type": "dense",
      "position": [280.0, 200.0],
      "parameters": {"units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": "glorot_uniform"}
    },
    {
      "id": "output_01",
      "type": "output",
      "position": [420.0, 200.0],
      "parameters": {"activation": "linear", "optimizer": "Adam", "learning_rate": 0.001, "loss": "binary_crossentropy", "metrics": ["accuracy"]}
    }
  ],
  "connections": [
    {"source": "input_01", "target": "conv1"},
    {"source": "conv1", "target": "bn1"},
    {"source": "bn1", "target": "relu1"},
    {"source": "relu1", "target": "pool1"},
    {"source": "pool1", "target": "conv2"},
    {"source": "conv2", "target": "bn2"},
    {"source": "bn2", "target": "relu2"},
    {"source": "relu2", "target": "gap"},
    {"source": "gap", "target": "dense1"},
    {"source": "dense1", "target": "dropout1"},
    {"source": "dropout1", "target": "dense_out"},
    {"source": "dense_out", "target": "output_01"}
  ]
}
```

### 8.1 布局要点说明

| 阶段 | Y 坐标 | 节点 | 说明 |
|------|--------|------|------|
| 特征提取1 | 0 | Input, Conv1, BN1, ReLU1, Pool1 | 第一个卷积块 |
| 特征提取2 | 100 | Conv2, BN2, ReLU2, GAP | 第二个卷积块 + 全局池化 |
| 分类头 | 200 | Dense1, Dropout, Dense2, Output | 全连接分类层 |

**注意事项**:
- 同一行内节点 X 坐标递增，紧密相关的节点间距较小 (120px)
- 不同阶段通过 Y 坐标区分，阶段间距 100px
- 跨行连接时，下一行从 X=0 开始，形成清晰的视觉流

---

*最后更新: 2025-12-29*

