# AI声学信号训练平台 - 项目架构

> **维护说明**: 每次修改代码结构后请同步更新此文件

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| GUI框架 | PyQt6 (Catppuccin深色主题) |
| 节点编辑器 | 自定义 PyQt6 实现 |
| 深度学习 | TensorFlow 2.15+ / Keras 3.0+ |
| 音频处理 | librosa, soundfile, scipy, pydub, sounddevice |
| 可视化 | pyqtgraph, matplotlib |

## 核心概念

### 工作流模式

平台采用 **节点式工作流** 架构，用户通过拖拽节点、连接端口来定义完整的数据处理和训练流程。

```
┌─────────────────────────────────────────────────────────────────┐
│                        工作流示例：降噪模型训练                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐                │
│   │ 音频文件夹│──┬─→│ 添加噪声 │────→│ Mel频谱  │──→ [Input]     │
│   └──────────┘  │  └──────────┘     └──────────┘       │        │
│                 │                                       ▼        │
│                 │  ┌──────────┐                   ┌──────────┐  │
│                 └─→│ Mel频谱  │──→ [Target] ────→│  训练器  │  │
│                    └──────────┘                   └──────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 支持的任务类型

| 任务类型 | 输入 | 输出 | 用途 |
|---------|------|------|------|
| 分类 | 特征 | 标签 (0/1/2...) | 声音分类、异常检测 |
| 回归 | 音频/特征 | 音频/特征 | 降噪、语音增强 |
| 自编码 | 音频/特征 | 相同数据 | 特征学习、压缩 |

## 目录结构

```
DT_playground/
├── main.py                     # 应用入口 (日志配置, Qt初始化, 全局异常处理)
├── requirements.txt            # 依赖列表
├── .cursor/
│   ├── rules.mdc              # Cursor AI 编码规则
│   └── ARCHITECTURE.md        # 本文件 - 项目架构文档
├── audio_data/                 # 音频数据目录
├── workflows/                  # 保存的工作流文件 (JSON)
├── logs/                       # 运行日志
├── models/                     # 保存的训练模型
├── resources/
│   └── styles/
│       └── dark_theme.qss     # QSS样式文件
├── tests/
│   └── __init__.py
└── src/                        # 源代码
    ├── __init__.py
    ├── app.py                  # AudioTrainingApp 主应用类
    │
    ├── workflow/               # 【新增】工作流引擎
    │   ├── __init__.py
    │   ├── engine.py           # WorkflowEngine 工作流执行引擎
    │   ├── node_base.py        # BaseNode 节点基类
    │   ├── port.py             # Port, DataType 端口与数据类型定义
    │   ├── connection.py       # Connection 连接定义
    │   ├── workflow.py         # Workflow 工作流数据模型 (序列化/反序列化)
    │   └── nodes/              # 节点实现
    │       ├── __init__.py
    │       ├── data_source.py      # 数据源节点 (音频文件夹, 标签文件, 单个音频)
    │       ├── preprocessing.py    # 预处理节点 (重采样, 裁剪, 归一化, 静音裁剪)
    │       ├── augmentation.py     # 数据增强节点 (添加噪声, 时间拉伸, 音高偏移)
    │       ├── feature.py          # 特征提取节点 (Mel, MFCC, STFT, 统计特征)
    │       ├── training.py         # 训练节点 (训练器, 模型定义, 评估器)
    │       └── control.py          # 控制节点 (循环, 数据分割)
    │
    ├── audio/                  # 音频处理模块
    │   ├── __init__.py
    │   ├── loader.py           # 音频加载器
    │   ├── preprocessor.py     # 预处理 (重采样, 裁剪, 填充)
    │   ├── features.py         # FeatureExtractor (Mel/MFCC/STFT/色度特征)
    │   └── augmentation.py     # 数据增强 (时间拉伸, 音高偏移, 添加噪声)
    │
    ├── models/                 # 模型定义（已弃用，使用 model_builder/ 替代）
    │   └── __init__.py
    │
    ├── model_builder/          # 【新增】可视化模型构建器
    │   ├── __init__.py
    │   ├── layer_base.py       # LayerNode 层节点基类 (含 has_weights, trainable 属性)
    │   ├── model_graph.py      # ModelGraph 模型图数据结构 (含冻结状态应用)
    │   ├── keras_parser.py     # KerasModelParser Keras模型逆向解析器
    │   └── layers/             # 层节点实现
    │       ├── __init__.py
    │       ├── input_layers.py       # 输入层
    │       ├── core_layers.py        # 核心层 (Dense, Embedding)
    │       ├── conv_layers.py        # 卷积层 (Conv1D, Conv2D)
    │       ├── recurrent_layers.py   # 循环层 (LSTM, GRU)
    │       ├── attention_layers.py   # 注意力层 (Transformer, MultiHeadAttention)
    │       ├── pooling_layers.py     # 池化层
    │       ├── normalization_layers.py  # 归一化层
    │       ├── regularization_layers.py # 正则化层 (Dropout)
    │       ├── reshape_layers.py     # 形状变换层
    │       ├── activation_layers.py  # 激活函数层
    │       └── merge_layers.py       # 合并层
    │
    ├── training/               # 训练模块
    │   ├── __init__.py
    │   ├── trainer.py          # Trainer, TrainerWorker (QThread)
    │   ├── callbacks.py        # 训练回调 (UI进度更新, 早停)
    │   ├── data_generator.py   # AudioDataGenerator (Keras Sequence)
    │   └── evaluator.py        # ModelEvaluator (混淆矩阵, 分类报告)
    │
    ├── ui/                     # UI模块
    │   ├── __init__.py
    │   ├── main_window.py      # MainWindow (多视图切换布局)
    │   ├── styles.py           # Styles 统一样式管理
    │   │
    │   ├── views/              # 【新增】视图模块
    │   │   ├── __init__.py
    │   │   ├── workflow_view.py       # WorkflowView 节点编辑器视图
    │   │   ├── model_builder_view.py  # ModelBuilderView 可视化模型搭建视图
    │   │   ├── preview_view.py        # PreviewView 数据预览视图 (波形/频谱)
    │   │   └── training_view.py       # TrainingView 训练监控视图
    │   │
    │   ├── node_editor/        # 【新增】节点编辑器组件 (工作流)
    │   │   ├── __init__.py
    │   │   ├── node_graph.py       # NodeGraphWidget 节点画布 (PyQt6)
    │   │   ├── node_palette.py     # NodePalette 节点面板 (可拖拽)
    │   │   └── property_panel.py   # PropertyPanel 节点属性面板
    │   │
    │   ├── model_editor/       # 【新增】模型编辑器组件
    │   │   ├── __init__.py
    │   │   ├── layer_palette.py      # LayerPalette 层面板 (可拖拽)
    │   │   ├── model_graph_widget.py # ModelGraphWidget 模型画布
    │   │   └── layer_property_panel.py # LayerPropertyPanel 层属性面板
    │   │
    │   ├── widgets/            # 自定义控件
    │   │   ├── __init__.py
    │   │   ├── waveform_widget.py      # WaveformWidget (波形显示)
    │   │   ├── spectrogram_widget.py   # SpectrogramWidget (频谱图显示)
    │   │   └── audio_player.py         # AudioPlayerWidget (播放控制)
    │   │
    │   └── dialogs/            # 对话框
    │       ├── __init__.py
    │       ├── settings_dialog.py
    │       ├── about_dialog.py
    │       ├── export_dialog.py
    │       └── dataset_dialog.py       # 数据集划分对话框
    │
    ├── visualization/          # 可视化模块
    │   ├── __init__.py
    │   ├── waveform.py
    │   ├── spectrogram.py
    │   └── metrics.py
    │
    └── utils/                  # 工具模块
        ├── __init__.py
        ├── config.py           # ConfigManager 单例配置管理
        ├── dataset_manager.py  # DatasetManager (数据集划分)
        ├── audio_utils.py
        ├── file_utils.py
        └── pyqtgraph_fix.py
```

## 节点系统设计

### 多通道音频支持

平台原生支持多通道音频处理，数据格式统一为 `(channels, samples)`：

| 输入类型 | 原始格式 | 统一格式 | 说明 |
|---------|---------|---------|------|
| 单声道音频 | `(N,)` | `(1, N)` | 自动扩展为2D |
| 立体声音频 | `(2, N)` | `(2, N)` | 保持原样 |
| 多声道音频 | `(n_ch, N)` | `(n_ch, N)` | 保持原样 |

#### AudioData 数据结构

```python
@dataclass
class AudioData:
    """音频数据包装 - 统一为 (channels, samples) 格式"""
    data: np.ndarray        # 音频数据，形状为 (channels, samples)
    sample_rate: int        # 采样率
    file_path: str = ""     # 源文件路径
    duration: float = 0.0   # 时长（秒）
    
    @property
    def channels(self) -> int:
        """返回通道数"""
        return self.data.shape[0]
    
    @property
    def samples(self) -> int:
        """返回样本数"""
        return self.data.shape[1]
    
    @property
    def is_mono(self) -> bool:
        """是否为单声道"""
        return self.channels == 1
    
    def get_channel(self, channel_idx: int) -> np.ndarray:
        """获取指定通道的数据 (返回 1D 数组)"""
    
    def to_mono(self, method: str = "mean") -> 'AudioData':
        """转换为单声道"""
```

#### FeatureData 数据结构

```python
@dataclass
class FeatureData:
    """特征数据包装 - 多通道格式"""
    data: np.ndarray        # 特征数据
                            # - 时频特征: (channels, features, frames)
                            # - 统计特征: (channels, features)
    feature_type: str       # 特征类型
    sample_rate: int        # 原始采样率
    hop_length: int         # 帧移
    source_file: str = ""   # 源文件路径
    
    @property
    def channels(self) -> int:
        """返回通道数"""
```

#### 处理策略

所有预处理、数据增强、特征提取节点都采用 **分通道处理** 策略：

1. **预处理节点**: 对每个通道分别应用相同的处理（重采样、归一化等）
2. **数据增强节点**: 对每个通道分别增强，使用相同的随机参数保持一致性
3. **特征提取节点**: 对每个通道分别提取特征，堆叠为 `(channels, features, frames)`
4. **静音裁剪**: 使用混合信号检测边界，对所有通道应用相同的裁剪

```python
# 通用的多通道处理函数
def process_channels(data: np.ndarray, channel_func) -> np.ndarray:
    """
    对每个通道分别应用处理函数
    
    Args:
        data: 音频数据，形状为 (channels, samples)
        channel_func: 处理单个通道的函数，接受 1D 数组返回 1D 数组
    
    Returns:
        处理后的数据，形状为 (channels, new_samples)
    """
```

### 数据类型 (DataType)

```python
class DataType(Enum):
    """所有数据类型均支持单个或批量（列表）形式，运行时通过 isinstance 判断"""
    AUDIO = "audio"           # 音频数据 (AudioData 或 List[AudioData])
    FEATURE_1D = "feature_1d" # 1D特征 (如统计特征)
    FEATURE_2D = "feature_2d" # 2D特征 (如Mel频谱)
    FEATURE = "feature"       # 通用特征类型（兼容1D和2D）
    LABEL = "label"           # 标签数据 (单个或列表)
    MODEL = "model"           # Keras模型
    METRICS = "metrics"       # 训练/评估指标
    ANY = "any"               # 任意类型（用于通用节点）
    TRIGGER = "trigger"       # 触发信号（用于控制流）
```

> **设计说明**: 
> - 原有的 `AUDIO_LIST`、`FEATURE_LIST`、`LABEL_LIST` 已合并到对应的基础类型中
> - 所有类型均支持单个或批量形式，运行时通过 `isinstance(data, list)` 判断
> - `ANY` 类型双向兼容，允许灵活连接（如透传节点、数据分割节点）
> - 实际类型验证在节点的 `execute()` 方法中通过 `validate_audio_input()` 进行运行时检查

### 节点分类

#### 数据源节点 (DataSource)
| 节点 | 输出端口 | 说明 |
|------|----------|------|
| 📂 AudioFolderNode | audio | 加载目录下所有音频 (List[AudioData]) |
| 📄 LabelFileNode | labels | 加载 CSV/JSON 标签文件 |
| 🎵 AudioFileNode | audio | 加载单个音频文件 (AudioData) |

#### 预处理节点 (Preprocessing)
| 节点 | 输入 | 输出 | 参数 |
|------|------|------|------|
| 📏 ResampleNode | audio | audio | target_sr |
| ✂️ TrimPadNode | audio | audio | duration, mode |
| 📊 NormalizeNode | audio | audio | method (peak/rms) |
| 🎚️ SilenceTrimNode | audio | audio | threshold_db |

#### 数据增强节点 (Augmentation)
| 节点 | 输入 | 输出 | 参数 |
|------|------|------|------|
| 🔊 AddNoiseNode | audio | audio | noise_type, snr_db |
| ⏱️ TimeStretchNode | audio | audio | rate_range |
| 🎵 PitchShiftNode | audio | audio | semitones_range |
| 🔀 RandomAugmentNode | audio | audio | augmentations[] |

#### 特征提取节点 (Feature)
| 节点 | 输入 | 输出 | 参数 |
|------|------|------|------|
| 📈 MelSpectrogramNode | audio | feature_2d | n_mels, hop_length |
| 📊 MFCCNode | audio | feature_2d | n_mfcc, include_delta |
| 🎼 STFTNode | audio | feature_2d | n_fft, hop_length |
| 📉 StatisticsNode | audio | feature_1d | features[] |

#### 训练节点 (Training)
| 节点 | 输入端口 | 输出端口 | 说明 |
|------|----------|----------|------|
| 📥 LoadModelNode | - | model | 加载模型（支持.keras/.h5和.model.json） |
| 📤 SaveModelNode | model | model_path | 保存模型 |
| 🏋️ TrainerNode | input, target, model | trained_model | 执行训练 |
| 📊 EvaluatorNode | model, data | metrics | 评估模型 |

#### 控制节点 (Control)
| 节点 | 输入 | 输出 | 说明 |
|------|------|------|------|
| ◇ PassthroughNode | in | out | 透传节点（紧凑尺寸） |
| 🔄 LoopNode | data | item | 循环遍历数据集 |
| ✂️ SplitNode | data | train, val, test | 数据集划分 |

### 节点基类

```python
class BaseNode:
    """节点基类"""
    node_id: str              # 唯一标识
    node_type: str            # 节点类型
    display_name: str         # 显示名称
    category: str             # 分类 (data_source/preprocessing/...)
    inputs: Dict[str, Port]   # 输入端口
    outputs: Dict[str, Port]  # 输出端口
    parameters: Dict          # 节点参数
    
    def execute(self, inputs: Dict) -> Dict:
        """执行节点逻辑，返回输出"""
        raise NotImplementedError
    
    def validate(self) -> Tuple[bool, str]:
        """验证节点配置"""
        return True, ""
```

## UI架构

### 多视图布局

```
┌─────────────────────────────────────────────────────────────────┐
│  菜单栏  │  文件  │  编辑  │  视图  │  工作流  │  帮助  │        │
├─────────────────────────────────────────────────────────────────┤
│  工具栏  │ 新建 │ 打开 │ 保存 │ ─── │ 运行 │ 停止 │ ─── │ 视图切换 │
├──────────┬──────────────────────────────────────────┬───────────┤
│          │                                          │           │
│  节点    │           主视图区域                      │   属性    │
│  面板    │   (可切换: 工作流 / 模型 / 预览 / 训练)    │   面板    │
│          │                                          │           │
│ ──────── │                                          │           │
│ 数据源   │                                          │  节点参数  │
│ · 音频   │                                          │  配置区   │
│ · 标签   │                                          │           │
│ ──────── │                                          │ ───────── │
│ 预处理   │                                          │           │
│ · 重采样 │                                          │  预览区   │
│ · 裁剪   │                                          │ (选中节点  │
│ ──────── │                                          │  的输出)  │
│ 增强     │                                          │           │
│ ──────── │                                          │           │
│ 特征     │                                          │           │
│ ──────── │                                          │           │
│ 训练     │                                          │           │
│          │                                          │           │
└──────────┴──────────────────────────────────────────┴───────────┘
```

### 视图说明

| 视图 | 组件 | 功能 |
|------|------|------|
| WorkflowView | NodeGraph + 节点 | 拖拽编辑数据处理工作流 |
| ModelBuilderView | ModelGraph + 层节点 | 可视化拖拽搭建神经网络模型 |
| PreviewView | WaveformWidget + SpectrogramWidget | 预览音频/特征数据 |
| TrainingView | TrainingPanel + MetricsChart | 训练进度和指标监控 |

### 模型构建器

模型构建器提供可视化的神经网络搭建功能：

```
┌─────────────────────────────────────────────────────────────────┐
│  📐 模型构建器视图                                                │
├──────────┬──────────────────────────────────────────┬───────────┤
│          │                                          │           │
│  层节点  │         模型画布                          │   层属性  │
│  面板    │   (拖拽层节点，连接构建网络)               │   面板    │
│          │                                          │           │
│ ──────── │   ┌──────┐    ┌──────┐    ┌──────┐      │           │
│ 输入     │   │Input │───→│Conv1D│───→│Dense │      │ 神经元数  │
│ ──────── │   └──────┘    └──────┘    └──────┘      │ 激活函数  │
│ 核心层   │                    │                     │ 初始化器  │
│ · Dense  │               ┌────┴────┐               │           │
│ ──────── │               │Dropout  │               │           │
│ 卷积层   │               └────┬────┘               │           │
│ · Conv1D │               ┌────┴────┐               │           │
│ · Conv2D │               │ Output  │               │           │
│ ──────── │               └─────────┘               │           │
│ 循环层   │                                          │           │
│ · LSTM   │  [新建] [打开] [保存] [构建模型]         │           │
│ · GRU    │                                          │           │
│ ──────── │                                          │           │
│ 池化层   │                                          │           │
│ ──────── │                                          │           │
└──────────┴──────────────────────────────────────────┴───────────┘
```

#### 层节点分类

| 分类 | 层类型 | 说明 |
|------|--------|------|
| 输入 | Input | 定义模型输入形状 |
| 输出 | Output | 模型输出层，包含编译配置（优化器、损失函数、指标） |
| 核心层 | Dense, Embedding | 全连接层、嵌入层 |
| 卷积层 | Conv1D, Conv2D, SeparableConv1D | 一维/二维卷积 |
| 循环层 | LSTM, GRU, SimpleRNN | 循环神经网络 |
| 注意力层 | MultiHeadAttention, TransformerEncoder, TransformerDecoder, PositionalEncoding | Transformer架构 |
| 池化层 | MaxPooling, AveragePooling, GlobalPooling | 池化操作 |
| 归一化 | BatchNormalization, LayerNormalization | 归一化层 |
| 正则化 | Dropout, SpatialDropout, GaussianNoise | 防过拟合 |
| 形状变换 | Flatten, Reshape, Permute, UpSampling | 形状操作 |
| 激活函数 | Activation, LeakyReLU, PReLU, Softmax | 激活层 |
| 合并层 | Concatenate, Add, Multiply, Average | 多输入合并 |

#### 使用流程

1. 在模型视图中拖拽层节点到画布
2. 连接层节点（从输出端口拖到输入端口）
3. 在属性面板中配置层参数
4. 配置模型编译选项（优化器、损失函数、评估指标）
5. 点击"构建模型"验证并生成Keras模型
6. 保存模型定义文件 (*.model.json)
7. 在工作流中使用"加载模型"节点加载（支持 .model.json 文件）

#### 编译配置

编译配置现在**集成在 Output 输出层**中，在属性面板中选择 Output 层即可配置：

| 配置项 | 可选值 | 说明 |
|--------|--------|------|
| 输出激活函数 | linear, sigmoid, softmax, tanh, relu | 输出层激活函数 |
| 优化器 | Adam, SGD, RMSprop, AdamW, Nadam | 训练优化器 |
| 学习率 | 0.000001 ~ 1.0 | 优化器学习率 |
| 损失函数 | mse, mae, huber, binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy | 损失函数 |
| 评估指标 | 逗号分隔的字符串，如 accuracy,mae | 评估指标列表 |

编译配置从 Output 层参数读取，构建模型时自动应用。

#### 模型导入与微调

支持从已训练的 Keras 模型导入架构，用于模型微调和迁移学习：

1. **导入 Keras 模型**: 点击"📥 导入Keras"按钮，选择 `.keras` 或 `.h5` 文件
2. **查看层状态**: 每个层显示权重状态（✓ 有权重）和冻结状态（🔒 已冻结）
3. **冻结/解冻层**: 在属性面板的"训练控制"区域勾选/取消"可训练"复选框
4. **修改架构**: 可删除、添加层后重新构建
5. **导出架构**: 保存为 `.model.json` 格式

| LayerNode 属性 | 类型 | 说明 |
|---------------|------|------|
| `has_weights` | bool | 是否有权重（导入时自动检测）|
| `trainable` | bool | 是否可训练（False = 冻结）|

构建模型时，冻结状态会自动应用到对应的 Keras 层。

### 信号通信

```
NodePalette.node_dragged → NodeGraph.add_node
NodeGraph.node_selected → PropertyPanel.show_properties
NodeGraph.node_selected → PreviewView.preview_node_output (如果在预览模式)
NodeGraph.node_double_clicked → MainWindow._on_node_double_clicked (节点双击跳转)

PropertyPanel.parameter_changed → BaseNode.update_parameter
                               → NodeGraph.mark_dirty

WorkflowView.run_clicked → WorkflowEngine.execute
WorkflowEngine.node_started → WorkflowView.update_node_state(running)
WorkflowEngine.node_finished → WorkflowView.update_node_state(completed/error)
WorkflowEngine.breakpoint_hit → WorkflowView.show_breakpoint_mode(True)
WorkflowEngine.workflow_finished → TrainingView.show_results
WorkflowView.continue_requested → WorkflowEngine.continue_from_breakpoint
```

### 节点执行状态可视化

工作流执行时，节点会显示不同的状态标记：

| 状态 | 颜色 | 图标 | 说明 |
|-----|------|-----|------|
| idle | 无装饰 | - | 未运行 |
| running | 黄色边框 | ⏳ | 正在执行 |
| completed | 绿色边框 | ✅ | 执行完成 |
| error | 红色边框 | ❌ | 执行失败 |
| waiting | 蓝色边框 | ⏸️ | 断点等待中 |

断点触发时：
- 保持在工作流视图
- 高亮断点节点
- 工具栏显示「🔴 断点暂停中 | ▶️ 继续执行」

### 节点双击跳转规则

双击节点时，根据节点类型和运行状态自动跳转到相应视图：

| 节点分类 | 节点类型 | 双击行为 |
|---------|---------|---------|
| 数据源 | AudioFolderNode, AudioFileNode | ✅已运行→预览(音频) / ❌未运行→提示 |
| 数据源 | LabelFileNode | 显示标签数量提示 |
| 预处理 | ResampleNode, TrimPadNode等 | ✅已运行→预览(音频) / ❌未运行→提示 |
| 数据增强 | AddNoiseNode等 | ✅已运行→预览(音频) / ❌未运行→提示 |
| 特征提取 | MelSpectrogramNode, MFCCNode等 | ✅已运行→预览(2D图) / ❌未运行→提示 |
| 特征提取 | StatisticsNode | ✅已运行→预览(1D曲线) / ❌未运行→提示 |
| 训练 | TrainerNode | 直接跳转训练视图 |
| 训练 | LoadModelNode | 跳转模型视图 |
| 训练 | SaveModelNode | 显示保存路径 / 未运行→提示 |
| 训练 | EvaluatorNode, ShowMetricsNode | ✅已运行→预览(指标) / ❌未运行→提示 |
| 训练 | ShowHistoryNode | 跳转训练视图 |
| 控制流 | PassthroughNode, SplitNode | ✅已运行→预览 / ❌未运行→提示 |
| 控制流 | LoopNode | 显示提示（无法预览）|

## 工作流引擎

### 执行流程

```
1. 拓扑排序 - 根据连接关系确定执行顺序
2. 循环检测 - 识别循环节点并特殊处理
3. 逐节点执行:
   a. 收集输入数据 (来自上游节点的输出)
   b. 调用 node.execute(inputs)
   c. 缓存输出数据
   d. 发送进度信号
4. 处理循环节点 - 重复执行循环体
5. 完成/错误处理
```

### 工作流序列化

```json
{
  "version": "1.0",
  "name": "降噪模型训练",
  "nodes": [
    {
      "id": "node_1",
      "type": "AudioFolderNode",
      "position": [100, 100],
      "parameters": {
        "folder_path": "audio_data/clean"
      }
    },
    {
      "id": "node_2",
      "type": "AddNoiseNode",
      "position": [300, 50],
      "parameters": {
        "noise_type": "gaussian",
        "snr_db": 10
      }
    }
  ],
  "connections": [
    {
      "source": {"node": "node_1", "port": "audio_list"},
      "target": {"node": "node_2", "port": "audio"}
    }
  ]
}
```

## 样式系统

使用 `src/ui/styles.py` 统一管理:

```python
from src.ui.styles import Styles

# 颜色定义 (Catppuccin Mocha)
Styles.COLORS['blue']   # #89b4fa
Styles.COLORS['green']  # #a6e3a1
Styles.COLORS['red']    # #f38ba8

# 节点颜色 (按分类)
Styles.NODE_COLORS = {
    'data_source': '#89b4fa',    # 蓝色
    'preprocessing': '#a6e3a1',  # 绿色
    'augmentation': '#f9e2af',   # 黄色
    'feature': '#cba6f7',        # 紫色
    'training': '#f38ba8',       # 红色
    'control': '#94e2d5',        # 青色
}
```

## 类型验证机制

### 运行时类型验证

由于 `ANY` 类型双向兼容（允许灵活连接），实际类型验证在节点执行时进行：

```python
# preprocessing.py 中的验证函数
def validate_audio_input(data, node_name: str) -> Union[AudioData, List[AudioData]]:
    """验证输入数据是否为有效的音频数据"""
    if data is None:
        raise ValueError(f"{node_name}: 未提供输入音频")
    
    if isinstance(data, list):
        if not data:
            raise ValueError(f"{node_name}: 音频列表为空")
        if not isinstance(data[0], AudioData):
            raise TypeError(
                f"{node_name}: 期望 AudioData 类型，"
                f"但收到 {type(data[0]).__name__}。"
                f"请检查上游节点的输出类型是否正确。"
            )
    else:
        if not isinstance(data, AudioData):
            raise TypeError(
                f"{node_name}: 期望 AudioData 类型，"
                f"但收到 {type(data).__name__}。"
                f"请检查上游节点的输出类型是否正确。"
            )
    return data
```

### 使用方式

在需要特定类型输入的节点的 `execute()` 方法中：

```python
def execute(self) -> bool:
    try:
        audio_input = validate_audio_input(
            self.get_input_data("audio"),
            self.display_name
        )
    except (ValueError, TypeError) as e:
        self.error_message = str(e)
        return False
    
    # ... 继续处理
```

---
*最后更新: 2025-12-25* (添加多通道音频支持: AudioData/FeatureData统一为channels-first格式，所有节点分通道处理)
