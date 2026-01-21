# AI Acoustic Signal Training Platform - Project Architecture

> **Maintenance Note**: Please update this file after any code structure changes

## Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| GUI Framework | PyQt6 (Catppuccin dark theme) |
| Node Editor | Custom PyQt6 implementation |
| Deep Learning | TensorFlow 2.15+ / Keras 3.0+ |
| Audio Processing | librosa, soundfile, scipy, pydub, sounddevice |
| Visualization | pyqtgraph, matplotlib |

## Core Concepts

### Workflow Mode

The platform uses a **node-based workflow** architecture, where users define complete data processing and training pipelines by dragging nodes and connecting ports.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Workflow Example: Denoising Model Training     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐                │
│   │AudioFolder│──┬─→│ AddNoise │────→│   Mel    │──→ [Input]     │
│   └──────────┘  │  └──────────┘     └──────────┘       │        │
│                 │                                       ▼        │
│                 │  ┌──────────┐                   ┌──────────┐  │
│                 └─→│   Mel    │──→ [Target] ────→│ Trainer  │  │
│                    └──────────┘                   └──────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Supported Task Types

| Task Type | Input | Output | Use Case |
|-----------|-------|--------|----------|
| Classification | Features | Labels (0/1/2...) | Sound classification, anomaly detection |
| Regression | Audio/Features | Audio/Features | Denoising, speech enhancement |
| Autoencoding | Audio/Features | Same data | Feature learning, compression |

## Directory Structure

```
DT_playground/
├── main.py                     # Application entry (logging config, Qt init, global exception handling)
├── requirements.txt            # Dependency list
├── .cursor/
│   ├── rules.mdc              # Cursor AI coding rules
│   └── ARCHITECTURE.md        # This file - project architecture documentation
├── audio_data/                 # Audio data directory
├── workflows/                  # Saved workflow files (JSON)
├── logs/                       # Runtime logs
├── models/                     # Saved trained models
├── resources/
│   └── styles/
│       └── dark_theme.qss     # QSS stylesheet
├── tests/
│   └── __init__.py
└── src/                        # Source code
    ├── __init__.py
    ├── app.py                  # AudioTrainingApp main application class
    │
    ├── core/                   # Core module (common abstractions)
    │   ├── __init__.py
    │   ├── parameter.py        # Parameter unified class (base for NodeParameter/LayerParameter)
    │   ├── event_bus.py        # EventBus global event bus (cross-component communication)
    │   └── graph_base.py       # GraphNodeBase, GraphBase graph structure base classes (documentation reference)
    │
    ├── controllers/            # Controller layer (frontend-backend separation)
    │   ├── __init__.py
    │   ├── workflow_controller.py   # WorkflowController (manages Engine lifecycle)
    │   ├── training_controller.py   # TrainingController
    │   └── navigation_controller.py # NavigationController (view navigation)
    │
    ├── workflow/               # Workflow engine
    │   ├── __init__.py
    │   ├── engine.py           # WorkflowEngine workflow execution engine
    │   ├── node_base.py        # BaseNode node base class
    │   ├── port.py             # Port, DataType port and data type definitions
    │   ├── connection.py       # Connection definition
    │   ├── workflow.py         # Workflow data model (serialization/deserialization)
    │   └── nodes/              # Node implementations
    │       ├── __init__.py
    │       ├── data_source.py      # Data source nodes (audio folder, label file, single audio)
    │       ├── preprocessing.py    # Preprocessing nodes (resample, trim, normalize, silence trim)
    │       ├── augmentation.py     # Data augmentation nodes (add noise, time stretch, pitch shift)
    │       ├── feature.py          # Feature extraction nodes (Mel, MFCC, STFT, statistics)
    │       ├── training.py         # Training nodes (trainer, model definition, evaluator)
    │       └── control.py          # Control nodes (loop, data split)
    │
    ├── audio/                  # Audio processing module
    │   ├── __init__.py
    │   ├── loader.py           # Audio loader
    │   ├── preprocessor.py     # Preprocessing (resample, trim, pad)
    │   ├── features.py         # FeatureExtractor (Mel/MFCC/STFT/chroma features)
    │   └── augmentation.py     # Data augmentation (time stretch, pitch shift, add noise)
    │
    ├── models/                 # Model definitions (deprecated, use model_builder/ instead)
    │   └── __init__.py
    │
    ├── model_builder/          # [NEW] Visual model builder
    │   ├── __init__.py
    │   ├── layer_base.py       # LayerNode layer node base class (with has_weights, trainable attributes)
    │   ├── model_graph.py      # ModelGraph model graph data structure (with freeze state application)
    │   ├── keras_parser.py     # KerasModelParser Keras model reverse parser
    │   └── layers/             # Layer node implementations
    │       ├── __init__.py
    │       ├── input_layers.py       # Input layers
    │       ├── core_layers.py        # Core layers (Dense, Embedding)
    │       ├── conv_layers.py        # Convolutional layers (Conv1D, Conv2D)
    │       ├── recurrent_layers.py   # Recurrent layers (LSTM, GRU)
    │       ├── attention_layers.py   # Attention layers (Transformer, MultiHeadAttention)
    │       ├── pooling_layers.py     # Pooling layers
    │       ├── normalization_layers.py  # Normalization layers
    │       ├── regularization_layers.py # Regularization layers (Dropout)
    │       ├── reshape_layers.py     # Shape transformation layers
    │       ├── activation_layers.py  # Activation function layers
    │       └── merge_layers.py       # Merge layers
    │
    ├── training/               # Training module
    │   ├── __init__.py
    │   ├── trainer.py          # TrainerWorker (QThread, uses callbacks.TrainingCallback)
    │   ├── callbacks.py        # TrainingCallback, EarlyStoppingWithUI (unified callback classes)
    │   ├── data_generator.py   # AudioDataGenerator (Keras Sequence)
    │   └── evaluator.py        # ModelEvaluator (confusion matrix, classification report)
    │
    ├── ui/                     # UI module
    │   ├── __init__.py
    │   ├── main_window.py      # MainWindow (multi-view switching layout)
    │   ├── styles.py           # Styles unified style management
    │   │
    │   ├── views/              # [NEW] View module
    │   │   ├── __init__.py
    │   │   ├── workflow_view.py       # WorkflowView node editor view
    │   │   ├── model_builder_view.py  # ModelBuilderView visual model building view
    │   │   ├── preview_view.py        # PreviewView data preview view (waveform/spectrogram)
    │   │   └── training_view.py       # TrainingView training monitoring view
    │   │
    │   ├── graph_editor/       # Graph editor base classes (reusable module)
    │   │   ├── __init__.py
    │   │   └── base_items.py       # BasePortItem, BaseConnectionItem, BaseGraphScene, BaseGraphView
    │   │
    │   ├── node_editor/        # Node editor components (workflow)
    │   │   ├── __init__.py
    │   │   ├── node_graph.py       # NodeGraphWidget node canvas (inherits BaseGraphView)
    │   │   ├── node_palette.py     # NodePalette node panel (draggable)
    │   │   └── property_panel.py   # PropertyPanel node property panel
    │   │
    │   ├── model_editor/       # Model editor components
    │   │   ├── __init__.py
    │   │   ├── layer_palette.py      # LayerPalette layer panel (draggable)
    │   │   ├── model_graph_widget.py # ModelGraphWidget model canvas (inherits BaseGraphView)
    │   │   └── layer_property_panel.py # LayerPropertyPanel layer property panel
    │   │
    │   ├── widgets/            # Custom widgets
    │   │   ├── __init__.py
    │   │   ├── waveform_widget.py      # WaveformWidget (waveform display)
    │   │   ├── spectrogram_widget.py   # SpectrogramWidget (spectrogram display)
    │   │   └── audio_player.py         # AudioPlayerWidget (playback control)
    │   │
    │   └── dialogs/            # Dialogs
    │       ├── __init__.py
    │       ├── settings_dialog.py
    │       ├── about_dialog.py
    │       ├── export_dialog.py
    │       └── dataset_dialog.py       # Dataset split dialog
    │
    ├── visualization/          # Visualization module
    │   ├── __init__.py
    │   ├── waveform.py
    │   ├── spectrogram.py
    │   └── metrics.py
    │
    └── utils/                  # Utility module
        ├── __init__.py
        ├── config.py           # ConfigManager singleton configuration management
        ├── dataset_manager.py  # DatasetManager (dataset splitting)
        ├── audio_utils.py
        ├── file_utils.py
        └── pyqtgraph_fix.py
```

## Node System Design

### Multi-channel Audio Support

The platform natively supports multi-channel audio processing, with data format unified to `(channels, samples)`:

| Input Type | Original Format | Unified Format | Description |
|------------|-----------------|----------------|-------------|
| Mono audio | `(N,)` | `(1, N)` | Auto-expanded to 2D |
| Stereo audio | `(2, N)` | `(2, N)` | Unchanged |
| Multi-channel audio | `(n_ch, N)` | `(n_ch, N)` | Unchanged |

#### AudioData Data Structure

```python
@dataclass
class AudioData:
    """Audio data wrapper - unified to (channels, samples) format"""
    data: np.ndarray        # Audio data, shape (channels, samples)
    sample_rate: int        # Sample rate
    file_path: str = ""     # Source file path
    duration: float = 0.0   # Duration (seconds)
    
    @property
    def channels(self) -> int:
        """Return number of channels"""
        return self.data.shape[0]
    
    @property
    def samples(self) -> int:
        """Return number of samples"""
        return self.data.shape[1]
    
    @property
    def is_mono(self) -> bool:
        """Check if mono"""
        return self.channels == 1
    
    def get_channel(self, channel_idx: int) -> np.ndarray:
        """Get data for specified channel (returns 1D array)"""
    
    def to_mono(self, method: str = "mean") -> 'AudioData':
        """Convert to mono"""
```

#### FeatureData Data Structure

```python
@dataclass
class FeatureData:
    """Feature data wrapper - multi-channel format"""
    data: np.ndarray        # Feature data
                            # - Time-frequency features: (channels, features, frames)
                            # - Statistical features: (channels, features)
    feature_type: str       # Feature type
    sample_rate: int        # Original sample rate
    hop_length: int         # Hop length
    source_file: str = ""   # Source file path
    
    @property
    def channels(self) -> int:
        """Return number of channels"""
```

#### Processing Strategy

All preprocessing, data augmentation, and feature extraction nodes use a **per-channel processing** strategy:

1. **Preprocessing nodes**: Apply the same processing to each channel separately (resample, normalize, etc.)
2. **Data augmentation nodes**: Augment each channel separately, using the same random parameters for consistency
3. **Feature extraction nodes**: Extract features from each channel separately, stack to `(channels, features, frames)`
4. **Silence trimming**: Use mixed signal to detect boundaries, apply the same trimming to all channels

```python
# Generic multi-channel processing function
def process_channels(data: np.ndarray, channel_func) -> np.ndarray:
    """
    Apply processing function to each channel separately
    
    Args:
        data: Audio data, shape (channels, samples)
        channel_func: Function to process a single channel, accepts 1D array returns 1D array
    
    Returns:
        Processed data, shape (channels, new_samples)
    """
```

### Data Types (DataType)

```python
class DataType(Enum):
    """All data types support single or batch (list) form, determined at runtime via isinstance"""
    AUDIO = "audio"           # Audio data (AudioData or List[AudioData])
    FEATURE_1D = "feature_1d" # 1D features (e.g., statistical features)
    FEATURE_2D = "feature_2d" # 2D features (e.g., Mel spectrogram)
    FEATURE = "feature"       # Generic feature type (compatible with 1D and 2D)
    LABEL = "label"           # Label data (single or list)
    MODEL = "model"           # Keras model
    METRICS = "metrics"       # Training/evaluation metrics
    ANY = "any"               # Any type (for generic nodes)
    TRIGGER = "trigger"       # Trigger signal (for control flow)
```

> **Design Notes**: 
> - The former `AUDIO_LIST`, `FEATURE_LIST`, `LABEL_LIST` have been merged into their base types
> - All types support single or batch form, determined at runtime via `isinstance(data, list)`
> - `ANY` type is bidirectionally compatible, allowing flexible connections (e.g., passthrough nodes, data split nodes)
> - Actual type validation is done at runtime in the node's `execute()` method via `validate_audio_input()`

### Node Categories

#### Data Source Nodes (DataSource)
| Node | Output Port | Description |
|------|-------------|-------------|
| 📂 AudioFolderNode | audio | Load all audio in directory (List[AudioData]) |
| 📄 LabelFileNode | labels | Load CSV/JSON label file |
| 🎵 AudioFileNode | audio | Load single audio file (AudioData) |

#### Preprocessing Nodes (Preprocessing)
| Node | Input | Output | Parameters |
|------|-------|--------|------------|
| 📏 ResampleNode | audio | audio | target_sr |
| ✂️ TrimPadNode | audio | audio | duration, mode |
| 📊 NormalizeNode | audio | audio | method (peak/rms) |
| 🎚️ SilenceTrimNode | audio | audio | threshold_db |

#### Data Augmentation Nodes (Augmentation)
| Node | Input | Output | Parameters |
|------|-------|--------|------------|
| 🔊 AddNoiseNode | audio | audio | noise_type, snr_db |
| ⏱️ TimeStretchNode | audio | audio | rate_range |
| 🎵 PitchShiftNode | audio | audio | semitones_range |
| 🔀 RandomAugmentNode | audio | audio | augmentations[] |

#### Feature Extraction Nodes (Feature)
| Node | Input | Output | Parameters |
|------|-------|--------|------------|
| 📈 MelSpectrogramNode | audio | feature_2d | n_mels, hop_length |
| 📊 MFCCNode | audio | feature_2d | n_mfcc, include_delta |
| 🎼 STFTNode | audio | feature_2d | n_fft, hop_length |
| 📉 StatisticsNode | audio | feature_1d | features[] |

> **Implementation Note**: All feature extraction nodes internally use `src.audio.features.FeatureExtractor` for feature computation to avoid code duplication.

#### Training Nodes (Training)
| Node | Input Ports | Output Ports | Description |
|------|-------------|--------------|-------------|
| 📥 LoadModelNode | - | model | Load model (supports .keras/.h5 and .model.json) |
| 📤 SaveModelNode | model | model_path | Save model |
| 🏋️ TrainerNode | input, target, model | trained_model | Execute training |
| 📊 EvaluatorNode | model, data | metrics | Evaluate model |

#### Control Nodes (Control)
| Node | Input | Output | Description |
|------|-------|--------|-------------|
| ◇ PassthroughNode | in | out | Passthrough node (compact size) |
| 🔄 LoopNode | data | item | Loop over dataset |
| ✂️ SplitNode | data | train, val, test | Dataset split |

### Unified Parameter Class

Workflow nodes and model layer nodes use unified parameter definitions (`src/core/parameter.py`):

```python
class ParamType(Enum):
    """Parameter types"""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    CHOICE = "choice"      # Dropdown selection
    FILE = "file"          # File selection
    FOLDER = "folder"      # Folder selection
    TUPLE = "tuple"        # Tuple (e.g., shape)
    LIST = "list"          # List

@dataclass
class Parameter:
    """Unified parameter definition"""
    name: str
    param_type: ParamType
    default: Any
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    default_directory: str = ""  # For FILE/FOLDER types

# Backward compatibility aliases
NodeParameter = Parameter   # Used by workflow/node_base.py
LayerParameter = Parameter  # Used by model_builder/layer_base.py
```

### Node Base Class

```python
class BaseNode:
    """Node base class"""
    node_id: str              # Unique identifier
    node_type: str            # Node type
    display_name: str         # Display name
    category: str             # Category (data_source/preprocessing/...)
    inputs: Dict[str, Port]   # Input ports
    outputs: Dict[str, Port]  # Output ports
    parameters: Dict          # Node parameters
    
    def execute(self, inputs: Dict) -> Dict:
        """Execute node logic, return outputs"""
        raise NotImplementedError
    
    def validate(self) -> Tuple[bool, str]:
        """Validate node configuration"""
        return True, ""
```

## UI Architecture

### Multi-View Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Menu Bar  │  File  │  Edit  │  View  │  Workflow  │  Help  │    │
├─────────────────────────────────────────────────────────────────┤
│  Toolbar  │ New │ Open │ Save │ ─── │ Run │ Stop │ ─── │ View Switch │
├──────────┬──────────────────────────────────────────┬───────────┤
│          │                                          │           │
│  Node    │           Main View Area                 │  Property │
│  Panel   │   (Switchable: Workflow/Model/Preview/   │  Panel    │
│          │    Training)                             │           │
│ ──────── │                                          │           │
│ Data     │                                          │  Node     │
│ Source   │                                          │  Parameter│
│ · Audio  │                                          │  Config   │
│ · Labels │                                          │           │
│ ──────── │                                          │ ───────── │
│ Preproc  │                                          │           │
│ · Resamp │                                          │  Preview  │
│ · Trim   │                                          │ (Selected │
│ ──────── │                                          │  node     │
│ Augment  │                                          │  output)  │
│ ──────── │                                          │           │
│ Features │                                          │           │
│ ──────── │                                          │           │
│ Training │                                          │           │
│          │                                          │           │
└──────────┴──────────────────────────────────────────┴───────────┘
```

### View Descriptions

| View | Components | Function |
|------|------------|----------|
| WorkflowView | NodeGraph + Nodes | Drag-and-drop workflow editing |
| ModelBuilderView | ModelGraph + Layer Nodes | Visual drag-and-drop neural network building |
| PreviewView | WaveformWidget + SpectrogramWidget | Preview audio/feature data |
| TrainingView | TrainingPanel + MetricsChart | Training progress and metrics monitoring |

### Graph Editor Base Classes

The workflow editor and model editor share common graph editing UI base classes (`src/ui/graph_editor/base_items.py`):

| Base Class | Inheritors | Description |
|------------|------------|-------------|
| BasePortItem | PortItem, LayerPortItem | Port graphics item (connection point) |
| BaseNodeItem | NodeItem, LayerItem | Node graphics item (draggable) |
| BaseConnectionItem | ConnectionItem, LayerConnectionItem | Connection line |
| BaseGraphScene | NodeGraphScene, ModelGraphScene | Graph scene (manages nodes and connections) |
| BaseGraphView | NodeGraphView, ModelGraphView | Graph view (zoom, pan, drag-drop) |

**Reusable Features**:
- Node/port rendering and interaction
- Connection line drawing (Bezier curves)
- Grid background drawing
- Mouse wheel zoom
- Middle-button pan
- Drag-drop support
- Selection highlighting

### Model Builder

The model builder provides visual neural network construction:

```
┌─────────────────────────────────────────────────────────────────┐
│  📐 Model Builder View                                           │
├──────────┬──────────────────────────────────────────┬───────────┤
│          │                                          │           │
│  Layer   │         Model Canvas                     │  Layer    │
│  Node    │   (Drag layer nodes, connect to build    │  Property │
│  Panel   │    network)                              │  Panel    │
│          │                                          │           │
│ ──────── │   ┌──────┐    ┌──────┐    ┌──────┐      │           │
│ Input    │   │Input │───→│Conv1D│───→│Dense │      │ Units     │
│ ──────── │   └──────┘    └──────┘    └──────┘      │ Activation│
│ Core     │                    │                     │ Initializer│
│ · Dense  │               ┌────┴────┐               │           │
│ ──────── │               │Dropout  │               │           │
│ Conv     │               └────┬────┘               │           │
│ · Conv1D │               ┌────┴────┐               │           │
│ · Conv2D │               │ Output  │               │           │
│ ──────── │               └─────────┘               │           │
│ Recurrent│                                          │           │
│ · LSTM   │  [New] [Open] [Save] [Build Model]      │           │
│ · GRU    │                                          │           │
│ ──────── │                                          │           │
│ Pooling  │                                          │           │
│ ──────── │                                          │           │
└──────────┴──────────────────────────────────────────┴───────────┘
```

#### Layer Node Categories

| Category | Layer Types | Description |
|----------|-------------|-------------|
| Input | Input | Define model input shape |
| Output | Output | Model output layer, includes compile config (optimizer, loss, metrics) |
| Core | Dense, Embedding | Fully connected layers, embedding layers |
| Convolutional | Conv1D, Conv2D, SeparableConv1D | 1D/2D convolutions |
| Recurrent | LSTM, GRU, SimpleRNN | Recurrent neural networks |
| Attention | MultiHeadAttention, TransformerEncoder, TransformerDecoder, PositionalEncoding | Transformer architecture |
| Pooling | MaxPooling, AveragePooling, GlobalPooling | Pooling operations |
| Normalization | BatchNormalization, LayerNormalization | Normalization layers |
| Regularization | Dropout, SpatialDropout, GaussianNoise | Overfitting prevention |
| Reshape | Flatten, Reshape, Permute, UpSampling | Shape operations |
| Activation | Activation, LeakyReLU, PReLU, Softmax | Activation layers |
| Merge | Concatenate, Add, Multiply, Average | Multi-input merge |

#### Usage Flow

1. Drag layer nodes to canvas in model view
2. Connect layer nodes (drag from output port to input port)
3. Configure layer parameters in property panel
4. Configure model compile options (optimizer, loss function, evaluation metrics)
5. Click "Build Model" to validate and generate Keras model
6. Save model definition file (*.model.json)
7. Use "Load Model" node in workflow to load (supports .model.json files)

#### Compile Configuration

Compile configuration is now **integrated in the Output layer**, select the Output layer in property panel to configure:

| Config Item | Options | Description |
|-------------|---------|-------------|
| Output Activation | linear, sigmoid, softmax, tanh, relu | Output layer activation |
| Optimizer | Adam, SGD, RMSprop, AdamW, Nadam | Training optimizer |
| Learning Rate | 0.000001 ~ 1.0 | Optimizer learning rate |
| Loss Function | mse, mae, huber, binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy | Loss function |
| Metrics | Comma-separated string, e.g., accuracy,mae | Evaluation metrics list |

Compile configuration is read from Output layer parameters and automatically applied when building the model.

#### Model Import and Fine-tuning

Supports importing architecture from trained Keras models for fine-tuning and transfer learning:

1. **Import Keras Model**: Click "📥 Import Keras" button, select `.keras` or `.h5` file
2. **View Layer Status**: Each layer shows weight status (✓ has weights) and freeze status (🔒 frozen)
3. **Freeze/Unfreeze Layers**: Check/uncheck "Trainable" checkbox in property panel's "Training Control" area
4. **Modify Architecture**: Can delete, add layers then rebuild
5. **Export Architecture**: Save as `.model.json` format

| LayerNode Attribute | Type | Description |
|---------------------|------|-------------|
| `has_weights` | bool | Whether has weights (auto-detected on import) |
| `trainable` | bool | Whether trainable (False = frozen) |

Freeze status is automatically applied to corresponding Keras layers when building the model.

### Signal Communication and Event Bus

#### Communication Conventions

| Scenario | Recommended Method | Description |
|----------|-------------------|-------------|
| Cross-component communication | EventBus | Between different views, Controller and multiple Views |
| Intra-component communication | pyqtSignal | Parent-child components, internal Widget elements |

#### EventBus Event Categories

| Category | Events | Description |
|----------|--------|-------------|
| Workflow | workflow_started, workflow_finished, workflow_error | Workflow lifecycle |
| Node | node_started, node_finished, node_progress | Node execution status |
| Breakpoint | breakpoint_hit, breakpoint_continue | Debug breakpoints |
| Training | training_started, training_epoch_completed, training_finished | Training progress |
| Status | status_message | Status bar messages |

#### Signal Flow Examples

```
Intra-component communication (pyqtSignal):
  NodePalette.node_dragged → NodeGraph.add_node
  PropertyPanel.parameter_changed → BaseNode.update_parameter

Cross-component communication (EventBus):
  WorkflowController → EventBus.workflow_started → TrainingView
  WorkflowController → EventBus.node_finished → WorkflowView.update_node_state

Frontend-backend separation (View → Controller → Engine):
  WorkflowView → WorkflowController.run() → WorkflowEngine.execute()
  WorkflowEngine (signals) → WorkflowController (forwards) → EventBus → Views
```

### Frontend-Backend Separation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         UI Layer (Views)                         │
│  WorkflowView │ ModelBuilderView │ PreviewView │ TrainingView   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ pyqtSignal (intra-component)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Controller Layer                             │
│  WorkflowController │ TrainingController │ NavigationController │
│  - Manage Engine lifecycle                                       │
│  - Forward Engine signals to EventBus                           │
│  - Handle business logic                                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Engine/Service Layer                         │
│  WorkflowEngine │ FeatureExtractor │ TrainerWorker              │
│  - Pure business logic, no UI dependency                        │
└─────────────────────────────────────────────────────────────────┘
```

**Design Principles**:
- View layer does not directly hold or operate Engine
- View accesses Engine indirectly through Controller
- Engine signals are forwarded to EventBus by Controller
- Cross-component communication uniformly uses EventBus

### Node Execution State Visualization

During workflow execution, nodes display different state markers:

| State | Color | Icon | Description |
|-------|-------|------|-------------|
| idle | No decoration | - | Not running |
| running | Yellow border | ⏳ | Currently executing |
| completed | Green border | ✅ | Execution complete |
| error | Red border | ❌ | Execution failed |
| waiting | Blue border | ⏸️ | Waiting at breakpoint |

When breakpoint triggers:
- Stay in workflow view
- Highlight breakpoint node
- Toolbar shows "🔴 Breakpoint Paused | ▶️ Continue Execution"

### Node Double-click Navigation Rules

When double-clicking a node, automatically navigate to corresponding view based on node type and execution state:

| Node Category | Node Type | Double-click Behavior |
|---------------|-----------|----------------------|
| Data Source | AudioFolderNode, AudioFileNode | ✅ Executed → Preview (audio) / ❌ Not executed → Prompt |
| Data Source | LabelFileNode | Show label count prompt |
| Preprocessing | ResampleNode, TrimPadNode, etc. | ✅ Executed → Preview (audio) / ❌ Not executed → Prompt |
| Augmentation | AddNoiseNode, etc. | ✅ Executed → Preview (audio) / ❌ Not executed → Prompt |
| Feature Extraction | MelSpectrogramNode, MFCCNode, etc. | ✅ Executed → Preview (2D graph) / ❌ Not executed → Prompt |
| Feature Extraction | StatisticsNode | ✅ Executed → Preview (1D curve) / ❌ Not executed → Prompt |
| Training | TrainerNode | Navigate directly to training view |
| Training | LoadModelNode | Navigate to model view |
| Training | SaveModelNode | Show save path / Not executed → Prompt |
| Training | EvaluatorNode, ShowMetricsNode | ✅ Executed → Preview (metrics) / ❌ Not executed → Prompt |
| Training | ShowHistoryNode | Navigate to training view |
| Control Flow | PassthroughNode, SplitNode | ✅ Executed → Preview / ❌ Not executed → Prompt |
| Control Flow | LoopNode | Show prompt (cannot preview) |

## Workflow Engine

### Execution Flow

```
1. Topological Sort - Determine execution order based on connections
2. Cycle Detection - Identify loop nodes and handle specially
3. Execute nodes sequentially:
   a. Collect input data (from upstream node outputs)
   b. Call node.execute(inputs)
   c. Cache output data
   d. Emit progress signals
4. Handle loop nodes - Repeat loop body execution
5. Completion/error handling
```

### Workflow Serialization

```json
{
  "version": "1.0",
  "name": "Denoising Model Training",
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

## Style System

Managed uniformly via `src/ui/styles.py`:

```python
from src.ui.styles import Styles

# Color definitions (Catppuccin Mocha)
Styles.COLORS['blue']   # #89b4fa
Styles.COLORS['green']  # #a6e3a1
Styles.COLORS['red']    # #f38ba8

# Node colors (by category)
Styles.NODE_COLORS = {
    'data_source': '#89b4fa',    # Blue
    'preprocessing': '#a6e3a1',  # Green
    'augmentation': '#f9e2af',   # Yellow
    'feature': '#cba6f7',        # Purple
    'training': '#f38ba8',       # Red
    'control': '#94e2d5',        # Cyan
}
```

## Type Validation Mechanism

### Runtime Type Validation

Since `ANY` type is bidirectionally compatible (allows flexible connections), actual type validation is performed at node execution time:

```python
# Validation function in preprocessing.py
def validate_audio_input(data, node_name: str) -> Union[AudioData, List[AudioData]]:
    """Validate input data is valid audio data"""
    if data is None:
        raise ValueError(f"{node_name}: No input audio provided")
    
    if isinstance(data, list):
        if not data:
            raise ValueError(f"{node_name}: Audio list is empty")
        if not isinstance(data[0], AudioData):
            raise TypeError(
                f"{node_name}: Expected AudioData type, "
                f"but received {type(data[0]).__name__}. "
                f"Please check upstream node output type."
            )
    else:
        if not isinstance(data, AudioData):
            raise TypeError(
                f"{node_name}: Expected AudioData type, "
                f"but received {type(data).__name__}. "
                f"Please check upstream node output type."
            )
    return data
```

### Usage

In the `execute()` method of nodes requiring specific input types:

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
    
    # ... continue processing
```

---
*Last Updated: 2026-01-21* (Architecture refactoring: unified parameter class, graph editor base class extraction, frontend-backend separation, EventBus standardization)
