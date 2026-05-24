# AI Acoustic Signal Training Platform - Project Architecture

> **Maintenance Note**: Please update this file after any code structure changes

## Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| GUI Framework | PySide6 (Catppuccin dark theme) |
| Node Editor | Custom PySide6 implementation |
| Deep Learning | TensorFlow 2.17+ / Keras |
| Audio Processing | librosa, soundfile, scipy, resampy, sounddevice |
| Visualization | pyqtgraph, matplotlib |
| i18n | gettext catalogs under `src/locale/` |

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

---

## Directory Structure

```
DT_playground/
├── main.py                     # Application entry (logging, Qt init, splash, i18n bootstrap, restart handshake)
├── requirements.txt            # Dependency list
├── .cursor/
│   ├── rules/                 # Rule documents
│   │   ├── rules.mdc          # Cursor AI rules entry point
│   │   ├── architecture.md    # This file
│   │   ├── coding_standards.md # Coding standards
│   │   └── model_json_spec.md # Model JSON specification
│   ├── workflows/             # Workflow definitions
│   │   └── git_workflow.md    # Git workflow
│   └── skills/                # Skill definitions
├── audio_data/                 # Audio data directory
├── workflows/                  # Saved workflow files (JSON)
├── logs/                       # Runtime logs
├── models/                     # Saved trained models
├── resources/
│   └── styles/
│       └── dark_theme.qss     # Legacy/optional QSS stylesheet; runtime styles are mostly programmatic
├── tests/                      # Automated tests (pytest and unittest style)
└── src/                        # Source code
    ├── __init__.py
    ├── app.py                  # AudioTrainingApp main application class
    │
    ├── core/                   # Core module (common abstractions)
    │   ├── __init__.py
    │   ├── parameter.py        # Parameter unified class (base for NodeParameter/LayerParameter)
    │   ├── event_bus.py        # EventBus global event bus (cross-component communication)
    │   └── graph_base.py       # GraphNodeBase, GraphBase graph structure base classes
    │
    ├── controllers/            # Controller layer (frontend-backend separation)
    │   ├── __init__.py
    │   ├── workflow_controller.py   # WorkflowController (manages Engine lifecycle)
    │   ├── training_controller.py   # TrainingController
    │   ├── navigation_controller.py # NavigationController (view navigation)
    │   ├── navigation_dto.py        # Navigation DTOs
    │   └── view_types.py            # View type enum/constants
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
    │       ├── data_source.py      # Data source nodes
    │       ├── preprocessing.py    # Preprocessing nodes
    │       ├── augmentation.py     # Data augmentation nodes
    │       ├── feature/            # Feature extraction node package
    │       │   ├── base.py         # FeatureData and feature helpers
    │       │   ├── ai_embedding.py # AI feature extraction node
    │       │   ├── dim1d/          # 1D feature nodes
    │       │   └── dim2d/          # 2D feature nodes
    │       ├── training.py         # Training nodes
    │       └── control.py          # Control nodes
    │
    ├── audio/                  # Audio processing module
    │   ├── __init__.py
    │   ├── loader.py           # Audio loader
    │   ├── preprocessor.py     # Preprocessing
    │   ├── features.py         # FeatureExtractor
    │   └── augmentation.py     # Data augmentation
    │
    ├── models/                 # Model definitions (deprecated, use model_builder/)
    │   └── __init__.py
    │
    ├── model_builder/          # Visual model builder
    │   ├── __init__.py
    │   ├── layer_base.py       # LayerNode layer node base class
    │   ├── model_graph.py      # ModelGraph model graph data structure
    │   ├── keras_parser.py     # KerasModelParser Keras model reverse parser
    │   └── layers/             # Layer node implementations
    │       ├── __init__.py
    │       ├── input_layers.py
    │       ├── core_layers.py
    │       ├── conv_layers.py
    │       ├── recurrent_layers.py
    │       ├── attention_layers.py
    │       ├── pooling_layers.py
    │       ├── normalization_layers.py
    │       ├── regularization_layers.py
    │       ├── reshape_layers.py
    │       ├── activation_layers.py
    │       └── merge_layers.py
    │
    ├── training/               # Training module
    │   ├── __init__.py
    │   ├── trainer.py          # TrainerWorker (QThread)
    │   ├── callbacks.py        # TrainingCallback, EarlyStoppingWithUI
    │   ├── data_generator.py   # AudioDataGenerator (Keras Sequence)
    │   └── evaluator.py        # ModelEvaluator
    │
    ├── ui/                     # UI module
    │   ├── __init__.py
    │   ├── main_window.py      # Embedded main content widget
    │   ├── styles.py           # Styles unified style management
    │   ├── i18n.py             # gettext helpers
    │   ├── startup_splash.py   # Startup splash and app icon path helpers
    │   ├── title_bar.py        # Custom frameless title bar
    │   │
    │   ├── views/              # View module
    │   │   ├── __init__.py
    │   │   ├── workflow_view.py
    │   │   ├── workflow_tabs_view.py
    │   │   ├── workflow_editor_widget.py
    │   │   ├── model_builder_view.py
    │   │   ├── preview_view.py
    │   │   ├── training_view.py
    │   │   └── toolbars/
    │   │       ├── workflow_toolbar.py
    │   │       └── model_toolbar.py
    │   │
    │   ├── graph_editor/       # Graph editor base classes
    │   │   ├── __init__.py
    │   │   ├── base_items.py
    │   │   └── clipboard.py
    │   │
    │   ├── node_editor/        # Node editor components
    │   │   ├── __init__.py
    │   │   ├── node_graph.py
    │   │   ├── node_palette.py
    │   │   └── property_panel.py
    │   │
    │   ├── model_editor/       # Model editor components
    │   │   ├── __init__.py
    │   │   ├── layer_palette.py
    │   │   ├── model_graph_widget.py
    │   │   └── layer_property_panel.py
    │   │
    │   ├── widgets/            # Custom widgets
    │   │   ├── __init__.py
    │   │   ├── command_toolbar.py
    │   │   ├── command_toolbar_spec.py
    │   │   ├── waveform_widget.py
    │   │   ├── spectrogram_widget.py
    │   │   ├── audio_player.py
    │   │   └── preview/        # Preview widgets for audio/features/model/metrics/labels
    │   │
    │   └── dialogs/            # Dialogs
    │       ├── __init__.py
    │       ├── settings_dialog.py
    │       ├── about_dialog.py
    │       ├── export_dialog.py
    │       └── dataset_dialog.py
    │
    ├── visualization/          # Visualization module
    │   ├── __init__.py
    │   ├── waveform.py
    │   ├── spectrogram.py
    │   └── metrics.py
    │
    ├── services/               # Application services
    │   ├── __init__.py
    │   ├── system_info_service.py
    │   └── training_params_service.py
    │
    ├── locale/                 # gettext catalogs
    │   ├── messages.pot
    │   ├── en_US/LC_MESSAGES/messages.po
    │   └── zh_CN/LC_MESSAGES/messages.po
    │
    └── utils/                  # Utility module
        ├── __init__.py
        ├── config.py           # ConfigManager singleton and module-level config
        ├── dataset_manager.py  # DatasetManager
        ├── audio_utils.py
        ├── dialogs.py
        ├── file_utils.py
        ├── restart_manager.py
        └── pyqtgraph_fix.py
```

---

## Design Patterns

### Overall Architecture (MVC + Controller)

```
┌─────────────────────────────────────────────────────────────┐
│                        View Layer (UI)                       │
│  src/ui/views/    src/ui/widgets/    src/ui/dialogs/        │
└─────────────────────────┬───────────────────────────────────┘
                          │ Signal/Slot
┌─────────────────────────▼───────────────────────────────────┐
│                    Controller Layer                          │
│  src/controllers/workflow_controller.py                      │
│  src/controllers/training_controller.py                      │
│  src/controllers/navigation_controller.py                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Service/Engine Layer                      │
│  src/workflow/engine.py    src/training/trainer.py          │
│  src/services/                                                │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      Model Layer                             │
│  src/workflow/    src/model_builder/    src/audio/          │
│  src/core/graph_base.py    src/core/parameter.py            │
└─────────────────────────────────────────────────────────────┘
```

### Core Design Patterns

#### 1. Factory + Registry Pattern

Used for dynamic creation of nodes and layers, supporting plugin-style extensions.

```python
# Register using decorator
@register_node
class MyNode(BaseNode):
    node_type = "my_node"
    ...

# Create via factory
node = create_node("my_node")
```

**Application locations**:
- `src/workflow/node_base.py`: `register_node`, `create_node`
- `src/model_builder/layer_base.py`: `register_layer`, `create_layer`

#### 2. Observer Pattern (via Signal/Slot)

Use PySide6 signal-slot mechanism for loosely coupled event notifications.

```python
class WorkflowEngine(QObject):
    # Define signals
    workflow_started = Signal()
    node_progress = Signal(str, float, str)  # node_id, progress, data
    
    def _execute_node(self, node):
        self.node_progress.emit(node.node_id, progress, data)
```

**Application locations**:
- `src/workflow/engine.py`: Workflow execution events
- `src/training/trainer.py`: Training progress events
- `src/core/event_bus.py`: Global event bus

#### 3. Template Method Pattern

Define algorithm skeleton, subclasses implement specific steps.

```python
class BaseNode(ABC):
    def __init__(self):
        self._setup_ports()        # Hook method - subclass must implement
        self._setup_parameters()   # Hook method - subclass optional
    
    @abstractmethod
    def execute(self) -> bool:     # Abstract method - subclass must implement
        pass
```

**Application locations**:
- `src/workflow/node_base.py`: BaseNode
- `src/model_builder/layer_base.py`: LayerNode
- `src/core/graph_base.py`: GraphNodeBase

#### 4. Strategy Pattern

Different node types implement different execution strategies.

```python
# Each node has its own execute implementation
class AudioFolderNode(BaseNode):
    def execute(self): ...  # Load audio files strategy

class MelSpectrogramNode(BaseNode):
    def execute(self): ...  # Mel spectrogram extraction strategy
```

#### 5. Singleton Pattern

Global unique instance for configuration management.

```python
# src/utils/config.py
class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Module-level singleton used by application code
config = ConfigManager()
```

#### 6. Composite Pattern

Graph structure manages node collections.

```python
class Workflow:
    nodes: Dict[str, BaseNode]           # Node collection
    connections: List[Connection]        # Connection collection
    
    def get_execution_order(self) -> List[str]:  # Topological sort
        ...
```

**Application locations**:
- `src/workflow/workflow.py`: Workflow
- `src/model_builder/model_graph.py`: ModelGraph

#### 7. Mediator Pattern (via EventBus)

Decouple communication between views.

```python
# src/core/event_bus.py
class EventBus(QObject):
    """Global event bus"""
    preview_requested = Signal(str, dict)    # node_id, data
    training_updated = Signal(int, dict)     # epoch, metrics
    
    @classmethod
    def instance(cls) -> 'EventBus':
        ...
```

#### 8. Dependency Injection

Inject dependencies through constructor for testing and decoupling.

```python
# ✅ Recommended: Inject dependencies
class WorkflowController:
    def __init__(self, engine: WorkflowEngine, event_bus: EventBus):
        self._engine = engine
        self._event_bus = event_bus

# ❌ Avoid: Create dependencies directly
class WorkflowController:
    def __init__(self):
        self._engine = WorkflowEngine()  # Tight coupling
```

### Common Abstraction Layer (src/core/)

To avoid duplicate code between `workflow` and `model_builder`, extract common base classes:

```python
# src/core/graph_base.py
class GraphNodeBase(ABC):
    """Graph node base class - inherited by BaseNode and LayerNode"""
    node_id: str
    parameters: Dict[str, Parameter]
    
class GraphBase(ABC):
    """Graph container base class - inherited by Workflow and ModelGraph"""
    nodes: Dict[str, GraphNodeBase]
    connections: List[Connection]
    
    def get_execution_order(self) -> List[str]:  # Common topological sort
        ...

# src/core/parameter.py
class ParamType(Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "str"
    BOOL = "bool"
    CHOICE = "choice"
    FILE = "file"
    FOLDER = "folder"

@dataclass
class Parameter:
    name: str
    param_type: ParamType
    default_value: Any
    ...
```

### Separation of Concerns Principles

1. **UI properties should not be in data models**
   - `position` and other UI properties managed by view layer or stored separately
   
2. **Controller handles coordination**
   - View only handles display logic
   - Controller handles user interaction and business dispatch
   - Engine/Service handles core business logic

3. **Single Responsibility for each class**
   - Avoid classes exceeding 300 lines
   - Split into multiple collaborating classes when functionality grows

---

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

1. **Preprocessing nodes**: Apply the same processing to each channel separately
2. **Data augmentation nodes**: Augment each channel separately, using the same random parameters
3. **Feature extraction nodes**: Extract features from each channel separately, stack to `(channels, features, frames)`
4. **Silence trimming**: Use mixed signal to detect boundaries, apply the same trimming to all channels

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
> - All types support single or batch form, determined at runtime via `isinstance(data, list)`
> - `ANY` type is bidirectionally compatible, allowing flexible connections
> - Actual type validation is done at runtime in the node's `execute()` method

### Node Categories

#### Data Source Nodes (DataSource)
| Node | Output Port | Description |
|------|-------------|-------------|
| 📂 AudioFolderNode | audio | Load all audio in directory (List[AudioData]) |
| 📄 LabelFileNode | labels | Load CSV/JSON label file |
| 🎵 AudioFileNode | audio | Load single audio file (AudioData) |
| 💾 SaveAudioNode | audio | Save audio data to disk |

#### Preprocessing Nodes (Preprocessing)
| Node | Input | Output | Parameters |
|------|-------|--------|------------|
| 📏 ResampleNode | audio | audio | target_sr |
| ✂️ TrimPadNode | audio | audio | duration, mode |
| 📊 NormalizeNode | audio | audio | method (peak/rms) |
| 🪟 WindowingNode | audio | audio | window_size, hop_size |
| 🧹 SpectralSubtractionNode | audio | audio | noise reduction parameters |
| 🎚️ SilenceTrimNode | audio | audio | threshold_db |
| 🔀 ChannelMapperNode | audio | audio | channel mapping |
| 🔗 ChannelMergeNode | audio | audio | merge method |
| 🎛️ FilterNode | audio | audio | filter type and cutoff |

#### Data Augmentation Nodes (Augmentation)
| Node | Input | Output | Parameters |
|------|-------|--------|------------|
| 🔊 AddNoiseNode | audio | audio | noise_type, snr_db |
| 📈 SampleExpansionNode | audio | audio | expansion strategy |
| 🔉 AdjustGainNode | audio | audio | gain_db |
| ⏱️ TimeStretchNode | audio | audio | rate_range |
| 🎵 PitchShiftNode | audio | audio | semitones_range |
| 🏛️ ReverbNode | audio, ir(optional) | audio | decay, wet_mix, random_wet_mix |
| ✂️ AudioSliceNode | audio | audio | slice duration/overlap |

#### Feature Extraction Nodes (Feature)
| Node | Input | Output | Parameters |
|------|-------|--------|------------|
| 📈 MelSpectrogramNode | audio | feature_2d | n_mels, hop_length |
| 📊 MFCCNode | audio | feature_2d | n_mfcc, include_delta |
| 🎼 STFTNode | audio | feature_2d | n_fft, hop_length |
| 🎹 CQTNode | audio | feature_2d | bins_per_octave, n_bins |
| 🌈 SpectralContrastNode | audio | feature_2d | n_bands |
| 📉 StatisticsNode | audio | feature_1d | features[] |
| 📊 FFTNode | audio | feature_1d | n_fft |
| 🎵 PitchNode | audio | feature_1d | method |
| 📐 SpectralFlatnessNode | audio | feature_1d | n_fft, hop_length |
| 🤖 AIFeatureExtractionNode | audio | feature | embedding/model options |

#### Training Nodes (Training)
| Node | Input Ports | Output Ports | Description |
|------|-------------|--------------|-------------|
| 📥 LoadModelNode | - | model | Load model (.keras/.h5 and .model.json) |
| 📤 SaveModelNode | model | model_path | Save model |
| 🏋️ TrainerNode | input, target, model | trained_model | Execute training |
| 📊 EvaluatorNode | model, data | metrics | Evaluate model |
| 📈 ShowHistoryNode | history | preview | Visualize training curves |
| 📋 ShowMetricsNode | metrics | preview | Display evaluation metrics |
| 🔮 PredictNode | model, data | predictions | Run prediction |

#### Control Nodes (Control)
| Node | Input | Output | Description |
|------|-------|--------|-------------|
| ◇ PassthroughNode | in | out | Passthrough node |
| 🔄 LoopNode | data | item | Loop over dataset |
| ✂️ SplitNode | data | train, val, test | Dataset split |
| 🔗 MergeNode | inputs | data | Merge streams |
| ⏸ BreakpointNode | data | data | Pause workflow execution |
| ✅ ValidateShapeNode | data | data | Validate tensor/audio shape |

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
    default_value: Any
    display_name: str = ""
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

---

## UI Architecture

### Multi-View Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Custom Title Bar / App Shell (AudioTrainingApp)                 │
├─────────────────────────────────────────────────────────────────┤
│  Grouped Toolbars: workflow and model actions                    │
├──────────┬──────────────────────────────────────────┬───────────┤
│          │                                          │           │
│  Node    │           Main View Area                 │  Property │
│  Panel   │   (Switchable: Workflow/Model/Preview/   │  Panel    │
│          │    Training)                             │           │
│ ──────── │                                          │           │
│ Data     │                                          │  Node     │
│ Source   │                                          │  Parameter│
│ ──────── │                                          │  Config   │
│ Preproc  │                                          │           │
│ ──────── │                                          │ ───────── │
│ Augment  │                                          │  Preview  │
│ ──────── │                                          │           │
│ Features │                                          │           │
│ ──────── │                                          │           │
│ Training │                                          │           │
│          │                                          │           │
└──────────┴──────────────────────────────────────────┴───────────┘
```

`AudioTrainingApp` (`src/app.py`) is the frameless `QMainWindow` shell. `MainWindow`
(`src/ui/main_window.py`) is the embedded content widget. Workflow editing is routed
through `WorkflowTabsView`, with each tab backed by a `WorkflowEditorWidget`.

### View Descriptions

| View | Components | Function |
|------|------------|----------|
| WorkflowTabsView | WorkflowEditorWidget + tabs | Multi-tab workflow editing |
| WorkflowView | WorkflowEditorWidget wrapper | Single workflow editing surface |
| ModelBuilderView | ModelGraph + Layer Nodes | Visual drag-and-drop neural network building |
| PreviewView | Preview widgets | Preview audio, features, model data, metrics, and labels |
| TrainingView | TrainingPanel + MetricsChart | Training progress and metrics monitoring |

### Graph Editor Base Classes

The workflow editor and model editor share common graph editing UI base classes (`src/ui/graph_editor/base_items.py`):

| Base Class | Inheritors | Description |
|------------|------------|-------------|
| BasePortItem | PortItem, LayerPortItem | Port graphics item (connection point) |
| BaseNodeItem | NodeItem, LayerItem | Node graphics item (draggable) |
| BaseConnectionItem | ConnectionItem, LayerConnectionItem | Connection line |
| BaseGraphScene | NodeGraphScene, ModelGraphScene | Graph scene |
| BaseGraphView | NodeGraphView, ModelGraphView | Graph view (zoom, pan, drag-drop) |

**Reusable Features**:
- Node/port rendering and interaction
- Connection line drawing (rounded orthogonal segments)
- Grid background drawing
- Mouse wheel zoom
- Middle-button pan
- Drag-drop support
- Selection highlighting
- Clipboard copy/paste helpers

### i18n Architecture

User-facing UI text uses gettext:

| Component | Location | Responsibility |
|-----------|----------|----------------|
| Runtime helpers | `src/ui/i18n.py` | `tr_`, `ngettext`, `pgettext`, and locale installation |
| Catalog source | `src/locale/messages.pot` | Extracted message template |
| Translations | `src/locale/<lang>/LC_MESSAGES/messages.po` | Editable translations |
| Compiled catalogs | `src/locale/<lang>/LC_MESSAGES/messages.mo` | Runtime gettext files |
| Tooling | `tools/i18n.ps1` | Extract, merge, compile workflow |

Source code should use English msgids wrapped with `tr_(...)`; Simplified Chinese is
provided by the `zh_CN` catalog rather than hardcoded in Python.

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
│ Core     │                    │                     │           │
│ ──────── │               ┌────┴────┐               │           │
│ Conv     │               │Dropout  │               │           │
│ ──────── │               └────┬────┘               │           │
│ Recurrent│               ┌────┴────┐               │           │
│ ──────── │               │ Output  │               │           │
│ Pooling  │               └─────────┘               │           │
│ ──────── │                                          │           │
│          │  [New] [Open] [Save] [Build Model]      │           │
└──────────┴──────────────────────────────────────────┴───────────┘
```

#### Layer Node Categories

| Category | Layer Types | Description |
|----------|-------------|-------------|
| Input | Input | Define model input shape |
| Output | Output | Model output layer, includes compile config |
| Core | Dense, Embedding | Fully connected layers |
| Convolutional | Conv1D, Conv2D, SeparableConv1D | 1D/2D convolutions |
| Recurrent | LSTM, GRU, SimpleRNN | Recurrent neural networks |
| Attention | MultiHeadAttention, TransformerEncoder/Decoder | Transformer |
| Pooling | MaxPooling, AveragePooling, GlobalPooling | Pooling |
| Normalization | BatchNormalization, LayerNormalization | Normalization |
| Regularization | Dropout, SpatialDropout, GaussianNoise | Overfitting prevention |
| Reshape | Flatten, Reshape, Permute, UpSampling | Shape operations |
| Activation | Activation, LeakyReLU, PReLU, Softmax | Activation layers |
| Merge | Concatenate, Add, Multiply, Average | Multi-input merge |

#### Model Import and Fine-tuning

Supports importing architecture from trained Keras models for fine-tuning:

1. **Import Keras Model**: Select `.keras` or `.h5` file
2. **View Layer Status**: Each layer shows weight status and freeze status
3. **Freeze/Unfreeze Layers**: Toggle "Trainable" checkbox in property panel
4. **Export Architecture**: Save as `.model.json` format

| LayerNode Attribute | Type | Description |
|---------------------|------|-------------|
| `has_weights` | bool | Whether has weights (auto-detected) |
| `trainable` | bool | Whether trainable (False = frozen) |

---

## Signal Communication

### Communication Conventions

| Scenario | Recommended Method | Description |
|----------|-------------------|-------------|
| Cross-component communication | EventBus | Between different views, Controller and Views |
| Intra-component communication | Signal | Parent-child components, internal elements |

### EventBus Event Categories

| Category | Events | Description |
|----------|--------|-------------|
| Workflow | workflow_started, workflow_finished, workflow_error | Workflow lifecycle |
| Node | node_started, node_finished, node_progress | Node execution status |
| Breakpoint | breakpoint_hit, breakpoint_continue | Debug breakpoints |
| Training | training_started, training_epoch_completed, training_finished | Training progress |
| Status | status_message | Status bar messages |

### Signal Flow Examples

```
Intra-component communication (Signal):
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
│  WorkflowTabsView │ ModelBuilderView │ PreviewView │ TrainingView│
└───────────────────────────┬─────────────────────────────────────┘
                            │ Signal (intra-component)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Controller Layer                             │
│  WorkflowController │ TrainingController │ NavigationController │
│  - Manage Engine lifecycle                                       │
│  - Forward Engine signals to EventBus                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Engine/Service Layer                         │
│  WorkflowEngine │ TrainerWorker │ src/services/                 │
│  - Pure business logic, no UI dependency                        │
└─────────────────────────────────────────────────────────────────┘
```

**Design Principles**:
- View layer does not directly hold or operate Engine
- View accesses Engine indirectly through Controller
- Engine signals are forwarded to EventBus by Controller
- Cross-component communication uniformly uses EventBus

---

## Node Execution State Visualization

During workflow execution, nodes display different state markers:

| State | Color | Icon | Description |
|-------|-------|------|-------------|
| idle | No decoration | - | Not running |
| running | Yellow border | ⏳ | Currently executing |
| completed | Green border | ✅ | Execution complete |
| error | Red border | ❌ | Execution failed |
| waiting | Blue border | ⏸️ | Waiting at breakpoint |
| skipped | Muted/disabled decoration | - | Execution skipped |

---

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

---

## Style System

Managed uniformly via `src/ui/styles.py`:

```python
from src.ui.styles import Styles

# Color definitions (Catppuccin Mocha)
Styles.COLORS['blue']   # #89b4fa
Styles.COLORS['green']  # #a6e3a1
Styles.COLORS['red']    # #f38ba8

# Node colors (by category)
# Workflow node category colors are defined by NodeCategory.color
# in src/workflow/node_base.py.
```

---

## Type Validation Mechanism

### Runtime Type Validation

Since `ANY` type is bidirectionally compatible, actual type validation is performed at node execution time:

```python
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
                f"but received {type(data[0]).__name__}."
            )
    else:
        if not isinstance(data, AudioData):
            raise TypeError(
                f"{node_name}: Expected AudioData type, "
                f"but received {type(data).__name__}."
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

*Last Updated: 2026-05-24*
