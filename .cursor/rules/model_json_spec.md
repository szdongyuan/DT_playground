# Model Architecture JSON Generation Specification

> **Purpose**: This document defines the `.model.json` file format specification for the model builder, used to guide AI in generating model architecture files that meet platform requirements.

## 1. JSON Overall Structure

```json
{
  "version": "1.0",
  "name": "model_name",
  "description": "Model description",
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

### 1.1 Metadata Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| version | string | ✅ | Fixed as `"1.0"` |
| name | string | ✅ | Model name (English, underscore-separated) |
| description | string | ✅ | Model description |
| created_at | string | ✅ | ISO 8601 timestamp |
| modified_at | string | ✅ | ISO 8601 timestamp |
| compile_config | object | ✅ | Legacy/root compile configuration; used as fallback when no Output layer provides compile settings |

Root metadata is accepted leniently on load: missing `description`, `created_at`, or
`modified_at` values are filled with defaults by the loader. AI-generated files should
still include them for readability and repeatability.

### 1.2 Compile Configuration (compile_config)

| Field | Type | Options | Default |
|-------|------|---------|---------|
| optimizer | string | `Adam`, `SGD`, `RMSprop`, `AdamW`, `Nadam` | `Adam` |
| learning_rate | float | 0.000001 ~ 1.0 | 0.001 |
| loss | string | See table below | `mse` |
| metrics | array | See table below | `["mae"]` |

**Loss Function Options**:
- Regression: `mse`, `mae`, `huber`
- Binary Classification: `binary_crossentropy`
- Multi-class Classification: `categorical_crossentropy`, `sparse_categorical_crossentropy`

**Metric Options**: `accuracy`, `mae`, `mse`, `rmse`, `precision`, `recall`, `auc`, `cosine_similarity`

**Compile precedence**:

1. If an `output` layer exists, runtime model compilation reads optimizer, learning rate, loss, and metrics from that layer's `parameters`.
2. The root `compile_config` remains part of the saved format for compatibility and fallback.
3. Saved files can contain a stale root `compile_config` if Output layer settings were changed; prefer keeping both in sync when generating JSON.

---

## 2. Node Layout Specification ⭐ Important

### 2.0 Core Layout Principles 🚨 Must Read

> **⛔ Do NOT arrange all nodes in a single straight line when authoring JSON manually.** This is very unfriendly to users.
>
> **✅ Must fully utilize the 2D canvas**, making the network structure clear at a glance.

**Core Principles**:

1. **Functional Grouping**: Place functionally related nodes together (e.g., Conv+BN+ReLU form a "block")
2. **Stage Rows**: Different stages (encoder/decoder, feature extraction/classifier head) on different rows
3. **Vertical Layering**: Use Y-axis to express network depth or stage changes
4. **Compact Layout**: Closely related nodes can have smaller spacing (e.g., 120px), larger spacing between stages

Layout rules in this section are authoring guidance for generated `.model.json` files.
The current serializer/validator does not enforce multi-row layout, semantic IDs, or
connection ordering. Imported Keras models are initially arranged in a single row by
`KerasModelParser` and may need manual cleanup.

### 2.1 Node Dimensions

- **Node width**: ~180px (`LayerItem.LAYER_WIDTH`)
- **Node height**: ~60px (`LayerItem.LAYER_HEIGHT`)
- **Port diameter**: ~12px

### 2.2 Spacing Specification

| Layout Direction | Minimum Spacing | Recommended Spacing | Description |
|------------------|-----------------|---------------------|-------------|
| Horizontal (X) | 120px | **150px** | Horizontal spacing between nodes in same row |
| Vertical (Y) | 80px | **100px** | Vertical spacing between different rows/stages |
| Stage Gap (X) | 200px | **250px** | Horizontal spacing between different functional stages |

### 2.3 Layout Strategies

#### ⭐ Recommended Strategy: Functional Grouping + Multi-Row Layout (Default)

**All networks** (even simple sequential models) should use grouped multi-row layout:

```
Layout Approach:
1. Treat Conv+BN+Activation as a "conv block", arrange tightly
2. Pooling layers serve as stage separators, can break line or increase spacing
3. Classifier head (Dense+Dropout) on a new row
4. Maximum 4-6 nodes per row

Example: 12-layer classification network 2D layout

Row 1 (y=0):    Input → [Conv→BN→ReLU] → Pool
Row 2 (y=100):  [Conv→BN→ReLU] → Pool  
Row 3 (y=200):  [Conv→BN→ReLU] → GAP
Row 4 (y=300):  Dense → Dropout → Dense → Output
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

#### Strategy 2: S-Shape/Snake Layout (Very Long Networks)

When nodes exceed 15, use snake layout:

```
Maximum 5-6 nodes per row
Odd rows left-to-right, even rows right-to-left
Row spacing: 120px

Row 1 (y=0):   Input → Conv1 → BN1 → ReLU → Pool1 ──┐
                                                     │
Row 2 (y=120): Pool2 ← ReLU ← BN2 ← Conv2 ←─────────┘
    │
    └─→ Row 3 continues...
```

#### Strategy 3: Branch Layout (Residual/Skip Connections)

For networks with branch structures (e.g., U-Net, ResNet):

```
Main path in middle y=0
Branches upward y=-100, -200, ...
Branches downward y=100, 200, ...

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

### 2.4 Autoencoder Layout (U-Shape/V-Shape)

For autoencoder/U-Net structures, **must** use U-shape or V-shape layout:

```
Encoder extends downward-right:
x increasing, y increasing

Decoder extends upward-right:
x increasing, y decreasing

Bottleneck layer at lowest point

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

Example coordinates:
Encoder: (0, 0) → (150, 50) → (300, 100) → (450, 150)
Bottleneck: (600, 200)
Decoder: (750, 150) → (900, 100) → (1050, 50) → (1200, 0)
```

### 2.5 Coordinate Calculation Formulas

```python
# Functional grouping layout
def calculate_position(stage: int, index_in_stage: int, is_classifier_head: bool = False):
    """
    stage: Stage number (0, 1, 2, ...)
    index_in_stage: Node index within stage (0, 1, 2, ...)
    """
    y = stage * 100  # One row per stage
    
    if is_classifier_head:
        x = index_in_stage * 150  # Larger spacing for classifier head
    else:
        x = index_in_stage * 120  # Compact within conv block
    
    return (x, y)

# Autoencoder U-shape layout
def calculate_unet_position(layer_index: int, total_encoder: int, total_decoder: int):
    """Calculate U-Net layout coordinates"""
    if layer_index < total_encoder:
        # Encoder: downward-right
        x = layer_index * 150
        y = layer_index * 50
    elif layer_index == total_encoder:
        # Bottleneck layer
        x = total_encoder * 150
        y = total_encoder * 50 + 50
    else:
        # Decoder: upward-right
        dec_idx = layer_index - total_encoder - 1
        x = (total_encoder + 1 + dec_idx) * 150
        y = (total_decoder - dec_idx - 1) * 50
    return (x, y)
```

---

## 3. Layer Node (layers) Definition

### 3.1 Layer Node Structure

```json
{
  "id": "unique_id",
  "type": "layer_type",
  "position": [x, y],
  "parameters": {...},
  "layer_name": "optional_keras_layer_name",
  "trainable": true,
  "has_weights": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| id | string | ✅ | - | Unique identifier, 8-digit hex or semantic naming |
| type | string | ✅ | - | Layer type identifier (see table below) |
| position | [float, float] | ✅ | - | Canvas coordinates [x, y] |
| parameters | object | ✅ | - | Layer parameters |
| layer_name | string | ❌ | `""` | Keras layer name (optional) |
| trainable | bool | ❌ | `true` | Whether layer is trainable |
| has_weights | bool | ❌ | `false` | Whether an imported layer has weights; serialized only when true |

### 3.2 Trainability Specification 🔧

> **By default all layers are trainable** (`trainable: true`)

- When generating models, **no need to explicitly specify** `trainable` field (defaults to `true`)
- Only set `trainable: false` when **freezing layers** (e.g., transfer learning scenarios)
- Input and Output layers have no trainable parameters, this field has no effect on them

### 3.3 ID Naming Convention

Recommend using **semantic IDs** for better readability:

```
Format: {stage}_{layer_type}{number}

Examples:
- "input_01"
- "enc_conv1", "enc_conv2", "enc_bn1"
- "dec_conv1", "dec_bn1"
- "dense_out", "output_01"
```

The application can also generate default layer IDs as 8-character UUID hex strings.
Semantic IDs are preferred for hand-authored or AI-generated JSON because they make
connections easier to inspect.

### 3.4 Tuple-like Parameter Format

For 2D shapes, reshape targets, pooling sizes, strides, and permutation dimensions,
the current implementation stores tuple-like values as strings, not JSON arrays.

```json
{
  "parameters": {
    "shape": "(100, 40)",
    "kernel_size": "(3, 3)",
    "strides": "(1, 1)",
    "target_shape": "(64, 1)",
    "dims": "(2, 1)"
  }
}
```

---

## 4. Available Layer Types Quick Reference

### 4.1 Input/Output Layers

| type | Display Name | Parameters |
|------|--------------|------------|
| `input` | Input Layer | `shape`: "(H, W)" or "(L, C)", `dtype`: "float32", optional `name` |
| `output` | Output Layer | `activation` (none/linear/sigmoid/softmax/tanh/relu), `optimizer`, `learning_rate`, `loss`, `metrics` |

### 4.2 Core Layers

| type | Display Name | Main Parameters |
|------|--------------|-----------------|
| `dense` | Dense | `units`, `activation` (none/relu/sigmoid etc.), `use_bias`, `kernel_initializer` |
| `embedding` | Embedding | `input_dim`, `output_dim`, `mask_zero` |

### 4.3 Convolutional Layers

| type | Display Name | Main Parameters |
|------|--------------|-----------------|
| `conv1d` | Conv1D | `filters`, `kernel_size`, `strides`, `padding` (`valid`/`same`/`causal`), `activation`, `use_bias` |
| `conv2d` | Conv2D | `filters`, `kernel_size`: "(H,W)", `strides`: "(H,W)", `padding`, `activation`, `use_bias` |
| `conv1d_transpose` | Conv1DTranspose | Same as conv1d (for upsampling) |
| `conv2d_transpose` | Conv2DTranspose | Same as conv2d (for upsampling) |
| `separable_conv1d` | SeparableConv1D | `filters`, `kernel_size`, `strides`, `padding` |

Activation parameter values follow each layer implementation. Dense layers use lowercase
`"none"` for no activation; Conv2D-family UI defaults may use `"None"` for no activation.
Prefer explicit activation layers when the visual graph should show Conv/BN/Activation
as separate stages.

### 4.4 Pooling Layers

| type | Display Name | Main Parameters |
|------|--------------|-----------------|
| `max_pooling1d` | MaxPooling1D | `pool_size`, `strides`, `padding` |
| `max_pooling2d` | MaxPooling2D | `pool_size`: "(H,W)", `strides`, `padding` |
| `avg_pooling1d` | AveragePooling1D | `pool_size`, `strides`, `padding` |
| `global_max_pooling1d` | GlobalMaxPooling1D | No parameters |
| `global_avg_pooling1d` | GlobalAveragePooling1D | No parameters |
| `global_max_pooling2d` | GlobalMaxPooling2D | No parameters |
| `global_avg_pooling2d` | GlobalAveragePooling2D | No parameters |

### 4.5 Normalization Layers

| type | Display Name | Main Parameters |
|------|--------------|-----------------|
| `batch_norm` | BatchNormalization | `momentum`, `epsilon`, `center`, `scale` |
| `layer_norm` | LayerNormalization | `epsilon`, `center`, `scale` |

### 4.6 Regularization Layers

| type | Display Name | Main Parameters |
|------|--------------|-----------------|
| `dropout` | Dropout | `rate`: 0.0~1.0 |
| `spatial_dropout1d` | SpatialDropout1D | `rate` |
| `spatial_dropout2d` | SpatialDropout2D | `rate` |
| `gaussian_noise` | GaussianNoise | `stddev` |
| `gaussian_dropout` | GaussianDropout | `rate` |

### 4.7 Activation Layers

| type | Display Name | Main Parameters |
|------|--------------|-----------------|
| `activation` | Activation | `activation`: relu, sigmoid, tanh, softmax, gelu, swish, etc. |
| `leaky_relu` | LeakyReLU | `alpha` |
| `prelu` | PReLU | No parameters (auto-learned) |
| `elu` | ELU | `alpha` |
| `softmax` | Softmax | `axis` |

### 4.8 Shape Transformation Layers

| type | Display Name | Main Parameters |
|------|--------------|-----------------|
| `flatten` | Flatten | No parameters |
| `reshape` | Reshape | `target_shape`: "(H, W)" |
| `permute` | Permute | `dims`: "(2, 1)" |
| `repeat_vector` | RepeatVector | `n` |
| `upsampling1d` | UpSampling1D | `size` |
| `upsampling2d` | UpSampling2D | `size`: "(H, W)" |

### 4.9 Recurrent Layers

| type | Display Name | Main Parameters |
|------|--------------|-----------------|
| `lstm` | LSTM | `units`, `return_sequences`, `return_state`, `activation`, `recurrent_activation`, `bidirectional`, `dropout`, `recurrent_dropout` |
| `gru` | GRU | `units`, `return_sequences`, `activation`, `bidirectional`, `dropout`, `recurrent_dropout` |
| `simple_rnn` | SimpleRNN | `units`, `return_sequences`, `activation`, `dropout` |

### 4.10 Attention Layers

| type | Display Name | Main Parameters |
|------|--------------|-----------------|
| `multi_head_attention` | MultiHeadAttention | `num_heads`, `key_dim`, `value_dim`, `dropout`, `use_bias` |
| `transformer_encoder` | TransformerEncoder | `num_heads`, `key_dim`, `ff_dim`, `dropout_rate`, `activation`, `epsilon`, `pre_norm` |
| `transformer_decoder` | TransformerDecoder | `num_heads`, `key_dim`, `ff_dim`, `dropout_rate`, `activation` |
| `positional_encoding` | PositionalEncoding | `max_length`, `encoding_type`, `dropout_rate` |
| `attention` | Attention | `use_scale`, `score_mode`, `dropout` |
| `additive_attention` | AdditiveAttention | `use_scale`, `dropout` |

### 4.11 Merge Layers

| type | Display Name | Main Parameters |
|------|--------------|-----------------|
| `concatenate` | Concatenate | `axis` |
| `add` | Add | No parameters |
| `multiply` | Multiply | No parameters |
| `average` | Average | No parameters |
| `maximum` | Maximum | No parameters |
| `minimum` | Minimum | No parameters |
| `dot` | Dot | `axes`, `normalize` |

---

## 5. Connection Definition (connections)

```json
{
  "source": "source_layer_id",
  "target": "target_layer_id"
}
```

### 5.1 Connection Rules

1. **Each layer has only one input port and one output port** (except merge layers)
2. **Cannot create cyclic connections**
3. **Connection order should match topological sort**

### 5.2 Connection Order Specification

Arrange in topological order for readability:

```json
"connections": [
  {"source": "input_01", "target": "conv1"},
  {"source": "conv1", "target": "bn1"},
  {"source": "bn1", "target": "act1"},
  {"source": "act1", "target": "pool1"},
  ...
]
```

The serializer writes connections in insertion order. Topological ordering is strongly
recommended for hand-authored JSON but is not currently enforced during load or save.

### 5.3 Output Layer Requirement

An explicit `output` layer is recommended because it stores the compile settings used
by training. The current graph validator only requires at least one terminal layer, so
a graph ending in `dense` or `softmax` can still validate and build. In that case,
runtime compilation falls back to root `compile_config`.

### 5.4 Keras Import Notes

`KerasModelParser` maps Keras model layers into model builder layer types where possible.
Some parser mappings are not currently registered layer types and will be skipped during
import:

| Keras Layer | Parser Type | Current Status |
|-------------|-------------|----------------|
| `SeparableConv2D` | `separable_conv2d` | Not registered |
| `DepthwiseConv2D` | `depthwise_conv2d` | Not registered |
| `Bidirectional` | `bidirectional` | Not registered as a standalone layer; LSTM/GRU have a `bidirectional` parameter |
| `AveragePooling2D` | `avg_pooling2d` | Not registered |

Imported layers with weights are marked with `has_weights: true`; trainability is stored
with `trainable: false` only when a layer is frozen.

---

## 6. Common Model Templates

### 6.1 Audio Denoising Autoencoder (1D)

**Input**: `(144000, 1)` - 3 seconds @ 48kHz mono
**Task**: Regression (input → denoised output)

```
Structure: Input → [Conv1D+BN+ReLU]×3 → [Conv1DT+BN+ReLU]×3 → Output

Encoder (downsampling):
  Conv1D(32, k=15, s=4) → BN → ReLU  (144000→36000)
  Conv1D(64, k=9, s=4)  → BN → ReLU  (36000→9000)
  Conv1D(128, k=5, s=4) → BN → ReLU  (9000→2250)

Decoder (upsampling):
  Conv1DT(64, k=5, s=4)  → BN → ReLU  (2250→9000)
  Conv1DT(32, k=9, s=4)  → BN → ReLU  (9000→36000)
  Conv1DT(1, k=15, s=4, act=tanh)    (36000→144000)

Compile config:
  loss: mse
  metrics: mae
```

### 6.2 MFCC Binary Classifier

**Input**: `(281, 40)` - MFCC features (time frames, coefficients)
**Task**: Binary classification

```
Structure: Input → [Conv1D+BN+ReLU+Pool]×3 → GAP → Dense+Dropout → Dense(sigmoid) → Output

Feature extraction:
  Conv1D(32, k=3) → BN → ReLU → MaxPool(2)  (281→140)
  Conv1D(64, k=3) → BN → ReLU → MaxPool(2)  (140→70)
  Conv1D(128, k=3) → BN → ReLU              (70)
  GlobalAveragePooling1D                     (128)

Classifier head:
  Dense(64, relu) → Dropout(0.5) → Dense(1, sigmoid)

Compile config:
  loss: binary_crossentropy
  metrics: accuracy
```

### 6.3 Multi-class CNN

**Input**: `(time_frames, feature_dim)`
**Task**: Multi-class classification (N classes)

```
Structure: Input → CNN feature extraction → Flatten → Dense → Softmax → Output

Output layer:
  Dense(N, activation=softmax)

Compile config:
  loss: sparse_categorical_crossentropy or categorical_crossentropy
  metrics: accuracy
```

### 6.4 Transformer Encoder

**Input**: `(sequence_length, feature_dim)`
**Task**: Sequence classification

```
Structure: Input → PositionalEncoding → TransformerEncoder×N → GAP → Dense → Output

Recommended parameters:
  num_heads: 4-8
  key_dim: 64
  ff_dim: 128-256
  dropout_rate: 0.1
```

---

## 7. Generation Checklist ✅

Before generating model JSON, please check:

**Layout Specification** 🎯
- [ ] ⛔ **No single-line layout** (all nodes at y=0 is wrong!)
- [ ] ✅ Use **multi-row 2D layout**, organized by functional stages
- [ ] Closely related nodes grouped together (Conv+BN+ReLU as one group)
- [ ] Node spacing ≥ 120px (same group) or ≥ 150px (cross-group)
- [ ] Row spacing ≥ 100px

**Structure Specification**
- [ ] All layer IDs unique and semantic
- [ ] Input layer shape matches task
- [ ] Output layer contains correct loss and metrics, or root `compile_config` is intentionally used as fallback
- [ ] Root `compile_config` is kept in sync with Output layer parameters when both are present
- [ ] Connections in topological order for readability
- [ ] No cyclic connections
- [ ] Parameter values within valid ranges
- [ ] Tuple-like 2D parameters are strings, e.g. `"(3, 3)"`, not JSON arrays

**Trainability**
- [ ] By default all layers trainable (no need to explicitly set trainable)
- [ ] Only set `trainable: false` when freezing layers

---

## 8. Example: Complete Classifier JSON (2D Layout)

The following example demonstrates the correct **2D layout** approach, dividing the network into three stage rows:

```
Layout diagram:

Row 1 (y=0):   Input → Conv1 → BN1 → ReLU1 → Pool1
                 │
                 └───────────────────────────────┐
                                                 ▼
Row 2 (y=100): Conv2 → BN2 → ReLU2 → GAP ───────┐
                                                 │
                 ┌───────────────────────────────┘
                 ▼
Row 3 (y=200): Dense1 → Dropout → Dense2 → Output
```

```json
{
  "version": "1.0",
  "name": "mfcc_binary_classifier",
  "description": "MFCC feature binary classifier - 2D layout example",
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

### 8.1 Layout Key Points

| Stage | Y Coordinate | Nodes | Description |
|-------|--------------|-------|-------------|
| Feature Extraction 1 | 0 | Input, Conv1, BN1, ReLU1, Pool1 | First conv block |
| Feature Extraction 2 | 100 | Conv2, BN2, ReLU2, GAP | Second conv block + global pooling |
| Classifier Head | 200 | Dense1, Dropout, Dense2, Output | Fully connected classification layers |

**Notes**:
- Within same row, X coordinates increase, closely related nodes have smaller spacing (120px)
- Different stages distinguished by Y coordinate, stage spacing 100px
- When connecting across rows, next row starts from X=0, forming clear visual flow

---

*Last Updated: 2026-05-24*
