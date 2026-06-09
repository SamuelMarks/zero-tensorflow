# Semantic Implementation Plan (`zero-tensorflow`)

This document exhaustively tracks the transition of `zero-tensorflow` from **Structural Stubs** (API signature compliance) to **Semantic Implementations** (NumPy-backed mathematics, state management, and `ml-switcheroo-ir` LogicalNode emission).

---

## Phase 1: Core Engine & Tracing Foundation
To achieve zero-dependency WASM/WebGPU compilation via `ml-switcheroo`, the core constructs must support both eager NumPy execution and AST-based graph tracing.

- [x] **`Tensor` Primitive**
  - [x] Implement dual-state (Eager NumPy `NDArray` + Traced `LogicalNode`).
  - [x] Overload Python magic methods (`__add__`, `__mul__`, `__getitem__`, etc.) to emit IR.
- [x] **`tf.Variable`**
  - [x] Implement stateful assignment (`assign`, `assign_add`, `assign_sub`).
  - [x] Integrate with `GradientTape` for trainable weight tracking.
- [x] **`tf.GradientTape`**
  - [x] Implement forward-pass operation recording.
  - [x] Implement backward-pass VJP (Vector-Jacobian Product) math.
  - [x] Emit autodiff IR nodes when inside a tracing context.
- [x] **`@tf.function`**
  - [x] Implement AST tracing context manager.
  - [x] Capture closures and side-effects.

---

## Phase 2: Core Math & NN Operations
These operations must evaluate eagerly via NumPy and emit strict ONNX dialect `LogicalNode` schemas when traced.

### `tf.math` Operations
- [x] `add`, `subtract`, `multiply`, `divide`
- [x] `exp`, `log`, `pow`, `sqrt`
- [x] `reduce_sum`, `reduce_mean`, `reduce_max`, `reduce_min`
- [x] `matmul`, `tensordot`

### `tf.nn` Operations (Math + IR Emission)
- [x] `elu` (Math complete, add IR)
- [x] `leaky_relu` (Math complete, add IR)
- [x] `relu` (Math complete, add IR)
- [x] `selu` (Math complete, add IR)
- [x] `sigmoid` (Math complete, add IR)
- [x] `softmax` (Math complete, add IR)
- [x] `tanh` (Math complete, add IR)

---

## Phase 3: Keras States (Initializers & Schedules)

### Initializers (Implement Random Generators)
- [x] `Constant`, `Zeros`, `Ones`, `constant`, `zeros`, `ones`
- [x] `Identity`, `IdentityInitializer`, `identity`
- [x] `Orthogonal`, `OrthogonalInitializer`, `orthogonal`
- [x] `RandomNormal`, `random_normal`
- [x] `RandomUniform`, `random_uniform`
- [x] `TruncatedNormal`, `truncated_normal`
- [x] `VarianceScaling`, `variance_scaling`
- [x] `GlorotNormal`, `glorot_normal`
- [x] `GlorotUniform`, `glorot_uniform`
- [x] `HeNormal`, `he_normal`
- [x] `HeUniform`, `he_uniform`
- [x] `LecunNormal`, `lecun_normal`
- [x] `LecunUniform`, `lecun_uniform`
- [x] `STFT`, `STFTInitializer`, `stft`

### Learning Rate Schedules (Implement `__call__(step)`)
- [x] `CosineDecay`
- [x] `CosineDecayRestarts`
- [x] `ExponentialDecay`
- [x] `InverseTimeDecay`
- [x] `PiecewiseConstantDecay`
- [x] `PolynomialDecay`

---

## Phase 4: Keras Optimizers (State Variables & Update Rules)
Implement `apply_gradients`, momentum tracking, and EMA (Exponential Moving Average) updates.

- [x] `Adadelta`
- [x] `Adafactor`
- [x] `Adagrad`
- [x] `Adam`
- [x] `AdamW`
- [x] `Adamax`
- [x] `Ftrl`
- [x] `Lamb`
- [x] `Lion`
- [x] `LossScaleOptimizer`
- [x] `Muon`
- [x] `Nadam`
- [x] `RMSprop`
- [x] `SGD`

---

## Phase 5: Keras Layers (Math, Weights, & IR)
Implement `build(input_shape)`, `call(inputs)`, state management, and strict ONNX-compatible IR tracing for all 159 layers.

### Core & Shaping Layers
- [x] `Dense` (Weight initialization, bias, matmul, activation)
- [x] `Activation`
- [x] `Embedding`
- [x] `Flatten`
- [x] `Reshape`
- [x] `Permute`
- [x] `RepeatVector`

### Convolutional Layers
- [x] `Conv1D`, `Convolution1D`
- [x] `Conv2D`, `Convolution2D`
- [x] `Conv3D`, `Convolution3D`
- [x] `Conv1DTranspose`, `Convolution1DTranspose`
- [x] `Conv2DTranspose`, `Convolution2DTranspose`
- [x] `Conv3DTranspose`, `Convolution3DTranspose`
- [x] `DepthwiseConv1D`
- [x] `DepthwiseConv2D`
- [x] `SeparableConv1D`, `SeparableConvolution1D`
- [x] `SeparableConv2D`, `SeparableConvolution2D`

### Pooling Layers
- [x] `MaxPooling1D`, `MaxPool1D`
- [x] `MaxPooling2D`, `MaxPool2D`
- [x] `MaxPooling3D`, `MaxPool3D`
- [x] `AveragePooling1D`, `AvgPool1D`
- [x] `AveragePooling2D`, `AvgPool2D`
- [x] `AveragePooling3D`, `AvgPool3D`
- [x] `GlobalMaxPooling1D`, `GlobalMaxPool1D`
- [x] `GlobalMaxPooling2D`, `GlobalMaxPool2D`
- [x] `GlobalMaxPooling3D`, `GlobalMaxPool3D`
- [x] `GlobalAveragePooling1D`, `GlobalAvgPool1D`
- [x] `GlobalAveragePooling2D`, `GlobalAvgPool2D`
- [x] `GlobalAveragePooling3D`, `GlobalAvgPool3D`

### Recurrent Layers
- [x] `RNN`
- [x] `SimpleRNN`, `SimpleRNNCell`
- [x] `LSTM`, `LSTMCell`
- [x] `GRU`, `GRUCell`
- [x] `ConvLSTM1D`, `ConvLSTM2D`, `ConvLSTM3D`
- [x] `Bidirectional`
- [x] `StackedRNNCells`

### Attention Layers
- [x] `Attention`
- [x] `AdditiveAttention`
- [x] `MultiHeadAttention`
- [x] `GroupQueryAttention`

### Normalization Layers
- [x] `BatchNormalization` (Moving mean/var updates, training vs inference modes)
- [x] `LayerNormalization`
- [x] `GroupNormalization`
- [x] `RMSNormalization`
- [x] `UnitNormalization`
- [x] `SpectralNormalization`

### Regularization Layers
- [x] `Dropout` (Active only in training)
- [x] `AlphaDropout`
- [x] `GaussianDropout`
- [x] `GaussianNoise`
- [x] `SpatialDropout1D`, `SpatialDropout2D`, `SpatialDropout3D`
- [x] `ActivityRegularization`

### Merging & Math Layers
- [x] `Add`
- [x] `Subtract`
- [x] `Multiply`
- [x] `Average`
- [x] `Maximum`
- [x] `Minimum`
- [x] `Concatenate`
- [x] `Dot`
- [x] `EinsumDense`

### Advanced Activations
- [x] `LeakyReLU`
- [x] `PReLU`
- [x] `ELU`
- [x] `ThresholdedReLU`
- [x] `ReLU`
- [x] `Softmax`

### Cropping, Padding, UpSampling
- [x] `Cropping1D`, `Cropping2D`, `Cropping3D`
- [x] `ZeroPadding1D`, `ZeroPadding2D`, `ZeroPadding3D`
- [x] `UpSampling1D`, `UpSampling2D`, `UpSampling3D`
- [x] `CenterCrop`

### Preprocessing & Augmentation Layers
- [x] *Categorical & Text:* `StringLookup`, `IntegerLookup`, `CategoryEncoding`, `Hashing`, `HashedCrossing`, `TextVectorization`
- [x] *Numerical:* `Normalization`, `Discretization`
- [x] *Image Augmentation:* `Rescaling`, `Resizing`, `RandomCrop`, `RandomFlip`, `RandomTranslation`, `RandomRotation`, `RandomZoom`, `RandomHeight`, `RandomWidth`, `RandomContrast`, `RandomBrightness`
- [x] *Advanced Augmentation:* `AugMix`, `AutoContrast`, `CutMix`, `Equalization`, `MixUp`, `RandAugment`, `RandomColorDegeneration`, `RandomColorJitter`, `RandomElasticTransform`, `RandomErasing`, `RandomGaussianBlur`, `RandomGrayscale`, `RandomHue`, `RandomInvert`, `RandomPerspective`, `RandomPosterization`, `RandomSaturation`, `RandomSharpness`, `RandomShear`, `Solarization`
- [x] *Audio:* `MelSpectrogram`, `STFTSpectrogram`
- [x] *Misc:* `MaxNumBoundingBoxes`

### Functional / Wrappers
- [x] `TimeDistributed`
- [x] `Wrapper`
- [x] `Lambda`
- [x] `Pipeline`
- [x] `FlaxLayer`
- [x] `JaxLayer`
- [x] `TorchModuleWrapper`
- [x] `TFSMLayer`

---

## Phase 6: Losses & Metrics (Math & State)

### Keras Losses (Implement `__call__(y_true, y_pred)`)
- [x] `BinaryCrossentropy`, `CategoricalCrossentropy`, `SparseCategoricalCrossentropy`
- [x] `BinaryFocalCrossentropy`, `CategoricalFocalCrossentropy`
- [x] `MeanSquaredError`, `MeanAbsoluteError`, `MeanAbsolutePercentageError`, `MeanSquaredLogarithmicError`
- [x] `CosineSimilarity`
- [x] `Hinge`, `SquaredHinge`, `CategoricalHinge`
- [x] `Huber`, `LogCosh`, `Poisson`, `KLDivergence`
- [x] `Dice`, `Circle`, `Tversky`, `CTC`, `CategoricalGeneralizedCrossEntropy`

### Keras Metrics (Implement `update_state`, `result`, `reset_states`)
- [x] *Accuracy family:* `Accuracy`, `BinaryAccuracy`, `CategoricalAccuracy`, `SparseCategoricalAccuracy`, `TopKCategoricalAccuracy`, `SparseTopKCategoricalAccuracy`
- [x] *Crossentropy family:* `BinaryCrossentropy`, `CategoricalCrossentropy`, `SparseCategoricalCrossentropy`
- [x] *Error family:* `MeanSquaredError`, `RootMeanSquaredError`, `MeanAbsoluteError`, `MeanAbsolutePercentageError`, `MeanSquaredLogarithmicError`, `LogCoshError`
- [x] *Confusion Matrix family:* `TruePositives`, `TrueNegatives`, `FalsePositives`, `FalseNegatives`, `Precision`, `Recall`, `F1Score`, `FBetaScore`, `AUC`
- [x] *IoU family:* `IoU`, `BinaryIoU`, `MeanIoU`, `OneHotIoU`, `OneHotMeanIoU`
- [x] *Correlation family:* `CosineSimilarity`, `PearsonCorrelation`, `ConcordanceCorrelation`, `R2Score`
- [x] *Curve/Threshold family:* `PrecisionAtRecall`, `RecallAtPrecision`, `SensitivityAtSpecificity`, `SpecificityAtSensitivity`
- [x] *Stateless wrappers:* `Mean`, `Sum`, `MeanMetricWrapper`, `KLDivergence`, `Poisson`, `Hinge`, `SquaredHinge`, `CategoricalHinge`

---

## Phase 7: End-to-End Compliance & Validation
- [x] **MLP Forward/Backward:** Validate Golden Seed test against real TF.
- [x] **CNN Padding/Strides:** Verify exact spatial outputs against real TF.
- [x] **NanoGPT Tracing:** Validate Transformer mask broadcasts and layer norm gradients.
- [x] **AST -> ONNX Generation:** Ensure the `ml-switcheroo-ir` compliance checker reports 100% dialect validity for traced models.