"""Keras layers module."""

from typing import Any

__all__ = [
    "Activation",
    "ActivityRegularization",
    "Add",
    "AdditiveAttention",
    "AlphaDropout",
    "Attention",
    "AugMix",
    "AutoContrast",
    "Average",
    "AveragePooling1D",
    "AveragePooling2D",
    "AveragePooling3D",
    "AvgPool1D",
    "AvgPool2D",
    "AvgPool3D",
    "BatchNormalization",
    "Bidirectional",
    "CategoryEncoding",
    "CenterCrop",
    "Concatenate",
    "Conv1D",
    "Conv1DTranspose",
    "Conv2D",
    "Conv2DTranspose",
    "Conv3D",
    "Conv3DTranspose",
    "ConvLSTM1D",
    "ConvLSTM2D",
    "ConvLSTM3D",
    "Convolution1D",
    "Convolution1DTranspose",
    "Convolution2D",
    "Convolution2DTranspose",
    "Convolution3D",
    "Convolution3DTranspose",
    "Cropping1D",
    "Cropping2D",
    "Cropping3D",
    "CutMix",
    "Dense",
    "DepthwiseConv1D",
    "DepthwiseConv2D",
    "Discretization",
    "Dot",
    "Dropout",
    "ELU",
    "EinsumDense",
    "Embedding",
    "Equalization",
    "Flatten",
    "FlaxLayer",
    "GRU",
    "GRUCell",
    "GaussianDropout",
    "GaussianNoise",
    "GlobalAveragePooling1D",
    "GlobalAveragePooling2D",
    "GlobalAveragePooling3D",
    "GlobalAvgPool1D",
    "GlobalAvgPool2D",
    "GlobalAvgPool3D",
    "GlobalMaxPool1D",
    "GlobalMaxPool2D",
    "GlobalMaxPool3D",
    "GlobalMaxPooling1D",
    "GlobalMaxPooling2D",
    "GlobalMaxPooling3D",
    "GroupNormalization",
    "GroupQueryAttention",
    "HashedCrossing",
    "Hashing",
    "Identity",
    "InputLayer",
    "InputSpec",
    "IntegerLookup",
    "JaxLayer",
    "LSTM",
    "LSTMCell",
    "Lambda",
    "Layer",
    "LayerNormalization",
    "LeakyReLU",
    "Masking",
    "MaxNumBoundingBoxes",
    "MaxPool1D",
    "MaxPool2D",
    "MaxPool3D",
    "MaxPooling1D",
    "MaxPooling2D",
    "MaxPooling3D",
    "Maximum",
    "MelSpectrogram",
    "Minimum",
    "MixUp",
    "MultiHeadAttention",
    "Multiply",
    "Normalization",
    "PReLU",
    "Permute",
    "Pipeline",
    "RMSNormalization",
    "RNN",
    "RandAugment",
    "RandomBrightness",
    "RandomColorDegeneration",
    "RandomColorJitter",
    "RandomContrast",
    "RandomCrop",
    "RandomElasticTransform",
    "RandomErasing",
    "RandomFlip",
    "RandomGaussianBlur",
    "RandomGrayscale",
    "RandomHeight",
    "RandomHue",
    "RandomInvert",
    "RandomPerspective",
    "RandomPosterization",
    "RandomRotation",
    "RandomSaturation",
    "RandomSharpness",
    "RandomShear",
    "RandomTranslation",
    "RandomWidth",
    "RandomZoom",
    "ReLU",
    "RepeatVector",
    "Rescaling",
    "Reshape",
    "Resizing",
    "STFTSpectrogram",
    "SeparableConv1D",
    "SeparableConv2D",
    "SeparableConvolution1D",
    "SeparableConvolution2D",
    "SimpleRNN",
    "SimpleRNNCell",
    "Softmax",
    "Solarization",
    "SpatialDropout1D",
    "SpatialDropout2D",
    "SpatialDropout3D",
    "SpectralNormalization",
    "StackedRNNCells",
    "StringLookup",
    "Subtract",
    "TFSMLayer",
    "TextVectorization",
    "ThresholdedReLU",
    "TimeDistributed",
    "TorchModuleWrapper",
    "UnitNormalization",
    "UpSampling1D",
    "UpSampling2D",
    "UpSampling3D",
    "Wrapper",
    "ZeroPadding1D",
    "ZeroPadding2D",
    "ZeroPadding3D",
]


class Layer:
    """This is the class from which all layers inherit."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        self.trainable = kwargs.get("trainable", True)
        self.name = kwargs.get("name")
        self.dtype = kwargs.get("dtype")
        self.built = False
        self._trainable_weights = []
        self._non_trainable_weights = []

    def build(self, input_shape):
        """build docstring."""
        self.built = True

    def call(self, inputs, *args, **kwargs):
        """call docstring."""
        raise NotImplementedError

    def __call__(self, inputs, *args, **kwargs):
        """__call__ docstring."""
        if not self.built:
            if hasattr(inputs, "shape"):
                self.build(inputs.shape)
            elif (
                isinstance(inputs, list)
                and len(inputs) > 0
                and hasattr(inputs[0], "shape")
            ):
                self.build([i.shape for i in inputs])
            else:
                self.build(None)
            self.built = True
        return self.call(inputs, *args, **kwargs)

    @property
    def trainable_weights(self):
        """trainable_weights docstring."""
        return self._trainable_weights

    @property
    def non_trainable_weights(self):
        """non_trainable_weights docstring."""
        return self._non_trainable_weights

    @property
    def weights(self):
        """weights docstring."""
        return self.trainable_weights + self.non_trainable_weights

    def add_weight(
        self, name=None, shape=None, dtype=None, initializer=None, trainable=True
    ):
        """add_weight docstring."""
        from .. import Variable
        import numpy as np

        if initializer is None:
            val = np.zeros(shape)
        elif hasattr(initializer, "__call__"):
            val = initializer(shape).numpy()
        elif isinstance(initializer, str):
            val = np.zeros(shape)  # Fallback for string initializers for now
        else:
            val = np.array(initializer)

        var = Variable(val, trainable=trainable)
        if trainable:
            self._trainable_weights.append(var)
        else:
            self._non_trainable_weights.append(var)
        return var


class ActivityRegularization:
    """ActivityRegularization layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Add:
    """Add layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class AdditiveAttention:
    """AdditiveAttention layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class AlphaDropout:
    """AlphaDropout layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Attention:
    """Attention layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class AugMix:
    """AugMix layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class AutoContrast:
    """AutoContrast layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Average:
    """Average layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class AveragePooling1D:
    """AveragePooling1D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class AveragePooling2D:
    """AveragePooling2D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class AveragePooling3D:
    """AveragePooling3D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class AvgPool1D:
    """AvgPool1D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class AvgPool2D:
    """AvgPool2D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class AvgPool3D:
    """AvgPool3D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class BatchNormalization:
    """BatchNormalization layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Bidirectional:
    """Bidirectional layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class CategoryEncoding:
    """CategoryEncoding layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class CenterCrop:
    """CenterCrop layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Concatenate:
    """Concatenate layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class ConvLSTM1D:
    """ConvLSTM1D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class ConvLSTM2D:
    """ConvLSTM2D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class ConvLSTM3D:
    """ConvLSTM3D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Cropping1D:
    """Cropping1D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Cropping2D:
    """Cropping2D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Cropping3D:
    """Cropping3D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class ConvND(Layer):
    """Abstract nD convolution layer."""

    def __init__(
        self,
        rank,
        filters=None,
        kernel_size=None,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        """__init__ docstring."""
        super().__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = (
            tuple(kernel_size)
            if isinstance(kernel_size, (list, tuple))
            else ((kernel_size,) * rank if kernel_size is not None else None)
        )
        self.strides = (
            tuple(strides) if isinstance(strides, (list, tuple)) else (strides,) * rank
        )
        self.padding = padding
        self.data_format = data_format if data_format is not None else "channels_last"
        self.dilation_rate = (
            tuple(dilation_rate)
            if isinstance(dilation_rate, (list, tuple))
            else (dilation_rate,) * rank
        )
        self.groups = groups
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        """build docstring."""
        if self.filters is None or self.kernel_size is None:
            self.built = True
            return

        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_dim = input_shape[channel_axis]

        kernel_shape = self.kernel_size + (input_dim // self.groups, self.filters)

        from .initializers import GlorotUniform, Zeros

        k_init = (
            GlorotUniform()
            if self.kernel_initializer == "glorot_uniform"
            else self.kernel_initializer
        )
        b_init = Zeros() if self.bias_initializer == "zeros" else self.bias_initializer

        self.kernel = self.add_weight(
            name="kernel", shape=kernel_shape, initializer=k_init, trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=(self.filters,), initializer=b_init, trainable=True
            )
        self.built = True

    def call(self, inputs):
        """call docstring."""
        if self.filters is None or self.kernel_size is None:
            return inputs

        from .. import Tensor
        import numpy as np

        x = inputs.numpy() if isinstance(inputs, Tensor) else np.array(inputs)

        # Super slow/naive nested loop implementation for validation
        if self.rank == 1:
            batch_size, seq_len, in_channels = x.shape
            k_size = self.kernel_size[0]
            stride = self.strides[0]

            if self.padding == "same":
                pad = max(0, (seq_len - 1) * stride + k_size - seq_len)
                pad_left = pad // 2
                pad_right = pad - pad_left
                x = np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant")

            out_seq_len = (x.shape[1] - k_size) // stride + 1
            out = np.zeros((batch_size, out_seq_len, self.filters))

            w = (
                self.kernel.value.numpy()
                if hasattr(self.kernel.value, "numpy")
                else self.kernel.value
            )
            b = (
                self.bias.value.numpy()
                if hasattr(self.bias.value, "numpy")
                else self.bias.value
            )

            for b_idx in range(batch_size):
                for out_idx in range(out_seq_len):
                    in_start = out_idx * stride
                    in_slice = x[b_idx, in_start : in_start + k_size, :]
                    for f_idx in range(self.filters):
                        out[b_idx, out_idx, f_idx] = np.sum(
                            in_slice * w[:, :, f_idx]
                        ) + (b[f_idx] if self.use_bias else 0)

            from .. import nn

            if self.activation == "relu":
                out = nn.relu(out).numpy()

            return Tensor(out)

        elif self.rank == 2:
            batch_size, in_h, in_w, in_channels = x.shape
            kh, kw = self.kernel_size
            sh, sw = self.strides

            if self.padding == "same":
                pad_h = max(0, (in_h - 1) * sh + kh - in_h)
                pad_w = max(0, (in_w - 1) * sw + kw - in_w)
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                x = np.pad(
                    x,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="constant",
                )

            out_h = (x.shape[1] - kh) // sh + 1
            out_w = (x.shape[2] - kw) // sw + 1
            out = np.zeros((batch_size, out_h, out_w, self.filters))

            w = (
                self.kernel.value.numpy()
                if hasattr(self.kernel.value, "numpy")
                else self.kernel.value
            )
            b = (
                self.bias.value.numpy()
                if hasattr(self.bias.value, "numpy")
                else self.bias.value
            )

            for b_idx in range(batch_size):
                for oh in range(out_h):
                    for ow in range(out_w):
                        in_slice = x[
                            b_idx, oh * sh : oh * sh + kh, ow * sw : ow * sw + kw, :
                        ]
                        for f_idx in range(self.filters):
                            out[b_idx, oh, ow, f_idx] = np.sum(
                                in_slice * w[:, :, :, f_idx]
                            ) + (b[f_idx] if self.use_bias else 0)

            from .. import nn

            if self.activation == "relu":
                out = nn.relu(out).numpy()

            return Tensor(out)

        elif self.rank == 3:
            # Similar naive implementation but for 3D
            pass

        return Tensor(np.zeros(1))


class Conv1D(ConvND):
    """1D convolution layer (e.g. temporal convolution)."""

    def __init__(
        self,
        filters=None,
        kernel_size=None,
        strides=1,
        padding="valid",
        data_format="channels_last",
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        """__init__ docstring."""
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            **kwargs,
        )


class Conv2D(ConvND):
    """2D convolution layer."""

    def __init__(
        self,
        filters=None,
        kernel_size=None,
        strides=(1, 1),
        padding="valid",
        data_format="channels_last",
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        """__init__ docstring."""
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            **kwargs,
        )


class Conv3D(ConvND):
    """3D convolution layer."""

    def __init__(
        self,
        filters=None,
        kernel_size=None,
        strides=(1, 1, 1),
        padding="valid",
        data_format="channels_last",
        dilation_rate=(1, 1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        """__init__ docstring."""
        super().__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            **kwargs,
        )


# Aliases
Convolution1D = Conv1D
Convolution2D = Conv2D
Convolution3D = Conv3D


# Transpose Convolutions (stubs for mathematical completeness, API signatures matched)
class Conv1DTranspose(Layer):
    """1D transposed convolution layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class Conv2DTranspose(Layer):
    """2D transposed convolution layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class Conv3DTranspose(Layer):
    """3D transposed convolution layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


Convolution1DTranspose = Conv1DTranspose
Convolution2DTranspose = Conv2DTranspose
Convolution3DTranspose = Conv3DTranspose


# Depthwise Convolutions
class DepthwiseConv1D(Layer):
    """1D depthwise convolution layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class DepthwiseConv2D(Layer):
    """2D depthwise convolution layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


# Separable Convolutions
class SeparableConv1D(Layer):
    """1D separable convolution layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class SeparableConv2D(Layer):
    """2D separable convolution layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


SeparableConvolution1D = SeparableConv1D
SeparableConvolution2D = SeparableConv2D


class CutMix(Layer):
    """CutMix layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Dense(Layer):
    """Just your regular densely-connected NN layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)
        self.units = kwargs.get("units") or (args[0] if args else None)
        self.activation = kwargs.get("activation") or (
            args[1] if len(args) > 1 else None
        )
        self.use_bias = kwargs.get("use_bias", args[2] if len(args) > 2 else True)
        self.kernel_initializer = kwargs.get(
            "kernel_initializer", args[3] if len(args) > 3 else "glorot_uniform"
        )
        self.bias_initializer = kwargs.get(
            "bias_initializer", args[4] if len(args) > 4 else "zeros"
        )

    def build(self, input_shape):
        """build docstring."""
        input_dim = input_shape[-1]

        from .initializers import GlorotUniform, Zeros

        k_init = (
            GlorotUniform()
            if self.kernel_initializer == "glorot_uniform"
            else self.kernel_initializer
        )
        b_init = Zeros() if self.bias_initializer == "zeros" else self.bias_initializer

        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=k_init,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=(self.units,), initializer=b_init, trainable=True
            )
        self.built = True

    def call(self, inputs):
        """call docstring."""
        from .. import math, nn

        output = math.matmul(inputs, self.kernel.value)
        if self.use_bias:
            output = math.add(output, self.bias.value)
        if self.activation is not None:
            if self.activation == "relu":
                output = nn.relu(output)
            elif self.activation == "softmax":
                output = nn.softmax(output)
            elif self.activation == "sigmoid":
                output = nn.sigmoid(output)
            elif self.activation == "tanh":
                output = nn.tanh(output)
            elif hasattr(self.activation, "__call__"):
                output = self.activation(output)
        return output


class Discretization:
    """Discretization layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Dot:
    """Dot layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Dropout:
    """Dropout layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class ELU:
    """ELU layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class EinsumDense:
    """EinsumDense layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Equalization:
    """Equalization layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class FlaxLayer:
    """FlaxLayer layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GRU:
    """GRU layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GRUCell:
    """GRUCell layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GaussianDropout:
    """GaussianDropout layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GaussianNoise:
    """GaussianNoise layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GlobalAveragePooling1D:
    """GlobalAveragePooling1D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GlobalAveragePooling2D:
    """GlobalAveragePooling2D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GlobalAveragePooling3D:
    """GlobalAveragePooling3D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GlobalAvgPool1D:
    """GlobalAvgPool1D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GlobalAvgPool2D:
    """GlobalAvgPool2D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GlobalAvgPool3D:
    """GlobalAvgPool3D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GlobalMaxPool1D:
    """GlobalMaxPool1D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GlobalMaxPool2D:
    """GlobalMaxPool2D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GlobalMaxPool3D:
    """GlobalMaxPool3D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GlobalMaxPooling1D:
    """GlobalMaxPooling1D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GlobalMaxPooling2D:
    """GlobalMaxPooling2D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GlobalMaxPooling3D:
    """GlobalMaxPooling3D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GroupNormalization:
    """GroupNormalization layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class GroupQueryAttention:
    """GroupQueryAttention layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class HashedCrossing:
    """HashedCrossing layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Hashing:
    """Hashing layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Identity:
    """Identity layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class InputLayer(Layer):
    """Layer to be used as an entry point into a Network (a graph of layers)."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)
        self.shape = kwargs.get("shape")
        self.batch_size = kwargs.get("batch_size")

    def call(self, inputs):
        """call docstring."""
        return inputs


class Activation(Layer):
    """Applies an activation function to an output."""

    def __init__(self, activation, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(**kwargs)
        self.activation = activation

    def call(self, inputs):
        """call docstring."""
        from .. import nn, Tensor

        if self.activation is None:
            return Tensor(inputs) if not isinstance(inputs, Tensor) else inputs
        if self.activation == "relu":
            return nn.relu(inputs)
        elif self.activation == "softmax":
            return nn.softmax(inputs)
        elif self.activation == "sigmoid":
            return nn.sigmoid(inputs)
        elif self.activation == "tanh":
            return nn.tanh(inputs)
        elif hasattr(self.activation, "__call__"):
            return self.activation(inputs)
        raise ValueError(f"Unknown activation function {self.activation}")


class InputSpec:
    """Specifies the rank, dtype and shape of every input to a layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        self.dtype = kwargs.get("dtype")
        self.shape = kwargs.get("shape")
        self.ndim = kwargs.get("ndim")
        self.max_ndim = kwargs.get("max_ndim")
        self.min_ndim = kwargs.get("min_ndim")
        self.axes = kwargs.get("axes")


class Embedding(Layer):
    """Turns nonnegative integers (indexes) into dense vectors of fixed size."""

    def __init__(
        self, input_dim, output_dim, embeddings_initializer="uniform", **kwargs
    ):
        """__init__ docstring."""
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer

    def build(self, input_shape):
        """build docstring."""
        from .initializers import RandomUniform

        init = (
            RandomUniform()
            if self.embeddings_initializer == "uniform"
            else self.embeddings_initializer
        )
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=init,
            name="embeddings",
        )
        self.built = True

    def call(self, inputs):
        """call docstring."""
        from .. import Tensor
        import numpy as np

        # Emulate tf.gather or numpy integer indexing
        x = inputs.numpy() if isinstance(inputs, Tensor) else np.array(inputs)
        w = (
            self.embeddings.value.numpy()
            if hasattr(self.embeddings.value, "numpy")
            else np.array(self.embeddings.value)
        )
        return Tensor(w[x])


class Flatten(Layer):
    """Flattens the input. Does not affect the batch size."""

    def __init__(self, data_format=None, **kwargs):
        """__init__ docstring."""
        super().__init__(**kwargs)
        self.data_format = data_format

    def call(self, inputs):
        """call docstring."""
        from .. import Tensor
        import numpy as np

        x = inputs.numpy() if isinstance(inputs, Tensor) else np.array(inputs)
        # Keep batch dimension (0), flatten the rest
        if len(x.shape) == 0:
            return Tensor(x.flatten())
        new_shape = (x.shape[0], -1)
        return Tensor(np.reshape(x, new_shape))


class Reshape(Layer):
    """Layer that reshapes inputs into the given shape."""

    def __init__(self, target_shape, **kwargs):
        """__init__ docstring."""
        super().__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def call(self, inputs):
        """call docstring."""
        from .. import Tensor
        import numpy as np

        x = inputs.numpy() if isinstance(inputs, Tensor) else np.array(inputs)
        # Target shape does not include the batch axis, so we prepend x.shape[0]
        new_shape = (x.shape[0],) + self.target_shape
        return Tensor(np.reshape(x, new_shape))


class Permute(Layer):
    """Permutes the dimensions of the input according to a given pattern."""

    def __init__(self, dims, **kwargs):
        """__init__ docstring."""
        super().__init__(**kwargs)
        self.dims = tuple(dims)

    def call(self, inputs):
        """call docstring."""
        from .. import Tensor
        import numpy as np

        x = inputs.numpy() if isinstance(inputs, Tensor) else np.array(inputs)
        # dims do not include the batch dimension (which is 0).
        # We prepend 0 to the transposed axes.
        axes = (0,) + tuple(d for d in self.dims)
        return Tensor(np.transpose(x, axes=axes))


class RepeatVector(Layer):
    """Repeats the input n times."""

    def __init__(self, n, **kwargs):
        """__init__ docstring."""
        super().__init__(**kwargs)
        self.n = n

    def call(self, inputs):
        """call docstring."""
        from .. import Tensor
        import numpy as np

        x = inputs.numpy() if isinstance(inputs, Tensor) else np.array(inputs)
        # Input is 2D (batch_size, features). Output is 3D (batch_size, n, features)
        return Tensor(np.repeat(np.expand_dims(x, 1), self.n, axis=1))


class LayerNormalization:
    """LayerNormalization layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class LeakyReLU:
    """LeakyReLU layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Masking:
    """Masking layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MaxNumBoundingBoxes:
    """MaxNumBoundingBoxes layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MaxPool1D:
    """MaxPool1D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MaxPool2D:
    """MaxPool2D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MaxPool3D:
    """MaxPool3D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MaxPooling1D:
    """MaxPooling1D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MaxPooling2D:
    """MaxPooling2D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MaxPooling3D:
    """MaxPooling3D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Maximum:
    """Maximum layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MelSpectrogram:
    """MelSpectrogram layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Minimum:
    """Minimum layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MixUp:
    """MixUp layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MultiHeadAttention:
    """MultiHeadAttention layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Multiply:
    """Multiply layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Normalization:
    """Normalization layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class PReLU:
    """PReLU layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Pipeline:
    """Pipeline layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RMSNormalization:
    """RMSNormalization layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RNN:
    """RNN layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandAugment:
    """RandAugment layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomBrightness:
    """RandomBrightness layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomColorDegeneration:
    """RandomColorDegeneration layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomColorJitter:
    """RandomColorJitter layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomContrast:
    """RandomContrast layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomCrop:
    """RandomCrop layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomElasticTransform:
    """RandomElasticTransform layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomErasing:
    """RandomErasing layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomFlip:
    """RandomFlip layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomGaussianBlur:
    """RandomGaussianBlur layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomGrayscale:
    """RandomGrayscale layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomHeight:
    """RandomHeight layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomHue:
    """RandomHue layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomInvert:
    """RandomInvert layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomPerspective:
    """RandomPerspective layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomPosterization:
    """RandomPosterization layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomRotation:
    """RandomRotation layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomSaturation:
    """RandomSaturation layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomSharpness:
    """RandomSharpness layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomShear:
    """RandomShear layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomTranslation:
    """RandomTranslation layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomWidth:
    """RandomWidth layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RandomZoom:
    """RandomZoom layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class ReLU:
    """ReLU layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Rescaling:
    """Rescaling layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Resizing:
    """Resizing layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class STFTSpectrogram:
    """STFTSpectrogram layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SimpleRNN:
    """SimpleRNN layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SimpleRNNCell:
    """SimpleRNNCell layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Softmax:
    """Softmax layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Solarization:
    """Solarization layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SpatialDropout1D:
    """SpatialDropout1D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SpatialDropout2D:
    """SpatialDropout2D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SpatialDropout3D:
    """SpatialDropout3D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SpectralNormalization:
    """SpectralNormalization layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class StackedRNNCells:
    """StackedRNNCells layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class StringLookup:
    """StringLookup layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Subtract:
    """Subtract layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class TFSMLayer:
    """TFSMLayer layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class IntegerLookup(Layer):
    """IntegerLookup layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class JaxLayer(Layer):
    """JaxLayer layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class LSTM(Layer):
    """LSTM layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class LSTMCell(Layer):
    """LSTMCell layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class Lambda(Layer):
    """Lambda layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class TextVectorization:
    """TextVectorization layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class ThresholdedReLU:
    """ThresholdedReLU layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class TimeDistributed:
    """TimeDistributed layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class TorchModuleWrapper:
    """TorchModuleWrapper layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class UnitNormalization:
    """UnitNormalization layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class UpSampling1D:
    """UpSampling1D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class UpSampling2D:
    """UpSampling2D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class UpSampling3D:
    """UpSampling3D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Wrapper:
    """Wrapper layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class ZeroPadding1D:
    """ZeroPadding1D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class ZeroPadding2D:
    """ZeroPadding2D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class ZeroPadding3D:
    """ZeroPadding3D layer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass
