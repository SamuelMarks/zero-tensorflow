from zero_tensorflow.keras import layers


def test_activation():
    import numpy as np

    layer = layers.Activation("relu")
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = layer(x)
    assert np.allclose(y.numpy(), [0.0, 0.0, 1.0])

    layer2 = layers.Activation(None)
    assert np.allclose(layer2(x).numpy(), x)


def test_activityregularization():
    _ = layers.ActivityRegularization()


def test_add():
    _ = layers.Add()


def test_additiveattention():
    _ = layers.AdditiveAttention()


def test_alphadropout():
    _ = layers.AlphaDropout()


def test_attention():
    _ = layers.Attention()


def test_augmix():
    _ = layers.AugMix()


def test_autocontrast():
    _ = layers.AutoContrast()


def test_average():
    _ = layers.Average()


def test_averagepooling1d():
    _ = layers.AveragePooling1D()


def test_averagepooling2d():
    _ = layers.AveragePooling2D()


def test_averagepooling3d():
    _ = layers.AveragePooling3D()


def test_avgpool1d():
    _ = layers.AvgPool1D()


def test_avgpool2d():
    _ = layers.AvgPool2D()


def test_avgpool3d():
    _ = layers.AvgPool3D()


def test_batchnormalization():
    _ = layers.BatchNormalization()


def test_bidirectional():
    _ = layers.Bidirectional()


def test_categoryencoding():
    _ = layers.CategoryEncoding()


def test_centercrop():
    _ = layers.CenterCrop()


def test_concatenate():
    _ = layers.Concatenate()


def test_conv1d():
    _ = layers.Conv1D()


def test_conv1dtranspose():
    _ = layers.Conv1DTranspose()


def test_conv2d():
    _ = layers.Conv2D()


def test_conv2dtranspose():
    _ = layers.Conv2DTranspose()


def test_conv3d():
    _ = layers.Conv3D()


def test_conv3dtranspose():
    _ = layers.Conv3DTranspose()


def test_convlstm1d():
    _ = layers.ConvLSTM1D()


def test_convlstm2d():
    _ = layers.ConvLSTM2D()


def test_convlstm3d():
    _ = layers.ConvLSTM3D()


def test_convolution1d():
    _ = layers.Convolution1D()


def test_convolution1dtranspose():
    _ = layers.Convolution1DTranspose()


def test_convolution2d():
    _ = layers.Convolution2D()


def test_convolution2dtranspose():
    _ = layers.Convolution2DTranspose()


def test_convolution3d():
    _ = layers.Convolution3D()


def test_convolution3dtranspose():
    _ = layers.Convolution3DTranspose()


def test_cropping1d():
    _ = layers.Cropping1D()


def test_cropping2d():
    _ = layers.Cropping2D()


def test_cropping3d():
    _ = layers.Cropping3D()


def test_cutmix():
    _ = layers.CutMix()


def test_dense():
    import numpy as np
    from zero_tensorflow import Tensor

    layer = layers.Dense(units=4, activation="relu")
    x = np.random.normal(size=(2, 8))
    y = layer(x)
    assert y.shape == (2, 4)
    assert np.all(y.numpy() >= 0)  # relu check

    assert len(layer.trainable_weights) == 2
    assert layer.kernel.value.shape == (8, 4)
    assert layer.bias.value.shape == (4,)

    # Tracing
    from zero_tensorflow import function

    @function
    def run_dense(x):
        return layer(x)

    t = Tensor(x)
    traced_y = run_dense(t)
    assert isinstance(traced_y, Tensor)


def test_depthwiseconv1d():
    _ = layers.DepthwiseConv1D()


def test_depthwiseconv2d():
    _ = layers.DepthwiseConv2D()


def test_discretization():
    _ = layers.Discretization()


def test_dot():
    _ = layers.Dot()


def test_dropout():
    _ = layers.Dropout()


def test_elu():
    _ = layers.ELU()


def test_einsumdense():
    _ = layers.EinsumDense()


def test_equalization():
    _ = layers.Equalization()


def test_flatten():
    _ = layers.Flatten()


def test_flaxlayer():
    _ = layers.FlaxLayer()


def test_gru():
    _ = layers.GRU()


def test_grucell():
    _ = layers.GRUCell()


def test_gaussiandropout():
    _ = layers.GaussianDropout()


def test_gaussiannoise():
    _ = layers.GaussianNoise()


def test_globalaveragepooling1d():
    _ = layers.GlobalAveragePooling1D()


def test_globalaveragepooling2d():
    _ = layers.GlobalAveragePooling2D()


def test_globalaveragepooling3d():
    _ = layers.GlobalAveragePooling3D()


def test_globalavgpool1d():
    _ = layers.GlobalAvgPool1D()


def test_globalavgpool2d():
    _ = layers.GlobalAvgPool2D()


def test_globalavgpool3d():
    _ = layers.GlobalAvgPool3D()


def test_globalmaxpool1d():
    _ = layers.GlobalMaxPool1D()


def test_globalmaxpool2d():
    _ = layers.GlobalMaxPool2D()


def test_globalmaxpool3d():
    _ = layers.GlobalMaxPool3D()


def test_globalmaxpooling1d():
    _ = layers.GlobalMaxPooling1D()


def test_globalmaxpooling2d():
    _ = layers.GlobalMaxPooling2D()


def test_globalmaxpooling3d():
    _ = layers.GlobalMaxPooling3D()


def test_groupnormalization():
    _ = layers.GroupNormalization()


def test_groupqueryattention():
    _ = layers.GroupQueryAttention()


def test_hashedcrossing():
    _ = layers.HashedCrossing()


def test_hashing():
    _ = layers.Hashing()


def test_identity():
    _ = layers.Identity()


def test_inputlayer():
    _ = layers.InputLayer()


def test_inputspec():
    _ = layers.InputSpec()

    _ = layers.Layer()


def test_layernormalization():
    _ = layers.LayerNormalization()


def test_leakyrelu():
    _ = layers.LeakyReLU()


def test_masking():
    _ = layers.Masking()


def test_maxnumboundingboxes():
    _ = layers.MaxNumBoundingBoxes()


def test_maxpool1d():
    _ = layers.MaxPool1D()


def test_maxpool2d():
    _ = layers.MaxPool2D()


def test_maxpool3d():
    _ = layers.MaxPool3D()


def test_maxpooling1d():
    _ = layers.MaxPooling1D()


def test_maxpooling2d():
    _ = layers.MaxPooling2D()


def test_maxpooling3d():
    _ = layers.MaxPooling3D()


def test_maximum():
    _ = layers.Maximum()


def test_melspectrogram():
    _ = layers.MelSpectrogram()


def test_minimum():
    _ = layers.Minimum()


def test_mixup():
    _ = layers.MixUp()


def test_multiheadattention():
    _ = layers.MultiHeadAttention()


def test_multiply():
    _ = layers.Multiply()


def test_normalization():
    _ = layers.Normalization()


def test_prelu():
    _ = layers.PReLU()


def test_pipeline():
    _ = layers.Pipeline()


def test_rmsnormalization():
    _ = layers.RMSNormalization()


def test_rnn():
    _ = layers.RNN()


def test_randaugment():
    _ = layers.RandAugment()


def test_randombrightness():
    _ = layers.RandomBrightness()


def test_randomcolordegeneration():
    _ = layers.RandomColorDegeneration()


def test_randomcolorjitter():
    _ = layers.RandomColorJitter()


def test_randomcontrast():
    _ = layers.RandomContrast()


def test_randomcrop():
    _ = layers.RandomCrop()


def test_randomelastictransform():
    _ = layers.RandomElasticTransform()


def test_randomerasing():
    _ = layers.RandomErasing()


def test_randomflip():
    _ = layers.RandomFlip()


def test_randomgaussianblur():
    _ = layers.RandomGaussianBlur()


def test_randomgrayscale():
    _ = layers.RandomGrayscale()


def test_randomheight():
    _ = layers.RandomHeight()


def test_randomhue():
    _ = layers.RandomHue()


def test_randominvert():
    _ = layers.RandomInvert()


def test_randomperspective():
    _ = layers.RandomPerspective()


def test_randomposterization():
    _ = layers.RandomPosterization()


def test_randomrotation():
    _ = layers.RandomRotation()


def test_randomsaturation():
    _ = layers.RandomSaturation()


def test_randomsharpness():
    _ = layers.RandomSharpness()


def test_randomshear():
    _ = layers.RandomShear()


def test_randomtranslation():
    _ = layers.RandomTranslation()


def test_randomwidth():
    _ = layers.RandomWidth()


def test_randomzoom():
    _ = layers.RandomZoom()


def test_relu():
    _ = layers.ReLU()


def test_rescaling():
    _ = layers.Rescaling()


def test_resizing():
    _ = layers.Resizing()


def test_stftspectrogram():
    _ = layers.STFTSpectrogram()


def test_separableconv1d():
    _ = layers.SeparableConv1D()


def test_separableconv2d():
    _ = layers.SeparableConv2D()


def test_separableconvolution1d():
    _ = layers.SeparableConvolution1D()


def test_separableconvolution2d():
    _ = layers.SeparableConvolution2D()


def test_simplernn():
    _ = layers.SimpleRNN()


def test_simplernncell():
    _ = layers.SimpleRNNCell()


def test_softmax():
    _ = layers.Softmax()


def test_solarization():
    _ = layers.Solarization()


def test_spatialdropout1d():
    _ = layers.SpatialDropout1D()


def test_spatialdropout2d():
    _ = layers.SpatialDropout2D()


def test_spatialdropout3d():
    _ = layers.SpatialDropout3D()


def test_spectralnormalization():
    _ = layers.SpectralNormalization()


def test_stackedrnncells():
    _ = layers.StackedRNNCells()


def test_stringlookup():
    _ = layers.StringLookup()


def test_subtract():
    _ = layers.Subtract()


def test_tfsmlayer():
    _ = layers.TFSMLayer()


def test_textvectorization():
    _ = layers.TextVectorization()


def test_thresholdedrelu():
    _ = layers.ThresholdedReLU()


def test_timedistributed():
    _ = layers.TimeDistributed()


def test_torchmodulewrapper():
    _ = layers.TorchModuleWrapper()


def test_unitnormalization():
    _ = layers.UnitNormalization()


def test_upsampling1d():
    _ = layers.UpSampling1D()


def test_upsampling2d():
    _ = layers.UpSampling2D()


def test_upsampling3d():
    _ = layers.UpSampling3D()


def test_wrapper():
    _ = layers.Wrapper()


def test_zeropadding1d():
    _ = layers.ZeroPadding1D()


def test_zeropadding2d():
    _ = layers.ZeroPadding2D()


def test_zeropadding3d():
    _ = layers.ZeroPadding3D()


def test_embedding():
    import numpy as np
    from zero_tensorflow import Tensor

    layer = layers.Embedding(input_dim=10, output_dim=4)
    x = np.array([1, 2])
    y = layer(x)
    assert y.shape == (2, 4)
    assert isinstance(y, Tensor)


def test_integerlookup():
    _ = layers.IntegerLookup()


def test_jaxlayer():
    _ = layers.JaxLayer()


def test_lstm():
    _ = layers.LSTM()


def test_lstmcell():
    _ = layers.LSTMCell()


def test_lambda():
    _ = layers.Lambda()


def test_permute():
    import numpy as np

    layer = layers.Permute(dims=(2, 1))
    x = np.zeros((2, 3, 4))
    y = layer(x)
    assert y.shape == (2, 4, 3)


def test_repeatvector():
    import numpy as np

    layer = layers.RepeatVector(n=3)
    x = np.zeros((2, 4))
    y = layer(x)
    assert y.shape == (2, 3, 4)


def test_reshape():
    import numpy as np

    layer = layers.Reshape(target_shape=(2, 2))
    x = np.zeros((3, 4))
    y = layer(x)
    assert y.shape == (3, 2, 2)


def test_layer_base_class():
    from zero_tensorflow.keras.layers import Layer
    from zero_tensorflow import Tensor

    class MyLayer(Layer):
        pass

    layer = MyLayer()
    try:
        layer(Tensor([1, 2]))
    except NotImplementedError:
        pass

    # Test build with list of inputs
    class MyListLayer(Layer):
        def call(self, inputs):
            return inputs

    ll = MyListLayer()
    ll([Tensor([1]), Tensor([2])])

    # Test build with no shape
    ll2 = MyListLayer()
    ll2(1)

    # Test weights
    layer.add_weight(shape=(2, 2), trainable=False)
    layer.add_weight(shape=(2, 2), initializer="zeros")
    layer.add_weight(shape=(2, 2), initializer=[[1, 2], [3, 4]])

    assert len(layer.weights) == 3
    assert len(layer.trainable_weights) == 2
    assert len(layer.non_trainable_weights) == 1


def test_convnd_coverage():
    from zero_tensorflow.keras.layers import Conv1D, Conv2D, Conv3D
    from zero_tensorflow import Tensor
    import numpy as np

    # 1D
    c1 = Conv1D(filters=2, kernel_size=3, padding="same", activation="relu")
    x1 = Tensor(np.ones((1, 5, 2)))
    c1(x1)

    # 2D
    c2 = Conv2D(filters=2, kernel_size=(3, 3), padding="same", activation="relu")
    x2 = Tensor(np.ones((1, 5, 5, 2)))
    c2(x2)

    # 3D
    c3 = Conv3D(filters=2, kernel_size=(3, 3, 3), padding="same")
    x3 = Tensor(np.ones((1, 5, 5, 5, 2)))
    c3(x3)

    # Data format
    c1_first = Conv1D(filters=2, kernel_size=3, data_format="channels_first")
    c1_first(Tensor(np.ones((1, 2, 5))))


def test_layers_missing_coverage():
    from zero_tensorflow.keras.layers import Conv1D, Dense, Activation, Flatten
    from zero_tensorflow import Tensor
    import numpy as np

    # Conv1D with no filters
    c_none = Conv1D()
    c_none(Tensor(np.ones((1, 5, 2))))

    # Dense activations
    for act in ["softmax", "sigmoid", "tanh", lambda x: x]:
        d = Dense(units=2, activation=act)
        d(Tensor(np.ones((1, 2))))

    # Activation layer
    for act in [None, "relu", "softmax", "sigmoid", "tanh", lambda x: x]:
        a = Activation(act)
        a(Tensor(np.ones((1, 2))))

    try:
        Activation("unknown")(Tensor([1]))
    except ValueError:
        pass

    # Flatten
    f = Flatten()
    res = f(Tensor(np.ones((2, 3, 4))))
    assert res.shape == (2, 12)

    # Flatten 0D scalar
    res0 = f(Tensor(1.0))
    assert len(res0.shape) == 1


def test_input_layer_call():
    from zero_tensorflow.keras.layers import InputLayer
    from zero_tensorflow import Tensor

    il = InputLayer()
    il(Tensor([1]))
