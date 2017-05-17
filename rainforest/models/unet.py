from keras.models import Model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPool2D
from keras.layers.merge import Concatenate
from keras.layers import Input, Activation
from keras.activations import softmax
import keras.backend as K

smooth = 1.

def dice_coef(y_true, y_pred):
    """
    Compute the dice coefficient between two segmentations
    
    dice = (2*tp) / ((tp + fp) + (tp + fn))
    Only the 1st channel is used (binary segmentation)

    # Params
    - y_true : the ground truth prediction of shape 
               (batch_size, 2, width, height)
    - y_pred : predicted segmentation of shape
               (batch_size, 2, width, height)
    """
    y_true_f = K.flatten(y_true[:, 1, :, :])
    y_pred_f = K.flatten(y_pred[:, 1, :, :])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """
    Compute the dice coefficient loss between two segmentations
    
    dice_loss = 1 - dice

    # Params
    - y_true : the ground truth prediction of shape 
               (batch_size, 2, width, height)
    - y_pred : predicted segmentation of shape
               (batch_size, 2, width, height)
    """
    return 1 - dice_coef(y_true, y_pred)

def bin_cross(y_true, y_pred):
    y_true_f = K.flatten(y_true[:, 1, :, :])
    y_pred_f = K.flatten(y_pred[:, 1, :, :])
    return K.sum(binary_crossentropy(y_true_f, y_pred_f))


def flat_cent(y_true, y_pred):
    y_true = K.permute_dimensions(y_true, (0, 2, 3, 1))
    y_true = K.reshape(y_true, (-1, 2))

    y_pred = K.permute_dimensions(y_pred, (0, 2, 3, 1))
    y_pred = K.reshape(y_pred, (-1, 2))

    return K.categorical_crossentropy(y_true, y_pred)



def conv_bn_relu(inputs, n_filters, init):
    """
    Standard unet block: convolution, relu, batch_norm

    # Params
    - inputs : 4D input tensor
    - n_filters: number of filters in the conv layer
    - init : initialization

    # Returns
    - 4D output tensor
    """
    conv = Conv2D(n_filters, 3, padding='same', kernel_initializer=init, activation='relu')(inputs)
    conv = BatchNormalization(axis=1)(conv)
    return conv

def add_unet_block_cont(inputs, n_filters, init):
    """
    A Unet block in the first (contraction) stage
    conv, conv, pool

    # Params
    - inputs : 4D input tensor
    - n_filters: number of filters in the conv layer
    - init : initialization

    # Returns
    - 4D output tensor after convs (for bridge to expansion step)
    - 4D output tensor after maxpool
    """
    conv1 = conv_bn_relu(inputs, n_filters, init)
    conv2 = conv_bn_relu(inputs, n_filters, init)
    pool = MaxPool2D()(conv2)
    return conv2, pool

def add_unet_block_exp(inputs, bridge, n_filters, init):
    """
    A Unet block in the second (expansion) stage
    merge, upsample, conv, conv

    # Params
    - inputs : 4D input tensor
    - bridge: 4D tensor to concatenate
    - n_filters: number of filters in the conv layer
    - init : initialization

    # Returns
    - 4D output tensor after convs
    """    
    up = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(inputs), bridge])

    conv1 = conv_bn_relu(up, n_filters, init)
    conv2 = conv_bn_relu(conv1, n_filters, init)

    return conv2

def weighted_loss(losses):
    """
    Creates a single weighted loss function

    # Params
    - losses : a dictionary of (loss_function -> float),
    
    # Returns
    - A function taking y_true and y_pred that returns a weighted
      sum of the given losses.
    """
    def loss_func(y_true, y_pred):
        loss = 0
        for func, w in losses.items():
            loss += w * func(y_true, y_pred)
        return loss

    return loss_func

def get_unet(input_shape=(3, 256, 256), optimizer=Adam(lr=1e-3), depth=4,
              init='glorot_uniform', n_base_filters=32):
    """
    U-net with batchnorm and (possibly) 2 heads
    https://arxiv.org/abs/1505.04597

    # Params
    - input_shape : input to the network
    - optimizer : optimizer to use
    - init : Initialization for conv layers
    """
    loss = weighted_loss({dice_coef_loss: 1., flat_cent: 0.})
    metrics = [dice_coef, flat_cent]

    inputs = Input(input_shape)

    bn1 = BatchNormalization(axis=1)(inputs)

    # Contraction
    conv_layers = []
    out_layer = bn1
    for i in range(depth):
        conv, out_layer = add_unet_block_cont(out_layer, n_base_filters*2**i, init)
        conv_layers.append(conv)

    out_layer = conv_bn_relu(out_layer, n_base_filters*2**depth, init)
    out_layer = conv_bn_relu(out_layer, n_base_filters*2**depth, init)

    # Expansion
    for i in reversed(range(depth)):
        out_layer = add_unet_block_exp(out_layer, conv_layers[i], n_base_filters*2**i, init)

    outputs = []

    label = Conv2D(2, 1, kernel_initializer=init, activation='relu')(out_layer)
    segmentation = Activation(lambda x: softmax(x, axis=1), name='label')(label)
    
    model = Model(inputs=inputs, outputs=segmentation)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    model.summary()

    return model
