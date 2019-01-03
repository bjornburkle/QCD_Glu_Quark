import keras
from keras.models import Sequential
from keras import layers
from keras.layers.merge import add

# TODO
# For kernel size and stride, went from a single value 'n' to a tuple '(n,n)'. Need to make sure that this is the correct way to go from pytorch to keras
def ResBlock(x, in_channels, out_channels):
    residual = x

    downsample = out_channels//in_channels

    #TODO check the way that these layers are working, make sure it is done properly
    #TODO see if I want the batch normalization layers
    conv = layers.Conv2D(out_channels, input_shape=keras.backend.shape(x), activation='relu', kernel_size=(3,3), strides=(downsample,downsample), padding='SAME')(x)
    #conv = layers.BatchNormalization()(conv)
    conv = layers.Conv2D(out_channels, kernel_size=(3,3), padding='SAME')(conv)
    #conv = layers.BatchNormalization()(conv)
    #shortcut = layers.Conv2D(out_channels, kernel_size=1, strides=downsample)(x)

    if downsample > 1:
        #residual = shortcut
        residual = layers.Conv2D(out_channels, kernel_size=1, strides=downsample)(x)

    block = layers.Add()([conv, residual])
    #block = conv + residual
    block = layers.Activation('relu')(block)

    return block

def ResNet(in_channels, nblocks, fmaps):

    inputs = layers.Input(shape=(125,125,in_channels))

    #conv0 - changed padding from 1 to 'SAME'
    x = layers.Conv2D(fmaps[0], input_shape=keras.backend.shape(inputs), kernel_size=(7,7), strides=(2,2), padding='SAME')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x) #TODO make sure that pool_size is the same as kernal_size

    #ResBlocks
    def block_layers(_x, _nblocks, _fmaps):
        for _ in range(_nblocks):
            _x = ResBlock(_x, _fmaps[0], _fmaps[1])
        return _x

    x = block_layers(x, nblocks, [fmaps[0],fmaps[0]])
    x = block_layers(x, 1, [fmaps[0],fmaps[1]])
    x = block_layers(x, nblocks, [fmaps[1],fmaps[1]])

    #TODO get pool size
    x = layers.MaxPooling2D()(x)
    #TODO change shape of output (done using view), but may not be needed in keras
    predictions = layers.Dense(1)(x)

    return predictions

