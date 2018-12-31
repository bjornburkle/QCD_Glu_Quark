import keras
from keras.models import Sequentual
from keras import layers
from keras.layers.merge import add

def ResBlock(x, in_channels, out_channels):
    residual = x

    downsample = out_channels//in_channels

    #TODO check the way that these layers are working, make sure it is done properly
    #TODO make sure that I want the batch normalization layers
    conv = layers.Conv2D(out_channels, activation='relu', kernel_size=(3,3), stride=downsample, padding=1)(x)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Conv2D(out_channels, kernel_size=3, kernel_size=(3,3), padding=1)(conv)
    conv = layers.BatchNormalization()(conv)
    #shortcut = layers.Conv2D(out_channels, kernel_size=1, stride=downsample)(x)

    if downsample > 1:
        #residual = shortcut
        residual = layers.Conv2D(out_channels, kernel_size=1, stride=downsample)(x)

    block = layers.merge.add([conv, residual])
    block = layers.Activation('relu')(block)

    return block

def ResNet(in_channels, nblocks, fmaps):

    model = Sequential() 

    #conv0
    model.add(layers.Conv2D(fmaps[0], kernal_size=7, stride=2, padding=1))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=2)) #TODO make sure that pool_size is the same as kernal_size

    #ResBlocks
    model.add(block_layers(nblocks, [fmaps[0],fmaps[0]])
    model.add(block_layers(1, [fmaps[0],fmaps[1]])
    model.add(block_layers(nblocks, [fmaps[1],fmaps[1]])

    #TODO get pool size
    model.add(layers.MaxPooling2D())
    #TODO change shape of output (done using view), but may not be needed in keras
    model.add(layers.Dense(1))

    def block_layers(_nblocks, _fmaps):
        _layers = Sequentual()
        for _ in range(_nblocks)
            _layers.add(ResBlock(_fmaps[0], _fmaps[1]))
        return _layers

    return model

