import numpy as np
np.random.seed(0)
import os, glob
import time
import h5py
import tensorflow.keras as keras
import math
#from tensorflow.keras.utils.io_utils import HDF5Matrix
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('-e', '--epochs', default=30, type=int, help='Number of training epochs.')
    parser.add_argument('-l', '--lr_init', default=5.e-4, type=float, help='Initial learning rate.')
    parser.add_argument('-b', '--resblocks', default=3, type=int, help='Number of residual blocks.')
    parser.add_argument('-c', '--cuda', default=0, type=int, help='Which gpuid to use.')
    parser.add_argument('-a', '--load_epoch', default=0, type=int, help='Which epoch to start training from')
    args = parser.parse_args()
    
    lr_init = args.lr_init
    resblocks = args.resblocks
    epochs = args.epochs

    expt_name = 'BoostedJets-opendata_ResNet_blocks%d_pT_x1_epochs%d'%(resblocks, epochs)

#expt_name = 'quark-gluon'
#expt_name = 'ResNet_blocks%d_RH1o100_ECAL+HCAL+Trk_lr%s_gamma0.5every10ep_epochs%d'%(resblocks, str(lr_init), epochs)

#datafile = '/storage/local/data1/gpuscratch/bburkle/BoostedJets_Decorrelated_OpenData_file-1.hdf5'
#datafile = '/storage/local/data1/gpuscratch/bburkle/BoostedJets_fixed_tracks_withPix_file-1.hdf5'
#datafile = '/storage/local/data1/gpuscratch/bburkle/BoostedJets_fixed_impact_file-1.hdf5'
datafile = '/storage/local/data1/gpuscratch/bburkle/BoostedJets-split_x1_file-1.hdf5'

#def hdf5Dataset(filename, start, end):
#    x = HDF5Matrix(filename, 'X_jets', start=start, end=end)
#    y = HDF5Matrix(filename, 'y', start=start, end=end)
#    return x, y

def val_set(filename, start, end, columns=[0,1,2]):
    f = h5py.File(filename, 'r')
    #x = np.float32(f['X_jets'][start:end,...,columns])
    x = np.float32(f['X_jets'][start:end])
    x = x[...,columns]
    y = np.uint8(f['y'][start:end])
    pt = np.float32(f['pt'][start:end])
    m0 = np.float32(f['m0'][start:end])
    f.close()
    return (x, y, pt, m0)

def train_set(filename, start, end, columns=[0,1,2]):
    f = h5py.File(filename, 'r')
    x = np.asarray(f['X_jets'][start:end], dtype=np.float32)
    #x = np.asarray(f['X_jets'][start:end,...,columns], dtype=np.float32)
    #x = f['X_jets']
    x = x[...,columns]
    y = np.asarray(f['y'][start:end], dtype=np.uint8)
    #y = f['y'][start:end]
    #y = f['y']
    f.close()
    return x, y

#def make_generator(filename, start, end, batch_sz=32, columns=[0,1,2], is_train=True):
def train_generator(filename, start, end, batch_sz=32, columns=[0,1,2], is_train=True):

    w = 125*1
    h = 125*1
    classes = 2

    f = h5py.File(filename, 'r')
    x = f['X_jets']
    y = f['y']
    batches = list(range(start, end, batch_sz))
    batches.append(end)
    batches = np.column_stack((batches[:-1], batches[1:]))

    def gen():
        if is_train:
            np.random.shuffle(batches)
        for i, j in batches:
            image = np.asarray(x[i:j,:,:,columns], dtype=np.float32)
            label = np.asarray(y[i:j], dtype=np.uint8)
            label = np.eye(classes)[label.reshape(-1)]
            yield (image, label)

    image_shape = (batch_sz, w, h, len(columns))
    y_shape = (batch_sz, classes)

    dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.uint8), (image_shape, y_shape))
    return dataset

def image_data_generator(x, y, is_training=True, batch_sz=32, columns=[0,1,2], data_size = 32*10000):

    w = 125*1
    h = 125*1
    classes=2

    def map_fn(image, label):
        x = image#[:,:,:,columns]
        #x = image
        y = tf.one_hot(tf.cast(label, tf.uint8), classes)
        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    #dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.uint8))
    if is_training:
        dataset = dataset.shuffle(data_size // batch_sz)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_fn, batch_sz,
        #num_parallel_batches=4,
        drop_remainder=True if is_training else False))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


# After N batches, will output the loss and accuracy of the last batch tested
class NBatchLogger(keras.callbacks.Callback):
    def __init__(self, display=100):
        self.seen = 0
        # Display: number of batches to wait before outputting loss
        self.display = display

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size',0)
        if self.seen % self.display == 0:
            print('\n{}/{} - Batch Accuracy: {},  Batch Loss: {}\n'.format(self.seen, self.params['nb_sample'], self.params['metrics'][0], logs.get('loss')))

# Defining a callback to create calculate the AUC score of the validation set after each epoch. If that epoch gave the best AUC, the model and metrics will be saved to hdf5 files
class SaveEpoch(keras.callbacks.Callback):
    auc_best = 0.5
    def __init__(self, test_data, expt_name):
        self.test_data = test_data
        #self.y_one_hot = tf.one_hot(test_data[1], 2)
        self.folder = expt_name

    def on_epoch_end(self, epoch, logs={}):
        x = self.test_data[0]
        y = self.test_data[1]
        pt = self.test_data[2]
        m0 = self.test_data[3]
        #acc, loss = self.model.evaluate(x, self.y_one_hot, verbose=1)
        y_pred = self.model.predict(x, verbose=1)[:,1]
        fpr, tpr, _ = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        #print('\nVal Accuracy --- %.4f\tVal Loss ---- %.4f' % (acc, loss))
        print('AUC for epoch %d is: %.4f, best is: %.4f\n' % (epoch+1, roc_auc, self.auc_best))
        if roc_auc > self.auc_best:
            self.auc_best = roc_auc
            score_str = 'epoch%d_auc%.4f'%(epoch, self.auc_best)
            self.model.save_weights('MODELS/%s/%s.hdf5'%(self.folder, score_str), 'w')

            h = h5py.File('METRICS/%s/%s.hdf5'%(self.folder, score_str), 'w')
            h.create_dataset('fpr', data=fpr)
            h.create_dataset('trp', data=tpr)
            h.create_dataset('y_truth', data=y)
            h.create_dataset('y_pred', data=y_pred)
            h.create_dataset('m0', data=m0)
            h.create_dataset('pt', data=pt)
            h.close()

def LR_Decay(epoch):
    drop = 0.5
    epochs_drop = 10
    lr = lr_init * math.pow(drop, math.floor((epoch+1)/epochs_drop))
    return lr

#train_sz = 384000*2 #qg amount in paper
#valid_sz = 12950 # qg amount in paper
#test_sz = 69653 #qg amount in paper
BATCH_SZ = 32
train_sz = 32*10000
valid_sz = 32*3000
test_sz = 32*3000

# CHANNELS
# 0 - pT Tracks
# 1 - d0 Tracks
# 2 - dz Tracks
# 3 - pix l1
# 4 - pix l2
# 5 - pix l3
channels = [0]
granularity=1

if __name__ == '__main__':
    decay = ''
    print(">> Input file:",datafile)
    expt_name = '%s_%s'%(decay, expt_name)
    for d in ['MODELS', 'METRICS']:
        if not os.path.isdir('%s/%s'%(d, expt_name)):
            os.makedirs('%s/%s'%(d, expt_name))
    
   
    train_x, train_y = train_set(datafile, 0, train_sz, columns=channels)
    train_x_placeholder = tf.placeholder(train_x.dtype, train_x.shape)
    train_y_placeholder = tf.placeholder(train_y.dtype, train_y.shape)
    #train_x_placeholder = tf.placeholder(np.float32, [None, 125*3, 125*3, 6])
    #train_y_placeholder = tf.placeholder(np.uint8, [None])
    train_data = image_data_generator(train_x_placeholder, train_y_placeholder, is_training=True, batch_sz=BATCH_SZ, columns=channels, data_size=train_sz)
    

    #train_data = train_generator(datafile, start=0, end=train_sz, batch_sz=BATCH_SZ, columns=channels)
    #train_data = image_data_generator(datafile, is_training=True, batch_sz=BATCH_SZ, columns=channels)

    val_data = val_set(datafile, start=train_sz, end=train_sz+valid_sz, columns=channels)

    #from tensorflow.keras.backend.tensorflow_backend import set_session
    from tensorflow.keras.backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))

    import keras_resnet_single as networks
    #print('Training set size is:', train_x.shape[0])
    #print('Image size is:', train_x.shape[1:])
    resnet = networks.ResNet.build(len(channels), resblocks, [16,32], (125*granularity,125*granularity,len(channels)), granularity)
    if args.load_epoch != 0:
        model_name = glob.glob('MODELS/%s/epoch%d_auc*.hdf5*'%(expt_name, args.load_epoch))
        assert len(model_name) == 2
        model_name = model_name[0].split('.hdf5')[0]+'.hdf5'
        print('Loading weights from file:', model_name)
        #resnet = keras.models.load_model(model_name)
        resnet.load_weights(model_name)
    opt = keras.optimizers.Adam(lr=lr_init, epsilon=1.e-8) # changed eps to match pytorch value
    resnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    resnet.summary()

    # Model Callbacks
    print_step = 1000
    #checkpoint = SaveEpoch((val_x, val_y), expt_name)
    checkpoint = SaveEpoch(val_data, expt_name)
    batch_logger = NBatchLogger(display=print_step)
    csv_logger = keras.callbacks.CSVLogger('%s.log'%(expt_name), separator=',', append=False)
    lr_scheduler = keras.callbacks.LearningRateScheduler(LR_Decay)
    #callbacks_list=[checkpoint, batch_logger, csv_logger, lr_scheduler]
    callbacks_list=[checkpoint, csv_logger, lr_scheduler]
    #callbacks_list=[checkpoint, csv_logger]

    '''
    history = resnet.fit(
        train_data,
        steps_per_epoch = train_sz / 32,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
        initial_epoch = args.load_epoch)
    '''

    train_iterator = train_data.make_initializable_iterator()
    #train_iterator = train_data.make_one_shot_iterator()
    with tf.Session() as sess:
        sess.run(train_iterator.initializer, feed_dict={
            train_x_placeholder: train_x,
            train_y_placeholder: train_y})
        history = resnet.fit(
            train_iterator,
            steps_per_epoch = train_sz // 32,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1,
            initial_epoch = args.load_epoch)

    print('Network has finished training')

