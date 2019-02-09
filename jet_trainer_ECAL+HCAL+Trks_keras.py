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

expt_name = 'quark-gluon'
#expt_name = 'ResNet_blocks%d_RH1o100_ECAL+HCAL+Trk_lr%s_gamma0.5every10ep_epochs%d'%(resblocks, str(lr_init), epochs)

#datafile = '/uscms/home/bburkle/nobackup/working_area/ml_code/QCD_Glu_Quark/data/BoostedJets_AllJets_Compressed_file-1.hdf5'
datafile = '/uscms/home/bburkle/nobackup/working_area/ml_code/QCD_Glu_Quark/data/quark-gluon_train-set_n793900.hdf5'
#data_dir = '/eos/uscms/store/user/bburkle/E2E'
#datafile = glob.glob('%s/*.hdf5'%data_dir)


#def hdf5Dataset(filename, start, end):
#    x = HDF5Matrix(filename, 'X_jets', start=start, end=end)
#    y = HDF5Matrix(filename, 'y', start=start, end=end)
#    return x, y

def val_set(filename, start, end):
    f = h5py.File(filename, 'r')
    x = f['X_jets'][start:end]
    y = f['y'][start:end]
    pt = f['pt'][start:end]
    m0 = f['m0'][start:end]
    f.close()
    return (x, y, pt, m0)

def train_set(filename, start, end, chunk):
    f = h5py.File(filename, 'r')
    #x = np.asarray(f['X_jets'][start:end], dtype=np.float32)
    #y = np.asarray(f['y'][start:end], dtype=np.uint8)
    x = np.empty(shape=(end-start,125,125,3), dtype=np.float32)
    y = np.empty(shape=(end-start), dtype=np.uint8)
    chunks  = list(range(start, end, int((end-start)/chunk)))
    #print(chunks)
    for chunk_start, chunk_end in zip(chunks[:-1], chunks[1:]):
        print('loading chunks %d-%d'%(chunk_start, chunk_end))
        x[chunk_start-start:chunk_end-start] = f['X_jets'][chunk_start:chunk_end]
        y[chunk_start-start:chunk_end-start] = f['y'][chunk_start:chunk_end]
    f.close()
    return x, y

def image_data_generator(x, y, is_training=True, batch_sz=32):

    w = 125
    h = 125
    channels = 3
    classes=2

    def map_fn(image, label):
        #x = tf.reshape(tf.cast(image, tf.float32), (w,h,channels))
        x = image
        y = tf.one_hot(tf.cast(label, tf.uint8), classes)
        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if is_training:
        dataset = dataset.shuffle(10000)
        #pass
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_fn, batch_sz,
        #num_parallel_batches=4,
        drop_remainder=True if is_training else False))
    #dataset = dataset.map(map_fn)
    #dataset = tf.map_fn(map_fn, dataset)
    #dataset = dataset.batch(batch_sz)
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
        self.folder = expt_name

    def on_epoch_end(self, epoch, logs={}):
        #x, y = self.test_data
        x = self.test_data[0]
        y = self.test_data[1]
        pt = self.test_data[2]
        m0 = self.test_data[3]
        y_pred = self.model.predict(x, verbose=1).ravel()
        fpr, tpr, _ = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        #roc_auc = 0.
        print('\nVal Accuracy --- %.4f\tVal Loss ---- %.4f' % (acc, loss))
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

# Test input file is size 32000
#train_sz = 10400*2
#train_sz = 704000
#train_sz = 320000
train_sz = 3200#384000*2 #qg amount in paper
valid_sz = 32*100#12950 # qg amount in paper
test_sz = 69653 #qg amount in paper
BATCH_SZ = 32


if __name__ == '__main__':
    decay = 'BoostedJets'
    print(">> Input file:",datafile)
    expt_name = '%s_%s'%(decay, expt_name)
    for d in ['MODELS', 'METRICS']:
        if not os.path.isdir('%s/%s'%(d, expt_name)):
            os.makedirs('%s/%s'%(d, expt_name))

    #train_x, train_y = hdf5Dataset(datafile, 0, train_sz)
    train_x, train_y = train_set(datafile, 0, train_sz, 4)
    train_x_placeholder = tf.placeholder(train_x.dtype, train_x.shape)
    train_y_placeholder = tf.placeholder(train_y.dtype, train_y.shape)
    train_data = image_data_generator(train_x_placeholder, train_y_placeholder, is_training=True, batch_sz=BATCH_SZ)
    #train_data = image_data_generator(train_x, train_y, is_training=True, batch_sz=BATCH_SZ)
    val_x, val_y = train_set(datafile, train_sz, train_sz+valid_sz, 4)
    val_x_placeholder = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_placeholder = tf.placeholder(val_y.dtype, val_y.shape)
    val_data = image_data_generator(val_x, val_y, is_training=False)
    #val_data = image_data_generator(val_x_placeholder, val_y_placeholder, is_training=False)
    #val_data = val_set(datafile, start=train_sz, end=train_sz+valid_sz)

    #from tensorflow.keras.backend.tensorflow_backend import set_session
    from tensorflow.keras.backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))

    import keras_resnet_single as networks
    #print('Training set size is:', train_x.shape[0])
    #print('Image size is:', train_x.shape[1:])
    resnet = networks.ResNet.build(3, resblocks, [16,32], (125,125,3))
    if args.load_epoch != 0:
        model_name = glob.glob('MODELS/%s/epoch%d_auc*.hdf5'%(expt_name, args.load_epoch))[0]
        assert model_name != ''
        print('Loading weights from file:', model_name)
        #resnet = keras.models.load_model(model_name)
        resnet.load_weights(model_name)
    opt = keras.optimizers.Adam(lr=lr_init, epsilon=1.e-8) # changed eps to match pytorch value
    resnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    resnet.summary()

    # Model Callbacks
    print_step = 1000
    #checkpoint = SaveEpoch((val_x, val_y), expt_name)
    #checkpoint = SaveEpoch(val_data, expt_name)
    batch_logger = NBatchLogger(display=print_step)
    csv_logger = keras.callbacks.CSVLogger('%s.log'%(expt_name), separator=',', append=False)
    lr_scheduler = keras.callbacks.LearningRateScheduler(LR_Decay)
    #callbacks_list=[checkpoint, batch_logger, csv_logger, lr_scheduler]
    #callbacks_list=[checkpoint, csv_logger, lr_scheduler]
    #callbacks_list=[checkpoint, csv_logger]

    #history = resnet.fit(x=train_x, y=train_y, batch_size=32, epochs=epochs, verbose=1, callbacks=callbacks_list, validation_data=(val_data[0], val_data[1]), shuffle='batch', initial_epoch = args.load_epoch)
    #history = resnet.fit(train_data.make_initializer_iterator(), steps_per_epoch=train_sz/32, epochs=epochs, verbose=1, callbacks=callbacks_list, initial_epoch = args.load_epoch, shuffle='batch')
    train_iterator = train_data.make_initializable_iterator()
    #val_iterator = val_data.make_initializable_iterator()
    with tf.Session() as sess:
        sess.run(train_iterator.initializer, feed_dict={
            train_x_placeholder: train_x,
            train_y_placeholder: train_y})
            #val_x_placeholder: val_x,
            #val_y_placeholder: val_y})
        history = resnet.fit(
            #train_data.make_one_shot_iterator(),
            train_iterator,
            steps_per_epoch = train_sz / 32,
            epochs=epochs,
            #validation_data = val_data,#val_iterator,
            #validation_steps = valid_sz / 32,
            #callbacks=callbacks_list,
            verbose=1,
            initial_epoch = args.load_epoch)


    print('Network has finished training')

