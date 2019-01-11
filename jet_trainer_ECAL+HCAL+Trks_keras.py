import numpy as np
np.random.seed(0)
import os, glob
import time
import h5py
import keras
import math
from keras.utils.io_utils import HDF5Matrix
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

import argparse
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
#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)

#expt_name = 'ResNet_blocks%d_RH1o100_ECAL+HCAL+Trk_lr%s_gamma0.5every10ep_epochs%d'%(resblocks, str(lr_init), epochs)
expt_name = 'keras_small_test'

#datafile = 'IMG/test_BoostedJets.hdf5'
datafile = 'root://cmsxrootd.fnal.gov//store/user/bburkle/E2E/BoostedJets_AllJets.hdf5'
#datafile = '/eos/uscms/store/user/bburkle/E2E/BoostedJets_AllJets.hdf5'

def hdf5Dataset(filename, start, end):
    x = HDF5Matrix(filename, 'X_jets', start=start, end=end)
    # These steps are being moved to hdf5 conversion step
    #x[x < 1.e-3] = 0. # Zero-Suppresion
    #x[-1,...] = 25.*x[-1,...] # For HCAL: to match pixel intensity distn of other layers
    #x = x/100. # To standardize
    y = HDF5Matrix(filename, 'y', start=start, end=end)
    return x, y

def val_set(filename, start, end):
    f = h5py.File(filename, 'r')
    x = f['X_jets'][start:end]
    y = f['y'][start:end]
    pt = f['pt'][start:end]
    m0 = f['m0'][start:end]
    f.close()
    return (x, y, pt, m0)

def train_set(filename, start, end):
    f = h5py.File(filename, 'r')
    x = f['X_jets'][start:end]
    y = f['y'][start:end]
    f.close()
    return x, y

# For this to work right, may need to split training, validation, and testing data sets into 3 separate input files
# Can also put the different datasets as different groups in the same hdf5 file
class DataGenerator(keras.utils.Sequence):
    def __init__(self, filename, batch_size=32, data_split=100, start=0, end=32*100, shuffle=True):
        self.hf = h5py.File(filename, 'r')
        self.x_jets = self.hf['X_jets'][start:end]
        self.y = self.hf['y'][start:end]
        #self.total_len = len(self.hf['y'])
        self.total_len = end - start
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = 0
        self.data_split = data_split
        self.len_segment = math.floor(self.total_len / data_split)
        self.seg_list = list(range(0, self.total_len, self.len_segment))
        print(self.total_len, self.len_segment)
        print(self.seg_list)
        self.cur_seg_idx = 0
        #self.x_cur = self.hf['X_jets'][self.start:self.end][:self.len_segment]
        self.x_cur = self.x_jets[:self.len_segment]
        #self.y_cur = self.hf['y'][self.start:self.end][:self.len_segment]
        self.y_cur = self.y[:self.len_segment]

    def __len__(self):
        return self.data_split

    def next_seg(self):
        #self.cur_seg_idx += self.len_segment
        #print(self.cur_seg_idx)
        chunk_start = self.seg_list[self.cur_seg_idx]
        #chunk_end = self.seg_list[self.cur_seg_idx+1]
        self.cur_seg_idx += 1
        chunk_end = chunk_start + self.len_segment
        self.x_cur = self.x_jets[chunk_start:chunk_end]
        self.y_cur = self.y[chunk_start:chunk_end]
        if self.cur_seg_idx == len(self.seg_list)-1:
            self.cur_seg_idx = 1
            self._on_epoch_end()

    def _on_epoch_end(self):
        if self.shuffle:
           #print('Shuffle Data Segments')
           np.random.shuffle(self.seg_list) 

    #def __getitem__(self, idx):
    def generate(self):
        while 1:
            idx = self.idx
            if idx >= self.len_segment:
                self.next_seg()
                idx = 0

            if idx + self.batch_size >= self.len_segment:
                batch_x = self.x_cur[idx:]
                batch_y = self.y_cur[idx:]
            else:
                batch_x = self.x_cur[idx:(idx + self.batch_size)]
                batch_y = self.y_cur[idx:(idx + self.batch_size)]
            self.idx = idx + self.batch_size
            #return batch_x, batch_y
            yield batch_x, batch_y

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

# TODO change the decay and decays variables to read the boosted jet files
decay = 'BoostedJets'
print(">> Input file:",datafile)
expt_name = '%s_%s'%(decay, expt_name)
for d in ['MODELS', 'METRICS']:
    if not os.path.isdir('%s/%s'%(d, expt_name)):
        os.makedirs('%s/%s'%(d, expt_name))
# TODO
# Make it so the pt and m0 variables are not passed as an input to the NN, but still make it so I have access to those variables when saving network metrics

# Test input file is size 32000
#train_sz = 10400*2
train_sz = 700000
valid_sz = 48000
test_sz = 48000

#train_x, train_y = hdf5Dataset(datafile, 0, train_sz)
train_x, train_y = train_set(datafile, 0, train_sz)
val_data = val_set(datafile, start=train_sz, end=train_sz+valid_sz)
#training_generator = DataGenerator(datafile, batch_size=32, data_split=100, start=0, end=train_sz).generate()
#validation_generator = DataGenerator(datafile, batch_size=32, data_split=10, start=train_sz, end=train_sz+valid_sz).generate()
#val_x, val_y = hdf5Dataset(datafile, train_sz, train_sz+valid_sz)

#TODO make sure I am properly running overgpu
#possible method 1, tf backend
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
#possible method 2, theano backend
#import theano
#theano.config.device='gpu'
#theano.config.floatX='float32'

import keras_resnet_single as networks
#print('Training set size is:', train_x.shape[0])
#print('Image size is:', train_x.shape[1:])
resnet = networks.ResNet.build(3, resblocks, [16,32], (125,125,3))
if args.load_epoch != 0:
    model_name = glob.glob('MODELS/%s/model_epoch%d_auc*.hdf5'%(expt_name, args.load_epoch))[0]
    assert model_name != ''
    print('Loading weights from file:', model_name)
    #resnet = keras.models.load_model(model_name)
    resnet = keras.models.load_weights(model_name)
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

#history = resnet.fit_generator(training_generator, steps_per_epoch=100, epochs=epochs, verbose=1, callbacks=callbacks_list, workers=10, initial_epoch=args.load_epoch, shuffle=False)
#history = resnet.fit_generator(training_generator, steps_per_epoch=train_sz/32, epochs=epochs, verbose=1, validation_data=validation_generator, validation_steps=10, callbacks=callbacks_list, workers=10, initial_epoch=args.load_epoch, shuffle=False)
history = resnet.fit(x=train_x, y=train_y, batch_size=32, epochs=epochs, verbose=1, callbacks=callbacks_list, validation_data=(val_data[0], val_data[1]), shuffle='batch', initial_epoch = args.load_epoch)

print('Network has finished training')

# TODO Make sure I also save the weights and everything for the final epoch, even if it isn't the best one
