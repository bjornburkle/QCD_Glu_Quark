import numpy as np
np.random.seed(0)
import os, glob
import time
import h5py
#import pyarrow as pa
#import pyarrow.parquet as pq
#import torch
#import torch.nn.functional as F
#import torch.optim as optim
#from torch.utils.data import *
import keras
from keras.utils.io_utils import HDF5Matrix
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

import argparse
parser = argparse.ArgumentParser(description='Training parameters.')
parser.add_argument('-e', '--epochs', default=30, type=int, help='Number of training epochs.')
parser.add_argument('-l', '--lr_init', default=5.e-4, type=float, help='Initial learning rate.')
parser.add_argument('-b', '--resblocks', default=3, type=int, help='Number of residual blocks.')
parser.add_argument('-c', '--cuda', default=0, type=int, help='Which gpuid to use.')
parser.add_argument('-a', '--load_argument', default=0, type=int, help='Which epoch to start training from')
args = parser.parse_args()

lr_init = args.lr_init
resblocks = args.resblocks
epochs = args.epochs
#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)

expt_name = 'ResNet_blocks%d_RH1o100_ECAL+HCAL+Trk_lr%s_gamma0.5every10ep_epochs%d'%(resblocks, str(lr_init), epochs)

'''
class hdf5Dataset(Dataset):
    def __init__(self, filename):
        self.hdf5 = h5py.File(filename)
        self.cols = None # read all columns
    def __getitem__(self, index):
        data = self.hdf5
        data['X_jets'] = np.float32(data['X_jets'][0])
        data['y'] = np.float32(data['y'])
        data['m0'] = np.float32(data['m0'])
        data['pt'] = np.float32(data['pt'])
        # Preprocessing
        data['X_jets'][data['X_jets'] < 1.e-3] = 0. # Zero-Suppresion
        data['X_jets'] = 25.*data['X_jets'][-1,...] # For HCAL: to match pixel intensity distn of other layers
        data['X_jets'] = data['X_jets']/100. # To standardize
        return dict(data)
    def __len__(self):
        return self.hdf5.shape[1]
'''

def hdf5Dataset(filename, start, end):
    x = HDF5Matrix(filename, 'X_jets', start=start, end=end)[0]
    x = x[x < 1e-3] = 0. # Zero-Suppresion
    x = 25.*x[-1,...] # For HCAL: to match pixel intensity distn of other layers
    x = x/100. # To standardize
    y = HDF5Matrix(filename, 'y', start=start, end=end)
    return x, y

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
    self.auc_best = 0.5
    def __init__(self, test_data, _expt_name):
        self.test_data = test_data
        self.folder = _expt_name

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        y_pred = model.predict(x, verbose=1)
        fpr, tpr, _ = roc_curve(y, y_pred)
        auc = roc_curve(fpr, tpr)
        print('\nAUC for epoch %d is: %.4f, best is %.4f' % (epoch, auc, self.auc_best))
        if auc > self.auc_best:
            self.auc_best = auc
            score_str = 'epoch%d_auc%.4f'%(epoch, self.auc_best)
            model.save_weights('MODELS/%s/%s.hdf5'%(self.folder, score_str))

            h = h5py.File('METRICS/%s/%s.hdf5'%(self.folder, score_str))
            h.create_dataset('fpr', data=fpr)
            h.create_dataset('trp', data=tpr)
            h.create_dataset('y_truth', data=y)
            h.create_dataset('y_pred', data=y_pred)
            #h.create_dataset('m0', data=x['m0'])
            #h.create_dataset('pt', data=x['pt'])
            h.close()

# TODO change the decay and decays variables to read the boosted jet files
decay = 'QCDToGGQQ_IMGjet_RH1all'
decays = glob.glob('IMG/%s_jet0_run?_n*.train.snappy.parquet'%decay)
print(">> Input files:",decays)
#assert len(decays) == 3, "len(decays) = %d"%(len(decays))
expt_name = '%s_%s'%(decay, expt_name)
for d in ['MODELS', 'METRICS']:
    if not os.path.isdir('%s/%s'%(d, expt_name)):
        os.makedirs('%s/%s'%(d, expt_name))
# TODO
# Make it so the pt and m0 variables are not passed as an input to the NN, but still make it so I have access to those variables when saving network metrics

train_sz = 700000
valid_sz = 50000
test_sz = 50000

train_x, train_y = hdf5Dataset(datafile, 0, train_sz)
val_x, val_y = hdf5Dataset(datafile, trin_sz, train_sz+valid_sz)

#TODO make sure I am properly running overgpu
#possible method 1, tf backend
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
set_session(tf.Session(config=config))
#possible method 2, theano backend
import theano
theano.config.device='gpu'
theano.config.floatX='float32'

import keras_resnet_single as networks
resnet = networks.ResNet(3, resblocks, [16, 32])
if args.load_epoch != 0:
    model_name = glob.glob('MODELS/%s/model_epoch%d_auc*.hdf5'%(expt_name, args.load_epoch))[0]
    assert model_name != ''
    print('Loading weights from file:', model_name)
    #resnet = keras.models.load_model(model_name)
    resnet = keras.models.load_weights(model_name)
resnet.compile(loss='binary_cross_entropy', optimizer=Adam(lr=lr_init), metrics=['accuracy'])
resnet.summary()

#checkpoint = resnet.callbacks.ModelCheckpoint('MODELS/{expt_name:s}/model_epoch{epoch:02d}_auc{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
print_step = 1000
checkpoint = SaveEpoch((val_x, val_y), expt_name, agrs.load_epoch)
batch_logger = NBatchLogger(display=print_step)
csv_logger = keras.callbacks.CSVLogger('%s.log'%(expt_name), separator=',', append=False)
callbacks_list=[checkpoint, batch_logger, csv_logger]

history = resnet.Fit(x=train_x, y=train_y, epochs=epochs, verbose=1, callbacks=callbacks_list, validation_data=(val_x, val_y), shuffle=True, initial_epoch = args.load_epoch)

print('Network has finished training')

# TODO Make sure I also save the weights and everything for the final epoch, even if it isn't the best one
