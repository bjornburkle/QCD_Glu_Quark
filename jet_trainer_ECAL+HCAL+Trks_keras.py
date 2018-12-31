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
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)

expt_name = 'ResNet_blocks%d_RH1o100_ECAL+HCAL+Trk_lr%s_gamma0.5every10ep_epochs%d'%(resblocks, str(lr_init), epochs)

class ParquetDataset(Dataset):
    def __init__(self, filename):
        self.parquet = pq.ParquetFile(filename)
        self.cols = None # read all columns
        #self.cols = ['X_jets.list.item.list.item.list.item','y'] 
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()
        data['X_jets'] = np.float32(data['X_jets'][0]) 
        data['y'] = np.float32(data['y'])
        data['m0'] = np.float32(data['m0'])
        data['pt'] = np.float32(data['pt'])
        # Preprocessing
        data['X_jets'][data['X_jets'] < 1.e-3] = 0. # Zero-Suppression
        data['X_jets'][-1,...] = 25.*data['X_jets'][-1,...] # For HCAL: to match pixel intensity distn of other layers
        data['X_jets'] = data['X_jets']/100. # To standardize
        return dict(data)
    def __len__(self):
        return self.parquet.num_row_groups

# This should read in the hdf5 files in the same fashion that the parquet files were read. However, need to change some more stuff down 
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

class NBatchLogger(keras.callbacks.Callback):
    def __init__(self, display=100):
        self.seen = 0
        # Display: number of batches to wait before outputting loss
        self.display = display

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size',0)
        if self.seen % self.display == 0:
            print('\n{}/{} - Batch Accuracy: {},  Batch Loss: {}\n'.format(self.seen, self.params['nb_sample'], self.params['metrics'][0], logs.get('loss')))

# Defining a callback to create an hdf5 file with the validation set metric at the end of every epoch
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
            h.create_dataset('m0', data=x['m0'])
            h.create_dataset('pt', data=x['pt'])
            h.close()

decay = 'QCDToGGQQ_IMGjet_RH1all'
decays = glob.glob('IMG/%s_jet0_run?_n*.train.snappy.parquet'%decay)
print(">> Input files:",decays)
#assert len(decays) == 3, "len(decays) = %d"%(len(decays))
expt_name = '%s_%s'%(decay, expt_name)
for d in ['MODELS', 'METRICS']:
    if not os.path.isdir('%s/%s'%(d, expt_name)):
        os.makedirs('%s/%s'%(d, expt_name))

# TODO change this so it concantenates the hdf5 input files and caches them in a way that doesn't use up too much RAM
# For pytorch, we can tell it to only send the image to the GPU to train on and thus ignore the other inputs, but this needs to be changed so we do not train on the other input variables
train_cut = 2*384*1000 # CMS OpenData study
dset_train = ConcatDataset([ParquetDataset(d) for d in decays])
idxs = np.random.permutation(len(dset_train))
train_sampler = sampler.SubsetRandomSampler(idxs[:train_cut])
train_loader = DataLoader(dataset=dset_train, batch_size=32, num_workers=10, sampler=train_sampler, pin_memory=True)

dset_val = ConcatDataset([ParquetDataset(d) for d in decays])
val_sampler = sampler.SubsetRandomSampler(idxs[train_cut:])
val_loader = DataLoader(dataset=dset_val, batch_size=120, num_workers=10, sampler=val_sampler)


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
#if args.load_epoch == 0:
#    resnet.compile(loss='binary_cross_entropy', optimizer=Adam(lr=lr_init), metrics=['accuracy'])
#
#else:
if args.load_epoch != 0:
    model_name = glob.glob('MODELS/%s/model_epoch%d_auc*.hdf5'%(expt_name, args.load_epoch))[0]
    assert model_name != ''
    print('Loading weights from file:', model_name)
    #resnet = keras.models.load_model(model_name)
    resnet = keras.models.load_weights(model_name)
resnet.compile(loss='binary_cross_entropy', optimizer=Adam(lr=lr_init), metrics=['accuracy'])
resnet.summary()

#checkpoint = resnet.callbacks.ModelCheckpoint('MODELS/{expt_name:s}/model_epoch{epoch:02d}_auc{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
checkpoint = SaveEpoch((val_x, val_y), expt_name, agrs.load_epoch)
batch_logger = NBatchLogger(display=print_step)
csv_logger = keras.callbacks.CSVLogger('%s.log'%(expt_name), separator=',', append=False)
callbacks_list=[checkpoint, batch_logger, csv_logger]

history = resnet.Fit(x=train_x, y=train_y, epochs=epochs, verbose=1, callbacks=callbacks_list, validation_data=(val_x, val_y), shuffle=True, initial_epoch = args.load_epoch)


'''
def do_eval(resnet, val_loader, f, roc_auc_best, epoch):
    global expt_name
    loss_, acc_ = 0., 0.
    y_pred_, y_truth_, m0_, pt_ = [], [], [], []
    now = time.time()
    for i, data in enumerate(val_loader):
        X, y, m0, pt = data['X_jets'].cuda(), data['y'].cuda(), data['m0'], data['pt']
        logits = resnet(X)
        loss_ += F.binary_cross_entropy_with_logits(logits, y).item()
        pred = logits.ge(0.).byte()
        acc_ += pred.eq(y.byte()).float().mean().item()
        y_pred = torch.sigmoid(logits)
        # Store batch metrics:
        y_pred_.append(y_pred.tolist())
        y_truth_.append(y.tolist())
        m0_.append(m0.tolist())
        pt_.append(pt.tolist())

    now = time.time() - now
    y_pred_ = np.concatenate(y_pred_)
    y_truth_ = np.concatenate(y_truth_)
    m0_ = np.concatenate(m0_)
    pt_ = np.concatenate(pt_)
    s = '%d: Val time:%.2fs in %d steps'%(epoch, now, len(val_loader))
    print(s)
    f.write('%s\n'%(s))
    s = '%d: Val loss:%f, acc:%f'%(epoch, loss_/len(val_loader), acc_/len(val_loader))
    print(s)
    f.write('%s\n'%(s))

    fpr, tpr, _ = roc_curve(y_truth_, y_pred_)
    roc_auc = auc(fpr, tpr)
    s = "VAL ROC AUC: %f"%(roc_auc)
    print(s)
    f.write('%s\n'%(s))

    if roc_auc > roc_auc_best:
        roc_auc_best = roc_auc
        f.write('Best ROC AUC:%.4f\n'%roc_auc_best)
        score_str = 'epoch%d_auc%.4f'%(epoch, roc_auc_best)

        filename = 'MODELS/%s/model_%s.pkl'%(expt_name, score_str)
        model_dict = {'model': resnet.state_dict(), 'optim': optimizer.state_dict()}
        torch.save(model_dict, filename)

        h = h5py.File('METRICS/%s/metrics_%s.hdf5'%(expt_name, score_str), 'w')
        h.create_dataset('fpr', data=fpr)
        h.create_dataset('tpr', data=tpr)
        h.create_dataset('y_truth', data=y_truth_)
        h.create_dataset('y_pred', data=y_pred_)
        h.create_dataset('m0', data=m0_)
        h.create_dataset('pt', data=pt_)
        h.close()

    return roc_auc_best
'''

# MAIN #
#eval_step = 1000
print_step = 1000
roc_auc_best = 0.5
print(">> Training <<<<<<<<")
f = open('%s.log'%(expt_name), 'w')
for e in range(epochs):

    epoch = e+1
    s = '>> Epoch %d <<<<<<<<'%(epoch)
    print(s)
    f.write('%s\n'%(s))

    # Run training
    lr_scheduler.step()
    resnet.train()
    now = time.time()
    for i, data in enumerate(train_loader):
        X, y = data['X_jets'].cuda(), data['y'].cuda()
        optimizer.zero_grad()
        logits = resnet(X)
        loss = F.binary_cross_entropy_with_logits(logits, y).cuda()
        loss.backward()
        optimizer.step()
        if i % print_step == 0:
            pred = logits.ge(0.).byte()
            acc = pred.eq(y.byte()).float().mean()
            s = '%d: Train loss:%f, acc:%f'%(epoch, loss.item(), acc.item())
            print(s)
        # For more frequent validation:
        #if epoch > 1 and i % eval_step == 0:
        #    resnet.eval()
        #    roc_auc_best = do_eval(resnet, val_loader, f, roc_auc_best, epoch)
        #    resnet.train()

    f.write('%s\n'%(s))
    now = time.time() - now
    s = '%d: Train time:%.2fs in %d steps'%(epoch, now, len(train_loader))
    print(s)
    f.write('%s\n'%(s))

    # Run Validation
    resnet.eval()
    roc_auc_best = do_eval(resnet, val_loader, f, roc_auc_best, epoch)

f.close()
