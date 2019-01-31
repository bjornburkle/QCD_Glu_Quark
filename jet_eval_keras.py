import h5py
import os, glob, re
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import keras
import keras_resnet_single as networks

#from jet_trainer_ECAL+HCAL+Trks_keras import train_sz, valid_sz, test_sz, val_set, datafile
from jet_trainer_imports import train_sz, valid_sz, test_sz, val_set, datafile

import argparse
parser = argparse.ArgumentParser(description='Evaluation parameters.')
parser.add_argument('-e', '--epoch', required=True, type=int, help='Training epoch to use for evaluation.')
parser.add_argument('-s', '--expt_name', required=True, type=str, help='String            corresponding to expt_name.')
parser.add_argument('-c', '--cuda', default=0, type=int, help='Which gpuid to use.')
args = parser.parse_args()

epoch = args.epoch
expt_name = args.expt_name
#nblocks = int(re.search('blocks([0-9]+?)_', expt_name).group(1))
nblocks = 3

# Uncomment following lines if you dont want to use the vars from the training script
#datafile = ''
#train_sz = 0
#valid_sz = 0
#test_sz = 0

#model_file = glob.glob('MODELS/%s/model_epoch%d_auc*.hdf5'%(expt_name, epoch))[0]
model_file = glob.glob('MODELS/%s/epoch%d_auc*.hdf5'%(expt_name, epoch))[0]
for d in ['METRICS_TEST']:
    if not os.path.isdir('%s/%s'%(d, expt_name)):
        os.mkdir('%s/%s'%(d, expt_name))

x, y, pt, m0 = val_set(datafile, start=train_sz+valid_sz, end=train_sz+valid_sz+test_sz)

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


resnet = networks.ResNet.build(3, nblocks, [16,32], (125,125,3))
#resnet = keras.models.load_weights(model_name)
resnet.load_weights(model_file)

y_pred = resnet.predict(x, verbose=1).ravel()
fpr, tpr, _ = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)

score_str = 'epoch%d_auc%.4f'%(epoch, roc_auc)
h = h5py.File('METRICS_TEST/%s/metrics_%s.hdf5'%(expt_name, score_str), 'w')
h.create_dataset('fpr', data=fpr)
h.create_dataset('tpr', data=tpr)
h.create_dataset('y_truth', data=y)
h.create_dataset('y_pred', data=y_pred)
h.create_dataset('m0', data=m0)
h.create_dataset('pt', data=pt)
h.close()
