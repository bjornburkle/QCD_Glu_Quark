import random
random.seed(1337)
import h5py
import time
import numpy as np
import glob, re

import pyarrow as pa
import pyarrow.parquet as pq

import argparse
parser = argparse.ArgumentParser(add_help=True, description='Set what file to do')
parser.add_argument('-f', '--file', type=int, help='Which file to process')
args = parser.parse_args()

def np2arrowArray(x):
    if len(x.shape) > 1:
        x = np.transpose(x, [2,0,1])
        return pa.array([x.tolist()])
    else:
        return pa.array([x.tolist()])

def convert_to_Parquet(decays, start, stop, chunk_size, expt_name, set_name, jets_per_file):
    
    # Open the input HDF5 file
    dsets = [h5py.File('%s'%decay, 'r') for decay in decays]
    #keys = ['X_jets', 'jetPt', 'jetM', 'y_jets'] # key names in in put hdf5
    keys = ['X_jets', 'pt', 'm0', 'y'] # desired key names in output parquet
    row0 = [np2arrowArray(dsets[0][key][0]) for key in keys]
    keys = ['X_jets', 'pt', 'm0', 'y'] # desired key names in output parquet
    table0 = pa.Table.from_arrays(row0, keys) 
    
    # Open the output Parquet file
    #filename = '%s.%s.snappy.parquet'%(expt_name,set_name)
    filename = '%s.%d.parquet' % (expt_name, set_name)
    writer = pq.ParquetWriter(filename, table0.schema, compression='snappy')

    # Loop over file chunks of size chunk_size
    nevts = stop - start
    #for i in range(nevts//chunk_size):
    for i in range(int(np.ceil(1.*jets_per_file/chunk_size))):
        
        begin = start + (set_name-1)*jets_per_file + i*chunk_size
        end = begin + chunk_size

        # Load array chunks into memory
        X = np.concatenate([dset['X_jets'][begin:end] for dset in dsets])
        pt = np.concatenate([dset['pt'][begin:end] for dset in dsets])
        m = np.concatenate([dset['m0'][begin:end] for dset in dsets])
        y = np.concatenate([dset['y'][begin:end] for dset in dsets])
        
        # Shuffle
        #l = list(zip(X, pt, m, y))
        #random.shuffle(l)
        #X, pt, m, y = zip(*l)

        # Convert events in the chunk one-by-one
        print('Doing events: [%d->%d)'%(begin,end))
        for j in range(len(y)):

            # Create a list for each sample
            sample = [
                np2arrowArray(X[j]),
                np2arrowArray(pt[j]),
                np2arrowArray(m[j]),
                np2arrowArray(y[j]),
            ]

            table = pa.Table.from_arrays(sample, keys)

            writer.write_table(table)

    writer.close()
    for dset in dsets:
        dset.close()
    return filename
    
# MAIN
chunk_size = 320
jetId = 0
nevts_total = 747200
evts_per_file = 46700
sets = range(nevts_total // evts_per_file)
expt_name = 'BoostedJets_opendata_x3'
decays = ['BoostedJets_x3_file-1.hdf5']


all_files = False
if args.file == 0:
    all_files = True
assert args.file <= len(sets)

#for set_name in list(['train', 'test']):
for set_name in sets:

    set_name += 1
    if not all_files and set_name != args.file:
        continue

    for runId in range(1):

        print(' >> Doing runId: %d'%runId)

        #decays = glob.glob('QCD_Pt_80_170_%s_IMGjet_n*_label?_jet%d_run%d.hdf5'%(list_idx, jetId, runId))
        print(' >>',decays)
        #assert len(decays) == 2
        #nevts_total = decays[0].split("_")[-4][1:]
        #nevts_total = int(nevts_total)
        print(' >> Total events per file:', evts_per_file)

        start, stop = 0, nevts_total


        now = time.time()
        f = convert_to_Parquet(decays, start, stop, chunk_size, expt_name, set_name, evts_per_file)
        print(' >> %s time: %.2f'%(set_name,time.time()-now))

        reader = pq.ParquetFile(f)
        for i in range(10):
            print(i, reader.read_row_group(i).to_pydict()['y'])
        print(' >> Total events written:',reader.num_row_groups)
