import h5py
import numpy as np
import csv
import argparse
import sys

global args
parser = argparse.ArgumentParser(
  description=__doc__,
  formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('dictfile', help="Raw chunking text file", type=str) # convert/ptb.dict
parser.add_argument('trainfile', help="HDF5 train file", type=str) # convert/ptb.hdf5
parser.add_argument('validfile', help="HDF5 valid file", type=str) # convert/ptb_test.hdf5
args = parser.parse_args(sys.argv[1:])

# Load dict
idx2w = {}
with open(args.dictfile,'r') as f:
	f = csv.reader(f)
	for row in f:
		row = row[0].split(' ')
		idx2w[int(row[1])] = row[0]
	print "Dict loaded!"

# Load/translate results
with open(args.trainfile.split(".")[0].split("/")[1] + "_words.txt",'w') as f:
	train = h5py.File(args.trainfile, 'r')
	for key in train['sent_lens']:
		indices = train[str(key)]
		for row in indices:
			sentence = [idx2w[idx] for idx in row]
			sentence = ' '.join(sentence)
			f.write(sentence + '\n')
	train.close()
	print "Train file translated!"

with open(args.validfile.split(".")[0].split("/")[1] + "_words.txt",'w') as f:
	test = h5py.File(args.validfile, 'r')
	for key in test['sent_lens']:
		indices = test[str(key)]
		for row in indices:
			sentence = [idx2w[idx] for idx in row]
			sentence = ' '.join(sentence)
			f.write(sentence + '\n')
	test.close()
	print "Valid file translated!"
