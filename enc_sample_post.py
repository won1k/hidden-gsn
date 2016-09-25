import h5py
import numpy as np
import csv

global args
parser = argparse.ArgumentParser(
  description=__doc__,
  formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('dictfile', help="Raw chunking text file", type=str) # ptb/train.txt
parser.add_argument('testfile', help="Raw chunking test text file", type=str) # ptb/test.txt
parser.add_argument('outputfile', help="HDF5 output file", type=str) # convert_seq/ptb_seq
parser.add_argument('dwin', help="Window dimension (0 if no padding)", type=int) # 5
args = parser.parse_args(arguments)

# Load dict
idx2w = {}
with open('convert_seq/ptb_seq.dict','r') as f:
	f = csv.reader(f)
	for row in f:
		row = row[0].split(' ')
		idx2w[int(row[1])] = row[0]
	print "Dict loaded!"

# Load/translate results
with open('enc_ptb_results_noise_train_words.txt','w') as f:
	train = h5py.File('enc_ptb_results_noise_train.hdf5', 'r')
	for key in train.keys():
		indices = np.transpose(train[key])
		for row in indices:
			sentence = [idx2w[idx] for idx in row]
			sentence = ' '.join(sentence)
			f.write(sentence + '\n')
	train.close()
	print "Train file translated!"

with open('enc_ptb_results_noise_valid_words.txt','w') as f:
	test = h5py.File('enc_ptb_results_noise_valid.hdf5', 'r')
	for key in test.keys():
		indices = np.transpose(test[key])
		for row in indices:
			sentence = [idx2w[idx] for idx in row]
			sentence = ' '.join(sentence)
			f.write(sentence + '\n')
	test.close()
	print "Valid file translated!"
