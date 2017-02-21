import os
import tarfile
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve
import h5py
import genSVHNdict

def prepData():
	url = 'http://ufldl.stanford.edu/housenumbers/'

	def get_SVHN(filename, expected_bytes, force=False):
		if force or not os.path.exists(filename):
			print 'Downloading:', filename
			filename, _ = urlretrieve(url + filename, filename)
			print 'Download Complete'
		statinfo = os.stat(filename)
		if statinfo.st_size == expected_bytes:
			print 'Found and verified', filename
		else:
			raise Exception(
				'Failed to verify ' + filename + '. Can you get to it with a browser?')
		return filename

	train_filename = get_SVHN('train.tar.gz', 404141560)
	test_filename = get_SVHN('test.tar.gz', 276555967)

	def extract(filename, force=False):
		root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
		if os.path.isdir(root) and not force:
			print '%s already present - Skipping extraction of %s.' % (root, filename)
		else:
			print 'Extracting data for %s. This may take a while. Please wait.' % root
			tar = tarfile.open(filename)
			tar.extractall()
			tar.close()
		data_folders = root
		print data_folders, "data extraction complete."
		return data_folders
		
	train_folders = extract(train_filename)
	test_folders = extract(test_filename)

	pickle_file = 'SVHN-dictionaries.pickle'

	if not os.path.exists(pickle_file):
		print "Creating dictionaries from digitStruct files."
		train_struct = h5py.File('train/digitStruct.mat')
		train_dictionary, train_samples = genSVHNdict.createDictionary(train_struct)
		print "Train dictionary complete."
		print "Train samples:", train_samples

		test_struct = h5py.File('test/digitStruct.mat')
		test_dictionary, test_samples = genSVHNdict.createDictionary(test_struct)
		print "Test dictionary complete."
		print "Test samples:", test_samples

		try:
			f = open(pickle_file, 'wb')
			save = {
				'train_dictionary': train_dictionary,
				'test_dictionary': test_dictionary,
			}
			pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
			f.close()
		except Exception as e:
			print 'Unable to save data to', pickle_file, ':', e
			raise
			
	statinfo = os.stat(pickle_file)
	print 'Compressed pickle size:', statinfo.st_size
