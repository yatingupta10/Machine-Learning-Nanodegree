import numpy as np
from keras.models import Model, load_model
import genPhotos
import modelEval

def evaluate(notebook=False):
	if notebook:
		photos_dataset, photos_labels, photos_sequences = genPhotos.generateData('mycode/photos')
	else:
		photos_dataset, photos_labels, photos_sequences = genPhotos.generateData('photos')

	print photos_dataset.shape, photos_labels.shape, photos_sequences.shape
	print np.mean(photos_dataset)

	clf = load_model('SVHN-1.h5')
	result = clf.evaluate(photos_dataset, [photos_sequences, photos_labels[:,:,0], photos_labels[:,:,1], 
	                            photos_labels[:,:,2], photos_labels[:,:,3], photos_labels[:,:,4]])
	print "\n", result
	print "\nAccuracy on photos:", modelEval.accuracy(clf, photos_dataset, photos_sequences, photos_labels)
