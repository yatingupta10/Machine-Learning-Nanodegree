from keras.models import Model, load_model
from six.moves import cPickle as pickle
import modelEval
import genPhotos

def evaluate(bboxes=False, notebook=False):
	pickle_file = 'SVHN-1.pickle'

	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)
		train_dataset = save['train_dataset']
		train_labels = save['train_labels']
		train_sequences = save['train_sequences']
		train_bboxes = save['train_bboxes']
		test_dataset = save['test_dataset']
		test_labels = save['test_labels']
		test_sequences = save['test_sequences']
		test_bboxes = save['test_bboxes']
		del save  # hint to help gc free up memory

	if bboxes:
		clf = load_model('SVHN-BB-1.h5')
		result = clf.evaluate(test_dataset, [test_sequences, test_labels[:,:,0], test_labels[:,:,1], 
		                            test_labels[:,:,2], test_labels[:,:,3], test_labels[:,:,4], test_bboxes])
		print "\n", result
		print "\nAccuracy on test set:", modelEval.accuracy(clf, test_dataset, test_sequences, test_labels)

		if notebook:
			photos_dataset, photos_labels, photos_sequences = genPhotos.generateData('mycode/photos')
		else:
			photos_dataset, photos_labels, photos_sequences = genPhotos.generateData('photos')
		print "\nAccuracy on photos:", modelEval.accuracy(clf, photos_dataset, photos_sequences, photos_labels)

	else:
		clf = load_model('SVHN-1.h5')
		result = clf.evaluate(test_dataset, [test_sequences, test_labels[:,:,0], test_labels[:,:,1], 
		                            test_labels[:,:,2], test_labels[:,:,3], test_labels[:,:,4]])
		print "\n", result
		print "\nAccuracy on test set:", modelEval.accuracy(clf, test_dataset, test_sequences, test_labels)
