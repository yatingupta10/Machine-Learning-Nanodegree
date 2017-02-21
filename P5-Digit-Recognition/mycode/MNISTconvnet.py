from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from six.moves import cPickle as pickle

def trainConvnet(batch_size=256, nb_epoch=3, validation_split=0.03, verbose=2, dense_layer_nodes=3096):
	pickle_file = 'MNIST-1.pickle'
	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)
		train_dataset = save['train_dataset']
		train_labels = save['train_labels']
		train_sequences = save['train_sequences']
		test_dataset = save['test_dataset']
		test_labels = save['test_labels']
		test_sequences = save['test_sequences']
		del save  # hint to help gc free up memory

	print 'Training set:', train_dataset.shape, train_labels.shape, train_sequences.shape
	print 'Test set:', test_dataset.shape, test_labels.shape, test_sequences.shape

	image_size = 64
	num_labels = 11
	max_sequence_length = 5
	num_channels = 1 # grayscale

	inputs = Input(shape=(num_channels, image_size, image_size))

	c1 = Convolution2D(48, 5, 5, border_mode='same')(inputs)
	a1 = Activation('relu')(c1)
	mp1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), border_mode='same')(a1)
	bn1 = BatchNormalization()(mp1)

	c2 = Convolution2D(64, 5, 5, border_mode='same')(bn1)
	a2 = Activation('relu')(c2)
	mp2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(a2)
	do2 = Dropout(0.25)(mp2)
	bn2 = BatchNormalization()(do2)

	c3 = Convolution2D(128, 5, 5, border_mode='same')(bn2)
	a3 = Activation('relu')(c3)
	mp3 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), border_mode='same')(a3)
	do3 = Dropout(0.25)(mp3)
	bn3 = BatchNormalization()(do3)

	c4 = Convolution2D(160, 5, 5, border_mode='same')(bn3)
	a4 = Activation('relu')(c4)
	mp4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(a4)
	do4 = Dropout(0.25)(mp4)
	bn4 = BatchNormalization()(do4)

	c5 = Convolution2D(192, 5, 5, border_mode='same')(bn4)
	a5 = Activation('relu')(c5)
	mp5 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), border_mode='same')(a5)
	do5 = Dropout(0.25)(mp5)
	bn5 = BatchNormalization()(do5)

	c6 = Convolution2D(192, 5, 5, border_mode='same')(bn5)
	a6 = Activation('relu')(c6)
	mp6 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(a6)
	do6 = Dropout(0.25)(mp6)
	bn6 = BatchNormalization()(do6)

	c7 = Convolution2D(192, 5, 5, border_mode='same')(bn6)
	a7 = Activation('relu')(c7)
	mp7 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), border_mode='same')(a7)
	do7 = Dropout(0.25)(mp7)
	bn7 = BatchNormalization()(do7)

	c8 = Convolution2D(192, 5, 5, border_mode='same')(bn7)
	a8 = Activation('relu')(c8)
	mp8 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(a8)
	do8 = Dropout(0.25)(mp8)
	bn8 = BatchNormalization()(do8)

	fl = Flatten()(bn8)

	d1 = Dense(dense_layer_nodes)(fl)
	a9 = Activation('relu')(d1)
	do9 = Dropout(0.25)(a9)
	bn9 = BatchNormalization()(do9)

	d3 = Dense(max_sequence_length)(bn9)
	bn10 = BatchNormalization()(d3)
	L = Activation('softmax')(bn10) # <--- Sequence length representation

	d4 = Dense(num_labels)(bn9)
	bn11 = BatchNormalization()(d4)
	S1 = Activation('softmax')(bn11) # First character/digit

	d5 = Dense(num_labels)(bn9)
	bn12 = BatchNormalization()(d5)
	S2 = Activation('softmax')(bn12)

	d6 = Dense(num_labels)(bn9)
	bn13 = BatchNormalization()(d6)
	S3 = Activation('softmax')(bn13)

	d7 = Dense(num_labels)(bn9)
	bn14 = BatchNormalization()(d7)
	S4 = Activation('softmax')(bn14)

	d8 = Dense(num_labels)(bn9)
	bn15 = BatchNormalization()(d8)
	S5 = Activation('softmax')(bn15)

	clf = Model(input=inputs, output=[L, S1, S2, S3, S4, S5]) # Seq length + five channel output

	clf.compile(loss='categorical_crossentropy', optimizer='adadelta', 
		metrics=['categorical_accuracy'])

	clf.fit(train_dataset, [train_sequences, train_labels[:,:,0], train_labels[:,:,1], 
		train_labels[:,:,2], train_labels[:,:,3], train_labels[:,:,4]], 
		batch_size=batch_size, nb_epoch=nb_epoch, validation_split=validation_split, verbose=verbose)

	clf.save('MNIST-1.h5')
	print "Training complete."
