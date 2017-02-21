import numpy as np
from scipy.misc import imresize

def generateData(source_data, source_labels, sequences, max_sequence_length=5, option=1, insert_blanks=True):

    num_labels = 11 # 0-9 + blank
    image_size = 64

    new_dataset = np.ndarray((sequences, 1, image_size, image_size), dtype=np.float32)
    new_labels = np.ndarray((sequences, num_labels, max_sequence_length), dtype=np.bool_)
    sequence_lengths = np.ndarray((sequences, max_sequence_length), dtype=np.bool_)
    
    random_lengths = np.random.randint(1, max_sequence_length + 1, sequences)
    for sequence in range(sequences):

        sequence_length = random_lengths[sequence]
        
        # Save sequence length to return variable
        sequence_lengths[sequence, :] = 0
        sequence_lengths[sequence, sequence_length - 1] = 1
        
        # Randomly select samples from source data
        sample_indices = []
        for bar in range(sequence_length):
            sample_indices.append(np.random.randint(source_data.shape[0]))

        # Save label data from sources
        new_labels[sequence, :, :] = 0
        new_labels[sequence, 0, sequence_length:max_sequence_length] = 1 # Label "blank" classes
        for digit in range(sequence_length):
            new_labels[sequence, source_labels[sample_indices[digit]] + 1, digit] = 1

        # Pull the images from the original sources and concatenate
        sample = np.matrix(source_data[sample_indices[0], :, :])
        for character in sample_indices[1:]:
            sample = np.concatenate((sample, np.matrix(source_data[character, :, :])), axis=1)
            
        # Resize sequence image to constant width
        new_image = imresize(sample, (image_size, image_size), interp='bilinear')
        
        # Append current sample to new data and label sets
        new_dataset[sequence, 0, :, :] = new_image
        
    return new_dataset, new_labels, sequence_lengths
