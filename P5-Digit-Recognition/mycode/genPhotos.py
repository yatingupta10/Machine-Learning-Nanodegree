import numpy as np
import os
from scipy import ndimage
from scipy.misc import imresize

def generateData(folder):
    image_size = 64
    pixel_channels = 1
    max_sequence_length = 5
    num_labels = 11
    pixel_depth = 255.

    image_files = os.listdir(folder)

    dataset = np.ndarray(shape=(len(image_files), pixel_channels, image_size, image_size), dtype=np.float32)
    labels = np.ndarray(shape=(len(image_files), num_labels, max_sequence_length), dtype=np.bool_)
    sequences = np.ndarray(shape=(len(image_files), max_sequence_length), dtype=np.bool_)
    
    labels[:, :, :] = 0
    sequences[:, :] = 0

    num_images = 0
    skipped_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        image_data = (ndimage.imread(image_file, flatten=True).astype(float) - pixel_depth / 2) / pixel_depth

        if min(image_data.shape) > 0:
            resized_image = imresize(image_data, (image_size, image_size), interp='bilinear')
            dataset[num_images, 0, :, :] = resized_image

            if image == "10.JPG":
                labels[num_images, 1, 0] = 1
                labels[num_images, 10, 1] = 1
                labels[num_images, 0, 2:] = 1
                sequences[num_images, 1] = 1
            elif image == "14.JPG":
                labels[num_images, 1, 0] = 1
                labels[num_images, 4, 1] = 1
                labels[num_images, 0, 2:] = 1
                sequences[num_images, 1] = 1
            elif image == "17.JPG":
                labels[num_images, 1, 0] = 1
                labels[num_images, 7, 1] = 1
                labels[num_images, 0, 2:] = 1
                sequences[num_images, 1] = 1
            elif image == "2000.JPG":
                labels[num_images, 2, 0] = 1
                labels[num_images, 10, 1] = 1
                labels[num_images, 10, 2] = 1
                labels[num_images, 10, 3] = 1
                labels[num_images, 0, 4] = 1
                sequences[num_images, 3] = 1
            elif image == "24.JPG":
                labels[num_images, 2, 0] = 1
                labels[num_images, 4, 1] = 1
                labels[num_images, 0, 2:] = 1
                sequences[num_images, 1] = 1
            elif image == "25.JPG":
                labels[num_images, 2, 0] = 1
                labels[num_images, 5, 1] = 1
                labels[num_images, 0, 2:] = 1
                sequences[num_images, 1] = 1
            elif image == "3.JPG":
                labels[num_images, 3, 0] = 1
                labels[num_images, 0, 1:] = 1
                sequences[num_images, 0] = 1
            elif image == "34.JPG":
                labels[num_images, 3, 0] = 1
                labels[num_images, 4, 1] = 1
                labels[num_images, 0, 2:] = 1
                sequences[num_images, 1] = 1
            else:
                print "Image file doesn't match!"

            num_images += 1
        else:
            print("Image not loaded!")

    print "Images loaded:", num_images

    return dataset, labels, sequences


