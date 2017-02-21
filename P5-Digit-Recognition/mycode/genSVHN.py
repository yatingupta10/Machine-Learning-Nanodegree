import os
import numpy as np
from scipy import ndimage
from scipy.misc import imresize

def generateData(folder, dictionary):
    
    image_size = 64
    pixel_channels = 1
    max_sequence_length = 5
    num_labels = 11
    pixel_depth = 255.
    bbox_dims = 5

    image_files = os.listdir(folder)

    dataset = np.ndarray(shape=(len(image_files), pixel_channels, image_size, image_size), dtype=np.float32)
    labels = np.ndarray(shape=(len(image_files), num_labels, max_sequence_length), dtype=np.bool_)
    sequences = np.ndarray(shape=(len(image_files), max_sequence_length), dtype=np.bool_)
    bboxes = np.ndarray(shape=(len(image_files), max_sequence_length * bbox_dims), dtype=np.float32)
    
    num_images = 0
    skipped_images = 0
    for image in image_files:
        if image in dictionary:
            label_sequence = dictionary[image]['label']
        else:
            label_sequence = None
            skipped_images += 1
            
        if label_sequence != None and len(label_sequence) > max_sequence_length:
            label_sequence = None
            skipped_images += 1
            print "Image", image, "has too many digits!"
        
        if label_sequence != None and len(label_sequence) > 0:
            image_file = os.path.join(folder, image)
            image_raw = (ndimage.imread(image_file, flatten=True).astype(float) - pixel_depth / 2) / pixel_depth
            
            lefts = dictionary[image]['left']
            tops = dictionary[image]['top']
            widths = dictionary[image]['width']
            heights = dictionary[image]['height']
            rights = []
            bottoms = []
            for left, width in zip(lefts, widths):
                rights.append(left + width)
            for top, height in zip(tops, heights):
                bottoms.append(top + height)
            if min(tops) < max(bottoms) and min(lefts) < max(rights):
                image_data = image_raw[min(tops):max(bottoms), min(lefts):max(rights)]
            else:
                image_data = image_raw
                print "Bounding box error!"
            
            if min(image_data.shape) > 0:
                resized_image = imresize(image_data, (image_size, image_size), interp='bilinear')
                dataset[num_images, 0, :, :] = resized_image

                image_shape = image_data.shape
                warp_height = 1. / image_shape[0] # fractional representation of location on image
                warp_width = 1. / image_shape[1]

                labels[num_images, :, :] = 0
                if len(label_sequence) < max_sequence_length:
                    labels[num_images, 0, len(label_sequence):] = 1 # Blank class labels
                    bboxes[num_images, bbox_dims * len(label_sequence):] = 0 # No bounding box present
                for index in range(len(label_sequence)):
                    labels[num_images, label_sequence[index], index] = 1
                    bboxes[num_images, bbox_dims * index + 0] = 1 # Indicates a bounding box is present
                    bboxes[num_images, bbox_dims * index + 1] = (dictionary[image]['top'][index] - 
                                                                min(tops)) * warp_height
                    bboxes[num_images, bbox_dims * index + 2] = (dictionary[image]['left'][index] - 
                                                                min(lefts)) * warp_width
                    bboxes[num_images, bbox_dims * index + 3] = dictionary[image]['height'][index] * warp_height
                    bboxes[num_images, bbox_dims * index + 4] = dictionary[image]['width'][index] * warp_width

                sequences[num_images, :] = 0
                sequences[num_images, len(label_sequence) - 1] = 1

                num_images += 1
            else:
                skipped_images += 1
                print "Skipped zero-size bbox image:", image
                                    
    print '\nSkipped images:', skipped_images

    dataset = dataset[0:num_images, :, :, :]
    labels = labels[0:num_images, :, :]
    sequences = sequences[0:num_images, :]
    bboxes = bboxes[0:num_images, :]

    return dataset, labels, sequences, bboxes
