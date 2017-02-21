def digitStruct_name(sample, hdf5_data):
    name_obj = hdf5_data['digitStruct']['name']
    name = ''.join([chr(value[0]) for value in hdf5_data[name_obj[sample][0]].value])
    return name

def digitStruct_bbox(sample, hdf5_data):
    def get_values(member, obj):
        value = []
        if obj.shape[0] > 1:
            for character in range(obj.shape[0]):
                value.append(int(hdf5_data[obj[character][0]][0][0]))
        else:
            value.append(int(obj[0][0]))
        box_info[member] = value

    box_info = {}
    box_info['label'] = []
    box_info['top'] = []
    box_info['left'] = []
    box_info['height'] = []
    box_info['width'] = []

    bbox = hdf5_data['digitStruct']['bbox'][sample]
    hdf5_data[bbox[0]].visititems(get_values)
    return box_info

def createDictionary(struct):
    dictionary = {}
    len_samples = struct['digitStruct']['name'].size
    for sample in range(len_samples):
        image_name = digitStruct_name(sample, struct)
        image_bbox = digitStruct_bbox(sample, struct)
        dictionary[image_name] = image_bbox
    return dictionary, len_samples
