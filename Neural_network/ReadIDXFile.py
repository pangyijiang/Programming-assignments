# -*- coding: UTF-8 -*-

# encoding: utf-8
"""
THE IDX FILE FORMAT

the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
The basic format is

magic number
size in dimension 0
size in dimension 1
size in dimension 2
.....
size in dimension N
data

The magic number is an integer (MSB first). The first 2 bytes are always 0.

The third byte codes the type of the data:
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....

The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.
"""

import numpy as np
import struct
import matplotlib.pyplot as plt

"""
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
"""
def ReturnIdx3ImageList(idx3image_file = 'train-images.idx3-ubyte', requestImageNum = 0):
    """
    :param idx3image_file: idx3 file path
    :return: image data
    """
    # load binary data
    tempFile = open(idx3image_file, 'rb')
    bin_data = tempFile.read()

    # file header information
    offset = 0
    fmt_header = '>iiii'    #struct.unpack_from read 4 int
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print("IDX image file: %s Info." % idx3image_file)
    print("magic_number = %d, num_images = %d, image_rows = %d, image_cols = %d" % (magic_number, num_images, num_rows, num_cols))
    #file image information
    if( (requestImageNum > 0) and (requestImageNum <= num_images) ):
        num_images = requestImageNum
    print("requestImageNum = %d" % num_images)

    # file image data
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B' #struct.unpack_from read spesific num of byte
    images = []
    for i in range(num_images):
        images.append(list(struct.unpack_from(fmt_image, bin_data, offset)))
        offset += struct.calcsize(fmt_image)

    tempFile.close()
    return images

"""
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
"""
def ReturnIdx3ImageListLabels(idx3imagelabels_file = 'train-labels.idx1-ubyte', requestImageNum = 0):
    """
    :param idx3imagelabels_file: idx1 file path
    :return: labels data
    """
    #load binary data
    tempFile = open(idx3imagelabels_file, 'rb')
    bin_data = tempFile.read()

    # file header information
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print("IDX labels file: %s Info." % idx3imagelabels_file)
    print("magic_number = %s, num_images = %s" % (magic_number, num_images))
    #num of image
    if( (requestImageNum > 0) and (requestImageNum <= num_images) ):
        num_images = requestImageNum
    print("requestImageNum = %d" % num_images)

    # file label data information
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    #labels = np.empty(num_images)
    labels = []
    for i in range(num_images):
        #labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        labels.append(list(struct.unpack_from(fmt_image, bin_data, offset)))
        offset += struct.calcsize(fmt_image)

    tempFile.close()
    return labels


if __name__ == '__main__':
    
    test_images = ReturnIdx3ImageList(requestImageNum = 3)
    test_labels = ReturnIdx3ImageListLabels(requestImageNum = 3)

    test_images = np.array(test_images)
    for i in range(20):
        plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
        plt.title("train_labels= %s" % test_labels[i])
        plt.show()
    

    