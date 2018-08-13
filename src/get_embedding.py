"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/data/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/data/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from glob import glob
from sklearn.manifold import TSNE
import matplotlib.pyplot as pt
import matplotlib.colors
import scipy.io

def main(args):
  
    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            
            # Get the paths for the corresponding images
            paths = sorted(glob("{}/*/*.{}".format(os.path.expanduser(args.data_dir), args.data_file_ext)))

            # Load the model
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            #image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]
        
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on data images')
            batch_size = args.data_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            print(nrof_batches)
            for i in range(nrof_batches):
                print(i)
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)

                feed_dict = { images_placeholder:images[:,6:102,6:102,:], phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            scipy.io.savemat('embeddings2.mat',{'emb_array':emb_array,'paths':paths})
            #tsne = TSNE(n_components=2).fit_transform(emb_array)
            #pt.scatter(tsne[1:,0],tsne[1:,1],c=rotations[1:],cmap=pt.get_cmap('Set1'))

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned data face patches.')
    parser.add_argument('--data_batch_size', type=int,
        help='Number of images to process in a batch in the data test set.', default=100)
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=108)
    parser.add_argument('--data_file_ext', type=str,
        help='The file extension for the data dataset.', default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--data_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
