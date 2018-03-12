"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/ijba/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/ijba/pairs.txt and the ROC curve
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
from scipy import misc

def main(args):

    pairs, template, im_labels, paths, folds, folds_temp = read_pairs(args.ijba_dir, args.ijba_pairs,args.ijba_nrof_folds)
    np.concatenate([[template.T],[im_labels.T]],axis=0)
    temp_labels = np.unique(np.vstack([template,im_labels]).T,axis=0)

    actual_issame = []
    for p in pairs:
        ind1 = temp_labels[np.where(temp_labels[:,0] == p[0]),1][0][0]
        ind2 = temp_labels[np.where(temp_labels[:,0] == p[1]),1][0][0]
        actual_issame.append(ind1==ind2)

    paths = paths.astype(dtype=object)
    #paths2 = paths.copy()
    #for i in range(len(paths2)):
    #    paths2[i] = os.path.join(args.ijba_dir, 'IJB-A_11_face_images', 'split'+str(folds_temp[i]),paths2[i])

    for i in range(len(paths)):
        paths[i] = os.path.join(args.ijba_dir, 'IJB-A_11_face_images', 'split'+str(folds_temp[i]),paths[i]).replace('.png','.jpg').replace('.jpeg','.jpg').replace('.JPEG','.jpg').replace('.PNG','.jpg').replace('.JPG','.jpg')

    paths_unq, paths_ind, paths_inv = np.unique(paths,True,True)

    #path_prefix = np.array([])
    #for split in range(1, ijba_nrof_folds + 1)
    #    path_prefix = np.vstack([path_prefix, np.matlib.repmat(),2,1)])
    mask = np.ones(len(paths_unq), dtype=bool)
    for i in range(len(paths_unq)):
        if not os.path.isfile(paths_unq[i]):
            mask[i] = False
    paths_unq_masked = paths_unq[mask]

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:

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
            print('Runnning forward pass on ijba images')
            batch_size = args.ijba_batch_size
            nrof_images = len(paths_unq_masked)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths_unq_masked[start_index:end_index]
                images = load_data(paths_batch, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            emb_array_avg = average_temp(emb_array, temp_labels, template, pairs, paths_inv, mask, embedding_size)

            print('ROC curves')
            tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array_avg, actual_issame, nrof_folds=args.ijba_nrof_folds)

            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f' % eer)


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
def load_data(image_paths, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        images[i,:,:,:] = img
    return images

def average_temp(emb_array, temp_labels, template, pairs, paths_inv, mask, embedding_size):
    mask_inv = np.ones_like(mask) * -1
    mask_inv[mask] = range(0, sum(mask == True))
    template_mean = np.zeros([embedding_size, max(temp_labels[:, 0]) + 1])
    ts= []
    for t in range(0, len(temp_labels)):
        img_ind = template == temp_labels[t, 0]
        t_ind = mask_inv[paths_inv[img_ind]]
        t_emb = emb_array[t_ind[t_ind != -1]]
        if len(t_emb) != 0:
            template_mean[:, temp_labels[t, 0]] = np.mean(t_emb, 0)
        else:
            ts.append(t)
    # this is to copy not found images
    # paths2 is before replacing the extensions
    #for t_find in temp_labels[ts, 0]:
    #    for i in np.where(template == t_find)[0]:
    #        img = misc.imread(paths2[i].replace(args.ijba_dir,'D:\data\IJB\IJB-A'))
    #        img = misc.imresize(img,[108,108])
    #        misc.imsave(paths[i],img[12:,6:-6])

    embeddings = template_mean[:, pairs]
    return np.swapaxes(embeddings, 0, 1)


def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[:,:,0]
    embeddings2 = embeddings[:,:,1]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
                                               np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.concatenate([np.arange(0, 0.1, 0.001),np.arange(0.1, 4, 0.1)])
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
                                              np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def get_paths(ijba_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(ijba_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(ijba_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(ijba_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(ijba_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def read_pairs(ijba_dir,ijba_pairs,ijba_nrof_folds):
    pairs =[]
    template = np.array([],dtype=int)
    labels = np.array([],dtype=int)
    paths = np.array([],dtype=str)
    folds = np.array([],dtype=int)
    folds_temp = np.array([],dtype=int)
    for split in range(1,ijba_nrof_folds+1):
        comp_path = os.path.join(ijba_dir, ijba_pairs,'split'+str(split),'verify_comparisons_'+str(split)+'.csv')
        meta_path = os.path.join(ijba_dir, ijba_pairs, 'split' + str(split), 'verify_metadata_' + str(split) + '.csv')
        comp = np.genfromtxt(comp_path, dtype=int, defaultfmt='%d %d', delimiter=",")
        pairs.extend(comp.astype(dtype=int))
        meta = np.genfromtxt(meta_path, dtype=str, defaultfmt='%d %d %s', delimiter=",",skip_header=True)
        template = np.append(template,meta[:, [0]].astype(dtype=int))
        labels = np.append(labels, meta[:, [1]].astype(dtype=int))
        paths = np.append(paths, meta[:, [2]].astype(dtype=str))
        folds = np.append(folds, (np.zeros(len(comp),dtype=int)+split).tolist())
        folds_temp = np.append(folds_temp, (np.zeros(len(meta[:, [0]]),dtype=int)+split).tolist())
    return np.array(pairs),template,labels,paths,folds,folds_temp

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('ijba_dir', type=str,
                        help='Path to the data directory containing aligned ijba face patches.')
    parser.add_argument('--ijba_batch_size', type=int,
                        help='Number of images to process in a batch in the ijba test set.', default=100)
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--ijba_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='IJB-A_11_sets')
    parser.add_argument('--ijba_file_ext', type=str,
                        help='The file extension for the ijba dataset.', default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--ijba_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
