import tensorflow as tf
import numpy as np
import os
import cPickle
import copy
from sklearn.metrics import roc_curve, auc, average_precision_score
from defs import *

def initializer(init, shape):
    if init == "zero":
        return tf.zeros(shape)
    elif init == "he":
        fan_in = np.prod(shape[0:-1])
        std = 1/np.sqrt(fan_in)
        return tf.random_uniform(shape, minval=-std, maxval=std)


def nonlinearity(nl):
    if nl == "relu":
        return tf.nn.relu
    elif nl == "tanh":
        return tf.nn.tanh
    elif nl == "linear" or nl == "none":
        return lambda x: x


def node_average_model(input, params, filters=None, dropout_keep_prob=1.0, trainable=True):
    vertices, edges, nh_indices = input
    nh_indices = tf.squeeze(nh_indices, axis=2)
    v_shape = vertices.get_shape()
    nh_sizes = tf.expand_dims(tf.count_nonzero(nh_indices + 1, axis=1, dtype=tf.float32), -1)  # for fixed number of neighbors, -1 is a pad value
    if params is None:
        # create new weights
        Wc = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wc", trainable=trainable)  # (v_dims, filters)
        Wn = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wn", trainable=trainable)  # (v_dims, filters)
        b = tf.Variable(initializer("zero", (filters,)), name="b", trainable=trainable)
    else:
        Wn, Wc = params["Wn"], params["Wc"]
        filters = Wc.get_shape()[-1].value
        b = params["b"]
    params = {"Wn": Wn, "Wc": Wc, "b": b}
    # generate vertex signals
    Zc = tf.matmul(vertices, Wc, name="Zc")  # (n_verts, filters)
    # create neighbor signals
    v_Wn = tf.matmul(vertices, Wn, name="v_Wn")  # (n_verts, filters)
    Zn = tf.divide(tf.reduce_sum(tf.gather(v_Wn, nh_indices), 1), tf.maximum(nh_sizes, tf.ones_like(nh_sizes)))  # (n_verts, v_filters)
    nonlin = nonlinearity("relu")
    sig = Zn + Zc + b
    h = tf.reshape(nonlin(sig), tf.constant([-1, filters]))
    h = tf.nn.dropout(h, dropout_keep_prob)
    return h, params


def dense(input, out_dims=None, dropout_keep_prob=1.0, nonlin=True, trainable=True):
    input = tf.nn.dropout(input, dropout_keep_prob)
    in_dims = input.get_shape()[-1].value
    out_dims = in_dims if out_dims is None else out_dims
    W = tf.Variable(initializer("he", [in_dims, out_dims]), name="w", trainable=trainable)
    b = tf.Variable(initializer("zero", [out_dims]), name="b", trainable=trainable)
    Z = tf.matmul(input, W) + b
    if(nonlin):
        nonlin = nonlinearity("relu")
        Z = nonlin(Z)
    Z = tf.nn.dropout(Z, dropout_keep_prob)
    return Z


def merge(input):
    input1, input2, examples = input
    out1 = tf.gather(input1, examples[:, 0])
    out2 = tf.gather(input2, examples[:, 1])
    output1 = tf.concat([out1, out2], axis=0)
    output2 = tf.concat([out2, out1], axis=0)
    return tf.concat((output1, output2), axis=1)

def average_predictions(input):
    combined = tf.reduce_mean(tf.stack(tf.split(input, 2)), 0)
    return combined

def build_feed_dict(model_variables_list, minibatch):
   in_vertex1, in_edge1, in_hood_indices1, in_vertex2, in_edge2, in_hood_indices2, examples, preds,labels, dropout_keep_prob = model_variables_list
   feed_dict = {
                    in_vertex1: minibatch["l_vertex"], in_edge1: minibatch["l_edge"],
                    in_vertex2: minibatch["r_vertex"], in_edge2: minibatch["r_edge"],
                    in_hood_indices1: minibatch["l_hood_indices"],
                    in_hood_indices2: minibatch["r_hood_indices"],
                    examples: minibatch["label"][:, :2],
                    labels: minibatch["label"][:, 2],
                    dropout_keep_prob: dropout_keep
   }
   return feed_dict

def build_graph_conv_model(in_nv_dims, in_ne_dims, in_nhood_size): 

   in_vertex1 = tf.placeholder(tf.float32,[None,in_nv_dims],"vertex1")
   in_vertex2 = tf.placeholder(tf.float32,[None,in_nv_dims],"vertex2")
   in_edge1 = tf.placeholder(tf.float32,[None,in_nhood_size,in_ne_dims],"edge1")
   in_edge2 = tf.placeholder(tf.float32,[None,in_nhood_size,in_ne_dims],"edge2")
   in_hood_indices1 = tf.placeholder(tf.int32,[None,in_nhood_size,1],"hood_indices1")
   in_hood_indices2 = tf.placeholder(tf.int32,[None,in_nhood_size,1],"hood_indices2")

   input1 = in_vertex1, in_edge1, in_hood_indices1
   input2 = in_vertex2, in_edge2, in_hood_indices2

   examples = tf.placeholder(tf.int32,[None,2],"examples")
   labels = tf.placeholder(tf.float32,[None],"labels")
   dropout_keep_prob = tf.placeholder(tf.float32,shape=[],name="dropout_keep_prob")

   #layer 1
   layer_no = 1
   name = "left_branch_{}_{}".format("node_average", layer_no)
   with tf.name_scope(name):
	output, params = node_average_model(input1, None, filters=256, dropout_keep_prob=0.5)
	input1 = output, in_edge1, in_hood_indices1

   name = "right_branch_{}_{}".format("node_average", layer_no)
   with tf.name_scope(name):
	output, _ = node_average_model(input2, params, filters=256, dropout_keep_prob=0.5)
	input2 = output, in_edge2, in_hood_indices2

   #layer 2
   layer_no = 2
   name = "left_branch_{}_{}".format("node_average", layer_no)
   with tf.name_scope(name):
	output, params = node_average_model(input1, None, filters=256, dropout_keep_prob=0.5)
	input1 = output, in_edge1, in_hood_indices1

   name = "right_branch_{}_{}".format("node_average", layer_no)
   with tf.name_scope(name):
	output, _ = node_average_model(input2, params, filters=256, dropout_keep_prob=0.5)
	input2 = output, in_edge2, in_hood_indices2

   # merged layers
   layer_no = 3
   name = "{}_{}".format("merge", layer_no)
   input = input1[0], input2[0], examples
   with tf.name_scope(name):
	input = merge(input)

   # dense layer
   layer_no = 4
   name = "{}_{}".format("dense", layer_no)
   with tf.name_scope(name):
	input = dense(input, out_dims=512, dropout_keep_prob=0.5, nonlin=True, trainable=True)

   # dense layer
   layer_no = 5
   name = "{}_{}".format("dense", layer_no)
   with tf.name_scope(name):
	input = dense(input, out_dims=1, dropout_keep_prob=0.5, nonlin=False, trainable=True)

   # average layer
   layer_no = 6
   name = "{}_{}".format("average_predictions", layer_no)
   with tf.name_scope(name):
       preds = average_predictions(input)

   return [in_vertex1, in_edge1, in_hood_indices1, in_vertex2, in_edge2, in_hood_indices2, examples, preds,labels, dropout_keep_prob]



def loss_op(preds, labels):
   # Loss and optimizer
   with tf.name_scope("loss"):
     scale_vector = (pn_ratio * (labels - 1) / -2) + ((labels + 1) / 2)
     logits = tf.concat([-preds, preds], axis=1)
     labels_stacked = tf.stack([(labels - 1) / -2, (labels + 1) / 2], axis=1)
     loss = tf.losses.softmax_cross_entropy(labels_stacked, logits, weights=scale_vector)
     return loss


