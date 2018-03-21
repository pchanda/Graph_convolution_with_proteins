import tensorflow as tf
import numpy as np
import os
import cPickle
import copy
from sklearn.metrics import roc_curve, auc, average_precision_score
from defs import *
from graph_conv import *

if __name__=='__main__':

  #load the training data
  train_data_file = os.path.join('./data/','train.cpkl')
  train_list, train_data = cPickle.load(open(train_data_file))

  in_nv_dims = train_data[0]["l_vertex"].shape[-1]
  in_ne_dims = train_data[0]["l_edge"].shape[-1]
  in_nhood_size = train_data[0]["l_hood_indices"].shape[1]
 
  model_variables_list = build_graph_conv_model(in_nv_dims, in_ne_dims, in_nhood_size)
  in_vertex1, in_edge1, in_hood_indices1, in_vertex2, in_edge2, in_hood_indices2, examples, preds,labels, dropout_keep_prob = model_variables_list

  loss = loss_op(preds, labels)
  #add train op
  with tf.name_scope("optimizer"):
      # generate an op which trains the model
      train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
   
  saver = tf.train.Saver(max_to_keep=250)

  with tf.Session() as sess:
     # set up tensorflow session
     #sess.run(tf.global_variables_initializer())
     sess.run(tf.initialize_all_variables())
     print("Training Model")

     for epoch in range(0, num_epochs):
       """
       Trains model for one pass through training data, one protein at a time
       Each protein is split into minibatches of paired examples.
       Features for the entire protein is passed to model, but only a minibatch of examples are passed
       """
       prot_perm = np.random.permutation(len(train_data))
       ii = 0
       nn = 0
       avg_loss = 0
       # loop through each protein
       for protein in prot_perm:
          # extract just data for this protein
          prot_data = train_data[protein]
          pair_examples = prot_data["label"]
          n  = len(pair_examples)
          shuffle_indices = np.random.permutation(np.arange(n)).astype(int)
          # loop through each minibatch
          for i in range(int(n / minibatch_size)):
             # extract data for this minibatch
             index = int(i * minibatch_size)
             example_pairs = pair_examples[shuffle_indices[index: index + minibatch_size]]
             minibatch = {}
             for feature_type in prot_data:
                 if feature_type == "label":
                     minibatch["label"] = example_pairs
                 else:
                     minibatch[feature_type] = prot_data[feature_type]
             # train the model
             feed_dict = build_feed_dict(model_variables_list, minibatch)
             _,loss_v = sess.run([train_op,loss], feed_dict=feed_dict)
             #print("Epoch =",epoch," iter = ",ii," loss = ",loss_v)
             avg_loss += loss_v
             ii += 1
          nn += n    
          #print("Epoch =",epoch," iter = ",ii," loss = ",loss_v)
       print("Epoch_end =",epoch,", avg_loss = ",avg_loss/ii," nn = ",nn)
       ckptfile = saver.save(sess, './saved_models/model_%d.ckpt'%(epoch))

     all_preds = []
     all_labels = []
     all_losses = []
     for prot_data in train_data:
       temp_data = copy.deepcopy(prot_data)
       n = prot_data['label'].shape[0] #no of labels for this protein molecule.
       #split the labels into chunks of minibatch_size.
       batch_split_points = np.arange(0,n,minibatch_size)[1:]
       batches = np.array_split(prot_data['label'],batch_split_points)
       for a_batch in batches:
          temp_data['label'] = a_batch     
          feed_dict = build_feed_dict(model_variables_list, temp_data)
          res = sess.run([loss,preds,labels], feed_dict=feed_dict)
          pred_v = np.squeeze(res[1])
          if len(pred_v.shape)==0:
             pred_v = [pred_v]
             all_preds += pred_v
          else:
             pred_v = pred_v.tolist()
             all_preds += pred_v
          all_labels += res[2].tolist()
          all_losses += [res[0]]

     fpr, tpr, _ = roc_curve(all_labels, all_preds)
     roc_auc = auc(fpr, tpr)
     print('mean loss = ',np.mean(all_losses))
     print('roc_auc = ',roc_auc)

