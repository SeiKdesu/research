#! /usr/bin/env python

import numpy as np
np.random.seed(123456789)

import os
import time
import datetime
import pickle as cPickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from util import *
from liblinearutil import *
import statistics

# Parameters
# ==================================================

parser = ArgumentParser("u2GAN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../runs_u2GAN_noPOS_REDDIT", help="")
parser.add_argument("--dataset", default="REDDITMULTI5K", help="Name of the dataset.")
parser.add_argument("--embedding_dim", default=4, type=int, help="Dimensionality of character embedding")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=8, type=int, help="Batch Size")
parser.add_argument("--idx_time", default=1, type=int, help="")
parser.add_argument("--num_epochs", default=10, type=int, help="Number of training epochs")
parser.add_argument("--saveStep", default=1, type=int, help="")
parser.add_argument("--allow_soft_placement", default=True, type=bool, help="Allow device soft device placement")
parser.add_argument("--log_device_placement", default=False, type=bool, help="Log placement of ops on devices")
parser.add_argument("--model_name", default='REDDITMULTI5K', help="")
parser.add_argument('--num_sampled', default=32, type=int, help='')
parser.add_argument("--dropout_keep_prob", default=0.5, type=float, help="Dropout keep probability")
parser.add_argument("--num_hidden_layers", default=4, type=int, help="Number of attention layers")
parser.add_argument("--num_heads", default=1, type=int, help="Number of attention heads within each attention layer")
parser.add_argument("--ff_hidden_size", default=1024, type=int, help="The hidden size for the feedforward layer")
parser.add_argument("--num_neighbors", default=4, type=int, help="")
parser.add_argument('--degree_as_tag', action="store_false", help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
parser.add_argument('--fold_idx', type=int, default=0, help='the index of fold in 10-fold validation. 0-9.')
parser.add_argument("--iterations", default=8, type=int, help="")
parser.add_argument('--split_idx', type=int, default=0, help='Split data into 10 parts, do U2GAN for each part. It is for large datasets like Reddit')
parser.add_argument("--tmpString", default="REDDITMULTI5K", help="Name of the dataset.")
args = parser.parse_args()
print(args)

# Load data
print("Loading data...")

large_graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
####split data into 10 parts
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
labels = [graph.label for graph in large_graphs]
idx_list = []
for idx in skf.split(np.zeros(len(labels)), labels):
    idx_list.append(idx[1])
graph_labels = []
for j in range(10):
    graphs = [large_graphs[i] for i in idx_list[j]]
    graph_labels += [graph.label for graph in graphs]
graph_labels = np.array(graph_labels)
# print(graph_labels)

lstfiles = []
for root, dirs, files in os.walk(args.run_folder):
    for file in files:
        embeddings_file = os.path.join(root, file)
        if args.dataset not in str(embeddings_file):
            continue
        if args.tmpString not in str(embeddings_file):
            continue
        lstfiles.append(embeddings_file)
        print(embeddings_file)

lstfiles = sorted(lstfiles, key=str.lower)
dict_files = {}
for file in lstfiles:
    # print(file)
    if file.split('/')[-3].split('_split')[-1][2:] not in dict_files:
        dict_files[file.split('/')[-3].split('_split')[-1][2:]] = []
    dict_files[file.split('/')[-3].split('_split')[-1][2:]].append(file)

dict_model = {}
for key in dict_files:
    if key not in dict_model:
        dict_model[key] = []
    for tmp_idx_str in range(1, 51):
        lst_files = []
        for tmp_key in dict_files[key]:
            if 'model-'+str(tmp_idx_str) == tmp_key.split('/')[-1]:
                print(key, tmp_key)
                lst_files.append(tmp_key)
        dict_model[key].append(lst_files)
    print('-----------------------')

for key in dict_model:
    write_acc = open(args.dataset + '_' + str(key) + '_acc.txt', 'w')
    for tmp in dict_model[key]:
        embedding_matrix = []
        for tmp_file in tmp:
            print(key, tmp_file)
            with open(tmp_file, 'rb') as f:
                features_matrix = cPickle.load(f)
                embedding_matrix.append(features_matrix)
        graph_embeddings = np.concatenate(embedding_matrix, 0)
        graph_labels = graph_labels[:graph_embeddings.shape[0]]
        print(graph_embeddings.shape)
        newidx_list = []
        for idx in skf.split(np.zeros(len(graph_labels)), graph_labels):
            newidx_list.append(idx)

        acc_10folds = []
        for fold_idx in range(10):
            train_idx, test_idx = newidx_list[fold_idx]
            train_graph_embeddings = graph_embeddings[train_idx]
            test_graph_embeddings = graph_embeddings[test_idx]
            train_labels = graph_labels[train_idx]
            test_labels = graph_labels[test_idx]

            prob = problem(train_labels, train_graph_embeddings)
            param = parameter('-s 0 -e 0.001')
            m = train(prob, param)

            p_label, p_acc, p_val = predict(test_labels, test_graph_embeddings, m)
            ACC, MSE, SCC = evaluations(test_labels, p_label)
            acc_10folds.append(ACC)

            print(key, 'epoch ', tmp[0].split('-')[-1], ' fold ', fold_idx, ' acc ', ACC)

        mean_10folds = statistics.mean(acc_10folds)
        std_10folds = statistics.stdev(acc_10folds)
        print(key, 'epoch ', tmp[0].split('-')[-1], ' mean: ', str(mean_10folds), ' std: ', str(std_10folds))

        write_acc.write('epoch ' + tmp[0].split('-')[-1] + ' mean: ' + str(mean_10folds) + ' std: ' + str(std_10folds) + '\n')

    write_acc.close()
