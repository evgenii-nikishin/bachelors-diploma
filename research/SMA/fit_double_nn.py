from __future__ import division, print_function
from svd_improvement import *
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
seed = 417


def create_architecture(n_users, n_items, n_latent, n_hidden, learning_rate=0.01):
    reg_users, reg_items = 0.06, 0.06
    coef_users, coef_items = 0.005 / n_items, 0.005 / n_users

    X_users = tf.placeholder("float", [None, n_items])
    mask_users = tf.placeholder("float", [None, n_items])  # bool actually
    X_items = tf.placeholder("float", [None, n_users])
    mask_items = tf.placeholder("float", [None, n_users])  # bool actually
    ratings = tf.placeholder("float", [None, 1])

    weights = {
        'users_encoder_last': tf.Variable(tf.random_normal([n_items, n_latent])),
        'users_decoder_last': tf.Variable(tf.random_normal([n_latent, n_items])),
        'items_encoder_last': tf.Variable(tf.random_normal([n_users, n_latent])),
        'items_decoder_last': tf.Variable(tf.random_normal([n_latent, n_users])),
        'ratings_layer_1': tf.Variable(tf.random_normal([2 * n_latent, n_hidden])),
        'ratings_layer_last': tf.Variable(tf.random_normal([n_hidden, 1]))
    }
    biases = {
        'users_encoder_last': tf.Variable(tf.random_normal([n_latent])),
        'users_decoder_last': tf.Variable(tf.random_normal([n_items])),
        'items_encoder_last': tf.Variable(tf.random_normal([n_latent])),
        'items_decoder_last': tf.Variable(tf.random_normal([n_users])),
        'ratings_layer_1': tf.Variable(tf.random_normal([n_hidden])),
        'ratings_layer_last': tf.Variable(tf.random_normal([1]))
    }

    users_encoder_last = tf.sigmoid(tf.add(tf.matmul(X_users, weights['users_encoder_last']),
                                           biases['users_encoder_last']))
    users_decoder_last = tf.sigmoid(tf.add(tf.matmul(users_encoder_last, weights['users_decoder_last']),
                                           biases['users_decoder_last']))
    items_encoder_last = tf.sigmoid(tf.add(tf.matmul(X_items, weights['items_encoder_last']),
                                           biases['items_encoder_last']))
    items_decoder_last = tf.sigmoid(tf.add(tf.matmul(items_encoder_last, weights['items_decoder_last']),
                                           biases['items_decoder_last']))

    ratings_layer_1 = tf.sigmoid(tf.add(tf.matmul(tf.concat([users_encoder_last, items_encoder_last], axis=1),
                                                  weights['ratings_layer_1']),
                                        biases['ratings_layer_1']))
    ratings_layer_last = tf.sigmoid(tf.add(tf.matmul(ratings_layer_1, weights['ratings_layer_last']),
                                           biases['ratings_layer_last']))

    cost_users = tf.div(tf.reduce_sum(tf.pow(tf.multiply(X_users - users_decoder_last, mask_users), 2)),
                        tf.reduce_sum(mask_users))
    users_weights = map(lambda x: x[1], filter(lambda x: x[0].startswith('users'), weights.items()))
    cost_users += reg_users * tf.reduce_sum(map(lambda x: tf.reduce_sum(tf.pow(x, 2)), users_weights))

    cost_items = tf.div(tf.reduce_sum(tf.pow(tf.multiply(X_items - items_decoder_last, mask_items), 2)),
                        tf.reduce_sum(mask_items))
    items_weights = map(lambda x: x[1], filter(lambda x: x[0].startswith('items'), weights.items()))
    cost_items += reg_items * tf.reduce_sum(map(lambda x: tf.reduce_sum(tf.pow(x, 2)), items_weights))

    cost_users *= coef_users
    cost_items *= coef_items
    cost_ratings = tf.reduce_mean(tf.pow(ratings - ratings_layer_last, 2))
    cost = cost_ratings + cost_users + cost_items

    tf.summary.scalar('users_autoencoder_loss', cost_users)
    tf.summary.scalar('items_autoencoder_loss', cost_items)
    tf.summary.scalar('ratings_net_loss', cost_ratings)
    tf.summary.scalar('total_loss', cost)
    merged_summary = tf.summary.merge_all()

    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    placeholders = (X_users, mask_users, X_items, mask_items, ratings)
    return placeholders, cost, optimizer, ratings_layer_last, merged_summary


def fit_nn(R, train_mask, n_latent, n_hidded):
    n_users, n_items = R.shape
    learning_rate = 0.001
    n_epoch = 25
    batch_size = 64

    placeholders, cost, optimizer, ratings_layer_last, merged_summary = \
        create_architecture(n_users, n_items, n_latent, n_hidded, learning_rate)
    X_users, mask_users, X_items, mask_items, ratings = placeholders

    init = tf.global_variables_initializer()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter('./tf_log/train',
                                         sess.graph)
    # test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    try:
        sess.run(init)

        for epoch in range(n_epoch):
            nnz_indices = zip(*np.where(R))
            np.random.shuffle(nnz_indices)
            batch_count = int(math.ceil(len(nnz_indices) / batch_size))
            nnz_indices = map(np.array, zip(*nnz_indices))

            for i in range(int(np.sqrt(batch_count))):  # range(batch_count):
                u_batch = R.values[nnz_indices[0][i * batch_size: (i + 1) * batch_size], :]
                i_batch = R.values[:, nnz_indices[1][i * batch_size: (i + 1) * batch_size]].T
                r_batch = R.values[nnz_indices[0][i * batch_size: (i + 1) * batch_size],
                                   nnz_indices[1][i * batch_size: (i + 1) * batch_size]][:, np.newaxis]

                u_mask_batch = train_mask[nnz_indices[0][i * batch_size: (i + 1) * batch_size], :]
                i_mask_batch = train_mask[:, nnz_indices[1][i * batch_size: (i + 1) * batch_size]].T

                _, c, summary = sess.run([optimizer, cost, merged_summary], feed_dict={X_users: u_batch,
                                                                                       mask_users: u_mask_batch,
                                                                                       X_items: i_batch,
                                                                                       mask_items: i_mask_batch,
                                                                                       ratings: r_batch})
                train_writer.add_summary(summary, int(np.sqrt(batch_count)) * epoch + i)

            print('Epoch: {:03d}, cost = {:.5f}'.format(epoch + 1, c))

        print("Optimization finished")
    except KeyboardInterrupt:
        print("Optimization was interrupted")

    return sess, placeholders, ratings_layer_last

if __name__ == "__main__":
    ratings, users, movies = load_data()
    # full matrix is relatively small, so it can be fitted in RAM
    R = construct_full_matrix(ratings, users, movies, mode='dense')
    train_mask, test_mask = get_train_test_masks(ratings, users, movies)

    sess, placeholders, ratings_layer_last = fit_nn(R, train_mask, 128, 32)
    sess.close()