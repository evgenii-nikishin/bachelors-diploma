from __future__ import print_function, division
import math
import numpy as np
import zipfile
import pandas as pd
from scipy.linalg import svd, diagsvd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import tensorflow as tf


# loading data routine

def read_data(archive, filename):
    data = archive.open('ml-1m/' + filename + '.dat').readlines()
    return np.array(map(lambda x: x[:-1].split('::'), data))


def preprocess_ratings(ratings_raw):
    ratings = dict()
    for user in np.unique(ratings_raw[:, 0]):
        ratings[user] = ratings_raw[np.where(ratings_raw[:, 0] == user)[0]][:, 1:]
    return ratings


def preprocess_users(users_raw):
    users = dict()
    
    users_int = users_raw[:, :-1].copy()
    users_int[:, 1] = (users_raw[:, 1] == 'M').astype(int)
    users_int = users_int.astype(int)
    
    for user in users_int:
        users[user[0]] = user[1:]
    
    return users


def preprocess_movies(movies_raw):
    movies = dict()
    
    for movie in movies_raw:
        movies[int(movie[0])] = movie[1:]
        movies[int(movie[0])].extend([0, 0]) # summary ratings, summary views
        
    return movies


def load_data():
    archive = zipfile.ZipFile('ml-1m.zip', 'r')
    
    ratings_raw = read_data(archive,'ratings').astype(int)
    users_raw = read_data(archive, 'users')
    movies_raw = map(list, read_data(archive, 'movies'))
    
    ratings = preprocess_ratings(ratings_raw)
    users = preprocess_users(users_raw)
    movies = preprocess_movies(movies_raw)
    
    return ratings, users, movies


def get_train_test_masks(ratings, users, movies, train_frac=0.9):
    train_mask = pd.DataFrame(data=0, index=users.keys(), columns=movies.keys(), dtype=bool)
    test_mask = pd.DataFrame(data=0, index=users.keys(), columns=movies.keys(), dtype=bool)
    
    for user, itemList in ratings.items():
        # itemList = [(i, r, t), ...]
        interactions = sorted(itemList, key=lambda x: x[2])
        thr = int(math.floor(len(interactions) * train_frac))
        
        train_mask.loc[user, map(lambda x: x[0], interactions[:thr])] = True
        test_mask.loc[user, map(lambda x: x[0], interactions[thr:])] = True
        
    return train_mask.values, test_mask.values


def construct_full_matrix(ratings, users, movies, mode='dense'):
    if mode == 'dense':
        R = pd.DataFrame(data=0, index=users.keys(), columns=movies.keys(), dtype='float32')
        for user in R.index:
            R.loc[user, ratings[user][:, 0]] = ratings[user][:, 1] / 5
    elif mode == 'sparse':
        pass
    
    return R


# auxiliary functions

def svd_2_factors(A, k):
    U, s, Vh = svds(csr_matrix(A), k=k)
    Sigma = np.zeros((U.shape[1], Vh.shape[0]))
    Sigma[range(len(s)), range(len(s))] = s
    V = np.dot(Vh.T, Sigma.T)
    return U, V


def rmse(A, A_appr, mask=None):
    if mask is None:
        return np.sqrt(np.mean(np.power(A - A_appr, 2).mean()))
    else:
        if isinstance(A, pd.DataFrame):
            return np.sqrt(np.mean(np.power(A.values[mask] - A_appr[mask], 2)))
        else:
            return np.sqrt(np.mean(np.power(A[mask] - A_appr[mask], 2)))
        

        
def divide_set(S, K):
    # returns K+1 subsets: s_0 = empty set, s_1 + ... + s_K = S
    affiliation_vector = np.random.randint(K, size=np.sum(S))
    indices = np.where(S)
    subsets = np.zeros((K+1, ) + S.shape, dtype=bool)

    for k in range(1, K+1):
        current_subset = affiliation_vector == k
        subsets[k, indices[0][current_subset], indices[1][current_subset]] = True
    
    return subsets


def plot_with_outliers(matrix, vmax=5.01):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, vmax=vmax, interpolation='none', cmap=plt.cm.hot)
    plt.colorbar(extend='max').cmap.set_over('blue')
    
    
# direct algorithm

def compute_hard_pred_subsets(R, train_mask, R_appr, num_subsets, choice_prob=0.7):
    av_error = rmse(R, R_appr, mask=train_mask)
    rho_mask = np.logical_and(np.random.rand(*R.shape) <= choice_prob, train_mask)
    diff_mask = np.logical_and(np.abs(R.values - R_appr) <= av_error, train_mask)
    easy_pred_indices = np.logical_and(np.logical_not(np.logical_xor(rho_mask, diff_mask)), train_mask)
    hard_pred_indices = np.logical_xor(train_mask[np.newaxis, :, :], divide_set(easy_pred_indices, num_subsets))
    return hard_pred_indices


def minimize_final_objective(R, U_0, V_0, hard_pred_indices, lambdas, lr=1.0, reg_U=0.06, reg_V=0.06, max_iter=15):
    U_old, V_old = U_0, V_0
    
    for iter_num in range(max_iter):
        U, V = U_old.copy(), V_old.copy()
        diff = np.dot(U, V.T) - R
        
        for k in range(lambdas.shape[0]):  # K+1
            approx_error_masked = np.sum(np.power(diff * hard_pred_indices[k], 2)) / np.sum(hard_pred_indices[k])
            shared_factor = lr * lambdas[k] / (approx_error_masked * np.sum(hard_pred_indices[k]))
            U -= shared_factor * np.dot(diff * hard_pred_indices[k], V_old)
            V -= shared_factor * np.dot((diff * hard_pred_indices[k]).T, U_old)
        
        U -= lr * reg_U * U
        V -= lr * reg_V * V
        U_old, V_old = U, V
        
    return U, V


def improve_approximation(R, train_mask, U_old, V_old, choice_prob, rank, lambdas):
    R_appr = np.dot(U_old, V_old.T)
    hard_pred_indices = compute_hard_pred_subsets(R, train_mask, R_appr, lambdas.shape[0]-1)
    return minimize_final_objective(R, U_old, V_old, hard_pred_indices, lambdas)


# autoencoder approach

def create_architecture(n_input, n_hidden_1, num_subsets, learning_rate=0.01):
    reg_U, reg_V = 0.06, 0.06
    lambdas = np.ones(num_subsets+1)
    
    X = tf.placeholder("float", [None, n_input])
    mask = tf.placeholder("float", [num_subsets+1, None, n_input])  # bool actually
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b1': tf.Variable(tf.random_normal([n_input])),
    }
    
    encoder_layer_1 = tf.add(tf.matmul(X, weights['encoder_h1']), 
                             biases['encoder_b1'])
    decoder_layer_1 = tf.add(tf.matmul(encoder_layer_1, weights['decoder_h1']), 
                             biases['decoder_b1'])
    y_pred = decoder_layer_1
    y_true = X

    cost = tf.sqrt(tf.div(tf.reduce_sum(tf.pow(tf.multiply((y_true - y_pred)[np.newaxis, :, :], mask), 2), axis=(1, 2)), 
                          tf.reduce_sum(mask, axis=(1, 2))))
    cost = tf.reduce_sum(tf.multiply(cost, tf.constant(lambdas, dtype='float')))
    cost += reg_U * tf.reduce_sum(tf.pow(weights['encoder_h1'], 2))
    cost += reg_V * tf.reduce_sum(tf.pow(weights['decoder_h1'], 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    return X, mask, cost, optimizer, y_pred


def fit_ae(R, hard_pred_indices, n_hidden_1, method='user_wise'):
    num_subsets = hard_pred_indices.shape[0] - 1
    if not (method == 'user_wise' or method == 'ui_concat'):
        raise ValueError('Unknown method')
    
    if method == 'user_wise':
        n_input = R.shape[1]
        learning_rate = 0.001
        n_epoch = 40
        batch_size = 32
        batch_count = int(math.ceil(R.shape[0] / batch_size))
    elif method == 'ui_concat':
        n_input = sum(R.shape)
        learning_rate = 0.001
        n_epoch = 5
        batch_size = 256
        nnz_indices = zip(*np.where(R))
        np.random.shuffle(nnz_indices)
        batch_count = int(math.ceil(len(nnz_indices) / batch_size))
        nnz_indices = map(np.array, zip(*nnz_indices))
        
    X, mask, cost, optimizer, X_pred = create_architecture(n_input, n_hidden_1, num_subsets, learning_rate)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    
    try:
        sess.run(init)

        for epoch in range(n_epoch):
            for i in range(batch_count):
                if method == 'user_wise':
                    X_batch = R.iloc[i*batch_size : (i+1)*batch_size, :]
                    mask_batch = hard_pred_indices[:, i*batch_size : (i+1)*batch_size, :]
                elif method == 'ui_concat':
                    u_batch = R.iloc[nnz_indices[0][i*batch_size : (i+1)*batch_size], :].values
                    i_batch = R.iloc[:, nnz_indices[1][i*batch_size : (i+1)*batch_size]].values.T
                    X_batch = np.hstack((u_batch, i_batch))
                    u_mask_batch = hard_pred_indices[:, nnz_indices[0][i*batch_size : (i+1)*batch_size], :]
                    i_mask_batch = hard_pred_indices[:, :, nnz_indices[1][i*batch_size : (i+1)*batch_size]]
                    i_mask_batch = np.transpose(i_mask_batch, axes=(0, 2, 1))
                    mask_batch = np.concatenate((u_mask_batch, i_mask_batch), axis=2)

                _, c = sess.run([optimizer, cost], feed_dict={X: X_batch, mask: mask_batch})
            print('Epoch: {:03d}, cost = {:.5f}'.format(epoch+1, c))

        print("Optimization finished")
    except KeyboardInterrupt:
        print("Optimization was interrupted")
    
    return sess, X, X_pred


def sma_with_ae(R, train_mask, R_appr, rank, method='user_wise'):
    num_subsets = 3
    hard_pred_indices = compute_hard_pred_subsets(R, train_mask, R_appr, num_subsets)
    sess, X, X_pred = fit_ae(R * train_mask, hard_pred_indices, rank, method=method)
    return sess, X, X_pred