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


def divide_set(S, K):
    # returns K+1 subsets: s_0 = empty set, s_1 + ... + s_K = S
    affiliation_vector = np.random.randint(K, size=np.sum(S))
    indices = np.where(S)
    subsets = np.zeros((K+1, ) + S.shape, dtype=bool)

    for k in range(1, K+1):
        current_subset = affiliation_vector == k
        subsets[k, indices[0][current_subset], indices[1][current_subset]] = True
    
    return subsets


def rmse(A, A_appr, mask=None):
    if mask is None:
        return np.sqrt(np.mean(np.power(A - A_appr, 2).mean()))
    else:
        if isinstance(A, pd.DataFrame):
            return np.sqrt(np.mean(np.power(A.values[mask] - A_appr[mask], 2)))
        else:
            return np.sqrt(np.mean(np.power(A[mask] - A_appr[mask], 2)))
        
        
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
        current_errors_train.append(rmse(R, np.dot(U, V.T)))
        
    return U, V


def improve_approximation(R, U_old, V_old, choice_prob, rank, lambdas):
    R_appr = np.dot(U, V.T)
    av_error = rmse(R, R_appr)
    rho_mask = np.random.rand(*R.shape) <= choice_prob
    diff_mask = np.abs(R - R_appr) <= av_error
    easy_pred_indices = np.logical_not(np.logical_xor(rho_mask, diff_mask))
    nnz_entries = np.zeros(R.shape, dtype=bool)
    nnz_entries[np.where(R)] = True
    hard_pred_indices = np.logical_xor(nnz_entries[np.newaxis, :, :], divide_set(easy_pred_indices, lambdas.shape[0]-1))
    return minimize_final_objective(R, U_old, V_old, hard_pred_indices, lambdas)


def plot_with_outliers(matrix, vmax=5.01):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, vmax=vmax, interpolation='none', cmap=plt.cm.hot)
    plt.colorbar(extend='max').cmap.set_over('blue')