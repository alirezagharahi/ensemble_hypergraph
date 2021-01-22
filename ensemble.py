# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import os
from scipy import spatial
import math
import scipy.sparse as sp
import implicit
import random

def create_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--input_dataset', default='',
            help='Input path of the dataset.')

    return parser

def threshold_interactions_df(df, row_name, col_name, row_min, col_min):
    """
    Desc:
        Using this function, we can consider user and items with more specified number of interactions.
        Then we know than we are using an interaction matrix without user and item cold-start problem
    ------
    Input:
        df: the dataframe which contains the interactions (df)
        row_name: column name referring to user (str)
        col_nam: column name referring to item (str)
        row_min: the treshold for number of interactions of users (int)
        col_min: the treshold for number of interactions of items (int)
    ------
    Output:
        dataframe in which users and items have more that thresholds interactions (df)
    """
    n_rows = df[row_name].unique().shape[0]
    n_cols = df[col_name].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
    print('Starting interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of cols: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))

    done = False
    while not done:
        starting_shape = df.shape[0]
        col_counts = df.groupby(row_name)[col_name].count()
        df = df[~df[row_name].isin(col_counts[col_counts < col_min].index.tolist())]
        row_counts = df.groupby(col_name)[row_name].count()
        df = df[~df[col_name].isin(row_counts[row_counts < row_min].index.tolist())]
        ending_shape = df.shape[0]
        if starting_shape == ending_shape:
            done = True

    n_rows = df[row_name].unique().shape[0]
    n_cols = df[col_name].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
    print('Ending interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of columns: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))
    return df

def train_test_split(interactions, split_count, fraction=None):
    """
    Desc:
        Using this function, we split avaialble data to train and test set.
    ------
    Input:
        interactions : interaction between users and streams (scipy.sparse matrix)
        split_count : number of interactions per user to move from training to test set (int)
        fraction : fraction of users to split their interactions train/test. If None, then all users (float)
    ------
    Output:
        train_set (scipy.sparse matrix)
        test_set (scipy.sparse matrix)
        user_index
    """
    train = interactions.copy().tocoo()
    test = sp.lil_matrix(train.shape)

    if fraction:
        try:
            user_index = np.random.choice(
                np.where(np.bincount(train.row) >= split_count * 2)[0],
                replace=False,
                size=np.int64(np.floor(fraction * train.shape[0]))
            ).tolist()
        except:
            print(('Not enough users with > {} '
                  'interactions for fraction of {}')\
                  .format(2*split_count, fraction))
            raise
    else:
        user_index = range(train.shape[0])

    train = train.tolil()

    for user in user_index:
        test_interactions = np.random.choice(interactions.getrow(user).indices,
                                        size=split_count,
                                        replace=False)
        train[user, test_interactions] = 0.
        test[user, test_interactions] = interactions[user, test_interactions]

    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_index

def get_df_matrix_mappings(df, row_name, col_name):
    """
    Desc:
        Using this function, we are able to get mappings between original indexes and new (reset) indexes
    ------
    Input:
        df: the dataframe which contains the interactions (df)
        row_name: column name referring to user_id (str)
        col_nam: column name referring to stream_slug (str)
    ------
    Output:
        rid_to_idx: a dictionary contains mapping between real row ids and new indexes (dict)
        idx_to_rid: a dictionary contains mapping between new indexes and real row ids (dict)
        cid_to_idx: a dictionary contains mapping between real column ids and new indexes (dict)
        idx_to_cid: a dictionary contains mapping between new indexes and real column ids (dict)
    """

    rid_to_idx = {}
    idx_to_rid = {}
    for (idx, rid) in enumerate(df[row_name].unique().tolist()):
        rid_to_idx[rid] = idx
        idx_to_rid[idx] = rid

    cid_to_idx = {}
    idx_to_cid = {}
    for (idx, cid) in enumerate(df[col_name].unique().tolist()):
        cid_to_idx[cid] = idx
        idx_to_cid[idx] = cid

    return rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid


def df_to_matrix(df, row_name, col_name,value):
    """
    Desc:
        Using this function, we transfrom the interaction matrix to scipy.sparse matrix
    ------
    Input:
        df: the dataframe which contains the interactions (df)
        row_name: column name referring to user_id (str)
        col_nam: column name referring to stream_slug (str)
        value: the value of interaction between row and column
    ------
    Output:
        interactions: Sparse matrix contains user and streams interactions (sparse csr)
        rid_to_idx: a dictionary contains mapping between real row ids and new indexes (dict)
        idx_to_rid: a dictionary contains mapping between new indexes and real row ids (dict)
        cid_to_idx: a dictionary contains mapping between real column ids and new indexes (dict)
        idx_to_cid: a dictionary contains mapping between new indexes and real column ids (dict)
    """


    rid_to_idx, idx_to_rid,cid_to_idx, idx_to_cid = get_df_matrix_mappings(df,row_name,col_name)

    def map_ids(row, mapper):
        return mapper[row]

    I = df[row_name].apply(map_ids, args=[rid_to_idx]).as_matrix()
    J = df[col_name].apply(map_ids, args=[cid_to_idx]).as_matrix()
    V = df[value]
    interactions = sp.coo_matrix((V, (I, J)), dtype=np.float64)
    interactions = interactions.tocsr()
    return interactions, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid

def build_E1(inter):
    """
    Desc:
        This function builds the E1 partition related to user-article interactions.
    ------
    Input:
        interactions
    ------
    Output:
        E1 hyperedges
    """
    n_users = inter.shape[0]
    interactions = inter.toarray()
    nb_of_es = n_users
    e1 = np.eye(nb_of_es)
    e2 = interactions.transpose()

    E1 = np.vstack((e1,e2))
    return E1

def build_E2(inter):
    """
    Desc:
        This function builds the E2 partition related to user similarities based on user interaction vector.
    ------
    Input:
        interactions
    ------
    Output:
        E2 hyperedges
    """
    n_users = inter.shape[0]
    n_articles = inter.shape[1]
    interactions = inter.toarray()
    nb_of_es = n_users
    e1 = np.eye(nb_of_es)
    e2 = np.zeros([n_articles,nb_of_es])
    matrix = interactions
    tree = spatial.KDTree(matrix)
    for user in range(n_users):
        target_embedding = matrix[user]
        neighbors_index = tree.query(target_embedding,9)[1]
        e1[neighbors_index,user] = 0.5
    E2 = np.vstack((e1,e2))

    return E2

def build_E_model(train_set,model,k):
    """
    Desc:
        This function builds the E_model partition related to model recommendations.
    ------
    Input:
        interactions
    ------
    Output:
        E model hyperedges
    """
    n_users = train_set.shape[0]
    interactions = train_set.toarray()
    nb_of_es = n_users
    e1 = np.eye(nb_of_es)
    e2 = interactions.transpose()
    for user in range(n_users):
        rank = model.recommend(userid=user,user_items=train_set.tocsr(),filter_already_liked_items=True)
        ranking = [x[0] for x in rank]
        for i in ranking[:k]:
            e2[i,user] = 0.5

    E3 = np.vstack((e1,e2))
    return E3

def precision_k (train_set,test_set,model,k=10):
    """
    Input:
        train_set : train_set (scipy.sparse matrix)
        train_set : test_set (scipy.sparse matrix)
        u_matrix: learned user low-rank matrice (numpy.ndarray)
        i_matrix: learned item low-rank matrice (numpy.ndarray)
        k: @k (int)
    ------
    Output:
        precision(float)
    """
    num_users,num_items=train_set.shape
    precision=[]
    for user in range(num_users):
        ranking = model.recommend(userid=user,user_items=train_set.tocsr(),filter_already_liked_items=True)
        ranking = [x[0] for x in ranking]
        r=[]
        for item_id in ranking:
            real_value = test_set[user,item_id]
            if real_value==0:
                r.append(real_value)
            else:
                r.append(1)
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        precision.append(np.mean(r))
    return np.mean(precision)

def precision_hybrid_k (train_set,test_set,model1,model2,weight,k=10):
    """
    Input:
        train_set : train_set (scipy.sparse matrix)
        train_set : test_set (scipy.sparse matrix)
        u_matrix: learned user low-rank matrice (numpy.ndarray)
        i_matrix: learned item low-rank matrice (numpy.ndarray)
        k: @k (int)
    ------
    Output:
        precision(float)
    """
    num_users,num_items=train_set.shape
    precision=[]
    for user in range(num_users):
        ranking_1 = model1.recommend(userid=user,user_items=train_set.tocsr(),filter_already_liked_items=True,N=15)
        ranking_2 = model2.recommend(userid=user,user_items=train_set.tocsr(),filter_already_liked_items=True,N=15)
        r1_dict = {x[0]:x[1] for x in ranking_1}
        r2_dict = {x[0]:x[1] for x in ranking_2}
        new_scores = {k: weight * r1_dict.get(k, 0) + (1-weight) * r2_dict.get(k, 0) for k in (set(r1_dict) | set(r2_dict))}
        ranking = sorted(new_scores.items(), key=lambda x: x[1],reverse=True)
        ranking = [x[0] for x in ranking]

        r=[]
        for item_id in ranking:
            real_value = test_set[user,item_id]
            if real_value==0:
                r.append(real_value)
            else:
                r.append(1)
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        precision.append(np.mean(r))
    return np.mean(precision)


def precision_hyper (train_set,test_set,F,k=10):

    num_users,num_items=train_set.shape
    precision=[]
    for user in range(num_users):
        y = np.zeros([num_users+num_items])
        y[user] = 1
        rv = F.dot(y)[num_users:num_users+num_items]
        recommendation = np.argsort(-rv)

        train_id = train_set[user].nonzero()[1]
        ranking = [x for x in recommendation if x not in train_id]

        r=[]
        for item_id in ranking:
            real_value = test_set[user,item_id]
            if real_value==0:
                r.append(real_value)
            else:
                r.append(1)
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        precision.append(np.mean(r))
    return np.mean(precision)


def build_affinity(H):
    """
    This function builds the affinity matrix A and the final optimization solution.
    """
    Edegrees = np.sum(H,0)**(-1)
    Vdegrees = np.sum(H,1)**(-1./2)
    Edegrees[np.where(np.isinf(Edegrees))]=0
    Vdegrees[np.where(np.isinf(Vdegrees))]=0
    Edegrees[np.where(np.isnan(Edegrees))]=0
    Vdegrees[np.where(np.isnan(Vdegrees))]=0
    De = np.diag(Edegrees)
    Dv = np.diag(Vdegrees)

    A = Dv.dot(H).dot(De).dot(H.T).dot(Dv) # Here, let the weight matrix W be I

    return A

def print_result(result):
    print("Precision@10: ")
    for i in result:
        print(i[0],i[1])

def main():

    parser = create_args_parser()
    args = parser.parse_args()

# set variables and parameters based on the dataset
    input = args.input_dataset
    if "movie" in input:
        row_id,column_id = 'userID','movieID'
        sep = '\t'
        row_min,column_min = 100,100
        bpr_f,bpr_i,bpr_r,bpr_lr = 129,1984,0.0412211670514582,0.00916093538495639
        als_f,als_i,als_r = 107,1393,0.0225369671263697
        w_h,h_r,h_e_r = 0.66635111606741,0.241367197412936,0.4554

    elif "aotm" in input:
        row_id,column_id = 'UserId','ItemId'
        sep = '\t'
        row_min,column_min = 35,35
        bpr_f,bpr_i,bpr_r,bpr_lr = 179,1645,0.019443559078079,0.0283640701834258
        als_f,als_i,als_r = 201,1276,0.0374087800158703
        w_h,h_r,h_e_r = 0.66635111606741,0.28162,0.805

    elif "chamealon" in input:
        row_id,column_id = 'userId','contentId'
        sep = ','
        row_min,column_min = 50,50
        bpr_f,bpr_i,bpr_r,bpr_lr = 168,1598,0.037443718743735,0.0174401403559288
        als_f,als_i,als_r = 152,1129,0.0432292432067999
        w_h,h_r,h_e_r = 0.398649112750105,0.648542543892749,0.8301
    else:
        print("the input dataset is not defined")
    #reading the data
    interactions = pd.read_csv(input,sep=sep)

    # preprocess
    interactions.rename(columns={row_id:'userId',column_id:'contentId'},inplace=True)
    interactions = interactions.drop_duplicates(subset=['userId', 'contentId'], keep='last')
    np.random.seed(7)
    random.seed(7)
    interactions['value']=1
    interaction = threshold_interactions_df(interactions,'userId','contentId',row_min,column_min)
    inter, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid = df_to_matrix(interaction,'userId','contentId','value')
    train_set, test_set, user_index = train_test_split(inter, 10, fraction=None)

    # creating baseliens
    als = implicit.als.AlternatingLeastSquares(factors=107,iterations=als_i,regularization=als_r, random_state=7) #alsmf
    bpr = implicit.bpr.BayesianPersonalizedRanking(factors=bpr_f,iterations=bpr_i,regularization=bpr_r,learning_rate=bpr_lr,num_threads=1, random_state=7) #bpr

    result = []

    # creating hypergraph baseliens
    E1 = build_E1(train_set)
    E2 = build_E2(train_set)

    H = np.hstack((E1,E2))
    A = build_affinity(H)

    Atemp = A.copy()
    y = np.zeros([Atemp.shape[0]])

    # hyper graph
    F = np.linalg.inv(np.eye(np.size(Atemp,0))-h_r*Atemp)
    precision_E1_E2 = precision_hyper (train_set,test_set,F,k=10)
    result.append(("Hypergraph :",precision_E1_E2))
    # fit baselines
    bpr.fit(train_set.transpose())
    als.fit(train_set.transpose())

    precision_bpr = precision_k (train_set,test_set,bpr,k=10)
    precision_als = precision_k (train_set,test_set,als,k=10)
    precision_als_hybrid = precision_hybrid_k (train_set,test_set,bpr,als,w_h,k=10)
    result.append(("BPR :",precision_bpr))
    result.append(("WRMF :",precision_als))
    result.append(("Weighted_Hybrid :",precision_als_hybrid))

    # creating ensemble
    E3=build_E_model(train_set,bpr,15)
    E4=build_E_model(train_set,als,15)

    H2 = np.hstack((E1,E2,E3,E4))
    A_2 = build_affinity(H2)
    Aensemble = A_2.copy()

    F_ensemble = np.linalg.inv(np.eye(np.size(Aensemble,0))-h_e_r*Aensemble)
    precision_ensemble = precision_hyper(train_set,test_set,F_ensemble,k=10)
    result.append(("Ensemble :",precision_ensemble))
    print_result(result)
if __name__ == '__main__':
    main()
