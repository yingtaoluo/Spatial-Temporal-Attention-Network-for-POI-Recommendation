import numpy as np
import torch
import os
import pdb
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import joblib
from torch.nn.utils.rnn import pad_sequence

max_len = 100  # max traj len; i.e., M


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def euclidean(point, each):
    lon1, lat1, lon2, lat2 = point[2], point[1], each[2], each[1]
    return np.sqrt((lon1 - lon2)**2 + (lat1 - lat2)**2)


def create_poi(fname):
    poi = []  # x: latitude, y: longitude
    with open('./data/' + fname + '_POI.csv', 'r') as fp:
        for i, line in enumerate(fp):
            # ignore the headers
            if i is 0:
                continue

            id, x, y = line.split(',')
            y = y[:-1]
            id, x, y = int(id), float(x), float(y)
            poi.append([id, x, y])
    poi.sort(key=lambda t: t[0])
    poi = np.array(poi)
    poi[:, 0] += 1  # start from 1
    np.save('./data/' + fname + '_POI.npy', poi)


def create_data(fname):
    data = []
    with open('./data/' + fname + '.csv', 'r') as fd:
        for i, line in enumerate(fd):
            # ignore the headers
            if i is 0:
                continue

            c1, c2, c3 = line.split(',')
            c3 = c3[:-1]
            if fname == 'gowalla':  # (u, t, l)
                data.append([int(c1), int(c3), int(c2)])
            else:  # (u, l, t)
                data.append([int(c1), int(c2), int(c3)])
    data = np.array(data) + 1  # +1 to avoid 0 as padding
    data = data[np.argsort(data[:, 0])]  # sort by user id
    np.save('./data/' + fname + '.npy', data)


def rst_mat1(traj, poi):
    # traj (*M, [u, l, t]), poi(L, [l, lat, lon])
    mat = np.zeros((len(traj), len(traj), 2))
    for i, item in enumerate(traj):
        for j, term in enumerate(traj):
            poi_item, poi_term = poi[item[1] - 1], poi[term[1] - 1]  # retrieve poi by loc_id
            mat[i, j, 0] = haversine(lon1=poi_item[2], lat1=poi_item[1], lon2=poi_term[2], lat2=poi_term[1])
            mat[i, j, 1] = abs(item[2] - term[2])
    return mat  # (*M, *M, [dis, tim])


def rs_mat2s(poi, l_max):
    # poi(L, [l, lat, lon])
    candidate_loc = np.linspace(1, l_max, l_max)  # (L)
    mat = np.zeros((l_max, l_max))  # mat (L, L)
    for i, loc1 in enumerate(candidate_loc):
        print(i) if i % 100 == 0 else None
        for j, loc2 in enumerate(candidate_loc):
            poi1, poi2 = poi[int(loc1) - 1], poi[int(loc2) - 1]  # retrieve poi by loc_id
            mat[i, j] = haversine(lon1=poi1[2], lat1=poi1[1], lon2=poi2[2], lat2=poi2[1])
    return mat  # (L, L)


def rt_mat2t(traj_time):  # traj_time (*M+1) triangle matrix
    # construct a list of relative times w.r.t. causality
    mat = np.zeros((len(traj_time)-1, len(traj_time)-1))
    for i, item in enumerate(traj_time):  # label
        if i == 0:
            continue
        for j, term in enumerate(traj_time[:i]):  # data
            mat[i-1, j] = np.abs(item - term)
    return mat  # (*M, *M)


def process_traj(dname):  # start from 1
    # data (?, [u, l, t]), poi (L, [l, lat, lon])
    data = np.load('./data/' + dname + '.npy')
    poi = np.load('./data/' + dname + '_POI.npy')
    num_user = data[-1, 0]  # max id of users, i.e. NUM
    data_user = data[:, 0]  # user_id sequence in data
    trajs, labels, mat1, mat2t, lens = [], [], [], [], []
    u_max, l_max = np.max(data[:, 0]), np.max(data[:, 1])

    for u_id in range(num_user+1):
        if u_id == 0:  # skip u_id == 0
            continue
        init_mat1 = np.zeros((max_len, max_len, 2))  # first mat (M, M, 2)
        init_mat2t = np.zeros((max_len, max_len))  # second mat of time (M, M)
        user_traj = data[np.where(data_user == u_id)]  # find all check-ins of u_id
        user_traj = user_traj[np.argsort(user_traj[:, 2])].copy()  # sort traj by time

        print(u_id, len(user_traj)) if u_id % 100 == 0 else None

        if len(user_traj) > max_len + 1:  # consider only the M+1 recent check-ins
            # 0:-3 are training data, 1:-2 is training label;
            # 1:-2 are validation data, 2:-1 is validation label;
            # 2:-1 are test data, 3: is the label for test.
            # *M would be the real length if <= max_len + 1
            user_traj = user_traj[-max_len-1:]  # (*M+1, [u, l, t])

        # spatial and temporal intervals
        user_len = len(user_traj[:-1])  # the len of data, i.e. *M
        user_mat1 = rst_mat1(user_traj[:-1], poi)  # (*M, *M, [dis, tim])
        user_mat2t = rt_mat2t(user_traj[:, 2])  # (*M, *M)
        init_mat1[0:user_len, 0:user_len] = user_mat1
        init_mat2t[0:user_len, 0:user_len] = user_mat2t

        trajs.append(torch.LongTensor(user_traj)[:-1])  # (NUM, *M, [u, l, t])
        mat1.append(init_mat1)  # (NUM, M, M, 2)
        mat2t.append(init_mat2t)  # (NUM, M, M)
        labels.append(torch.LongTensor(user_traj[1:, 1]))  # (NUM, *M)
        lens.append(user_len-2)  # (NUM), the real *M for every user

    # padding zero to the vacancies in the right
    mat2s = rs_mat2s(poi, l_max)  # contains dis of all locations, (L, L)
    zipped = zip(*sorted(zip(trajs, mat1, mat2t, labels, lens), key=lambda x: len(x[0]), reverse=True))
    trajs, mat1, mat2t, labels, lens = zipped
    trajs, mat1, mat2t, labels, lens = list(trajs), list(mat1), list(mat2t), list(labels), list(lens)
    trajs = pad_sequence(trajs, batch_first=True, padding_value=0)  # (NUM, M, 3)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)  # (NUM, M)

    data = [trajs, np.array(mat1), mat2s, np.array(mat2t), labels, np.array(lens), u_max, l_max]
    data_pkl = './data/' + dname + '_data.pkl'
    open(data_pkl, 'a')
    with open(data_pkl, 'wb') as pkl:
        joblib.dump(data, pkl)


if __name__ == '__main__':
    name = 'NYC'
    # create_poi(name)
    # create_data(name)
    process_traj(name)
