import numpy as np
import torch
import os
import pdb
from math import radians, cos, sin, asin, sqrt
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pickle

os.chdir("C:\\Users\\罗颖涛\\PycharmProjects\\POI")
# os.chdir("C:\\Users\\Administrator\\PycharmProjects\\POI")


def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r


def euclidean(point, each):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    lon1, lat1, lon2, lat2 = point[2], point[1], each[2], each[1]
    return np.sqrt((lon1 - lon2)**2 + (lat1 - lat2)**2)


def create_adj(fname):
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
        # poi.sort(key=lambda t: t[0])
    poi = np.array(poi)
    poi[:, 0] += 1  # start from 1
    np.save('./data/' + fname + '_POI.npy', poi)

    # build adjacent list
    adj = './data/' + fname + '_adj.txt'
    dis = './data/' + fname + '_dis.txt'
    file = open(adj, 'a')
    a = open(adj, 'w')
    file = open(dis, 'a')
    d = open(dis, 'w')
    amount = 0
    for i in range(poi.shape[0]):
        print(i, amount)
        for j in range(poi.shape[0]):
            dis = haversine(poi[i, 2], poi[i, 1], poi[j, 2], poi[j, 1])
            # adj = 1 if dis <= 8 and i is not j else 0  # GOW
            # adj = 1 if dis <= 0.2 and i is not j else 0  # SIN
            adj = 1 if dis <= 0.5 and i is not j else 0  # NYC & TKY
            if adj is not 0:
                a.write(str(j) + '\t')
                d.write(str(dis) + '\t')
                amount += 1
        a.write('\n')  # write adj list for node i
        d.write('\n')
    a.close()
    d.close()


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

    np.save('./data/' + fname + '_data.npy', data)


def group_region(fname, traj, u_dim):
    poi = np.load('./data/' + fname + '_POI.npy')  # location id start from 1
    gr = './data/' + fname + '_group.txt'
    file = open(gr, 'a')
    g = open(gr, 'wb')

    group = []  # (user, *num, ?, [id, lat, lon])
    for i in range(u_dim):
        user = []
        for item in traj:
            if item[0, 0].item() == i + 1:  # the user id to whom the traj belongs to is i+1
                user += item[:, 1].numpy().tolist()  # visited points in this traj
        if not user:
            continue
        locs = poi[:, 0]
        pos = []
        for item in user:
            pos.append(poi[np.where(locs == item)][0])
        pos = np.array(pos)  # poi of the user' all visited points (repeated)

        DB = DBSCAN(eps=0.002, min_samples=10).fit(pos[:, 1:]).labels_  # eps 0.01 -- dis 0.84174
        base = []  # (num, ?, [id, lat, lon])
        region, tmp = [], []
        num_group = DB.max() + 1  # start from 0 to max
        for num in range(num_group):
            base.append(pos[np.where(DB == num)].tolist())
            for point in poi.tolist():
                for each in base[num]:
                    # get all points within the region, dis less than 0.01 to at least one base point
                    if euclidean(point, each) < (0.001):  # half it here
                        tmp.append(point[0])
                        break
            region.append(tmp)  # region (num, ?)
            print(len(tmp))
            tmp = []
        print('...')
        group.append(region)

    pickle.dump(group, g)
    g.close()
    return group  # (user, *num, ?)


def load_adj(dname):  # start from 0
    adj_list, dis_list = [], []
    with open('./data/' + dname + '_adj.txt') as fd:
        for line in fd:
            fs = line.split('\t')[:-1]
            fs = list(map(int, fs))
            adj_list.append(fs)

    with open('./data/' + dname + '_dis.txt') as fd:
        for line in fd:
            fs = line.split('\t')[:-1]
            fs = list(map(float, fs))
            dis_list.append(fs)
    return adj_list, dis_list


def load_traj(dname):  # start from 1
    gr = './data/' + dname + '_group.txt'
    file = open(gr, 'a')
    g = open(gr, 'wb')

    # data (num_visit, [u, l, t]), +1 to avoid 0 as padding
    # poi (num_loc, [l, lat, lon])
    group = []  # (user, *num, ?, [id, lat, lon])
    poi = np.load('./data/' + dname + '_POI.npy')  # location id start from 1
    data = np.load('./data/' + dname + '_data.npy') + 1
    data = data[np.argsort(data[:, 0])]  # sort user
    delta_t = 60 * 24 * 1  # min * hour * day

    trajs_tmp, labels_tmp, traj, his_seq, tmp, u_id = [], [], [], np.array([]), [], 1
    train_trajs, test_trajs, train_labels, test_labels = [], [], [], []
    # re_user_id = 1

    for item in data:  # item ([u, l, t])
        if u_id == item[0]:  # get all info about user u_id
            tmp.append(item)
        else:  # build time-ranked info and next user
            if len(tmp) is 0:  # if no u_id in data, continue
                continue
            tmp = np.array(tmp)
            his_seq = tmp[np.argsort(tmp[:, 2])].copy()  # sort location, (num_id, [u, l, t])
            # his_seq[:, 0] = re_user_id  # delete redundant user ids

            visited_locs = his_seq[:, 1]
            if len(visited_locs) == 0:
                continue
            loc_ids = poi[:, 0]
            pos = []
            for visited_loc in visited_locs:
                pos.append(poi[np.where(loc_ids == visited_loc)][0])
            pos = np.array(pos)  # poi of the user' all visited points (repeated)

            DB = DBSCAN(eps=0.002, min_samples=10).fit(pos[:, 1:]).labels_  # eps 0.01 -- dis 0.84174
            num_group = DB.max() + 1  # start from 0 to max
            neg_num = len(DB[np.where(DB == -1)])
            print(DB)
            if len(DB) < neg_num*2:
                print('no!')
                continue
            base = []  # (num, ?, [id, lat, lon])
            region, reg_tmp = [], []
            for num in range(num_group):
                base.append(pos[np.where(DB == num)].tolist())
                for point in poi.tolist():
                    for every in base[num]:
                        # get all points within the region, dis less than 0.01 to at least one base point
                        if euclidean(point, every) < 0.005:  # half it here
                            reg_tmp.append(point[0])
                            break
                region.append(reg_tmp)  # region (num, ?)
                print(len(reg_tmp))
                reg_tmp = []

            # from historical sequence build trajectories
            for j, each in enumerate(his_seq):  # each ([u, l, t])
                each = each.tolist()
                if j == 0:
                    traj.append(each)  # traj (len_traj, [u, l, t])
                    continue
                if each[2] - his_seq[j - 1, 2] <= delta_t:
                    traj.append(each)
                else:
                    if len(traj) > 3:  # traj fewer than 3 check-ins are removed
                        # (?num_traj, ?len_traj-1, [u, l, t])
                        trajs_tmp.append(torch.LongTensor(traj[:-1]))
                        labels_tmp.append(traj[-1][1])  # loc of last one
                    traj = [each]

            if len(traj) > 3:
                trajs_tmp.append(torch.LongTensor(traj[:-1]))
                labels_tmp.append(traj[-1][1])

            traj = []
            tmp = []
            tmp.append(item)

            # user fewer than 5 trajs are removed
            if len(trajs_tmp) > 5:
                train_trajs = train_trajs + trajs_tmp[:-1]
                test_trajs = test_trajs + trajs_tmp[-1:]
                train_labels = train_labels + labels_tmp[:-1]
                test_labels = test_labels + labels_tmp[-1:]
                # re_user_id += 1
            trajs_tmp = []
            labels_tmp = []
            u_id += 1

    return [train_trajs, test_trajs], [train_labels, test_labels]  # (N, *len, 3), (N)


if __name__ == '__main__':
    create_adj('NYC')
    create_data('NYC')
