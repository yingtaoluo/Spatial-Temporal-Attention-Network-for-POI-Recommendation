import numpy as np
import torch
import os
import pdb
import matplotlib.pyplot as plt
from STATTN.load import haversine, load_traj
from sklearn.cluster import DBSCAN

# os.chdir("C:\\Users\\罗颖涛\\PycharmProjects\\GCN-POI")
os.chdir("C:\\Users\\Administrator\\PycharmProjects\\POI")

dname = 'NYC'
[traj, _], [_, _] = load_traj(dname)
data = np.load('./data/' + dname + '_data.npy') + 1
poi = []  # x: latitude, y: longitude
with open('./data/' + dname + '_POI.csv', 'r') as fp:
    for i, line in enumerate(fp):
        # ignore the headers
        if i is 0:
            continue

        id, x, y = line.split(',')
        y = y[:-1]
        id, x, y = int(id), float(x), float(y)
        poi.append([id, x, y])
    # poi.sort(key=lambda t: t[0])
poi = np.array(poi) + 1
print(haversine(lon1=-74.04, lat1=40.8, lon2=-74.03, lat2=40.8))

color = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
for i in range(100):
    if i < 10:
        continue
    user = []
    for item in traj:
        if item[0, 0].item() == i+1:  # the user id to whom the traj belongs to is i+1
            user += item[:, 1].numpy().tolist()  # visited points in this traj
    if not user: continue
    locs = poi[:, 0]
    pos = []
    for item in user:
        pos.append(poi[np.where(locs==item)][0])
    pos = np.array(pos)

    DB = DBSCAN(eps=0.005, min_samples=10).fit(pos[:, 1:]).labels_
    pos0 = []
    num_group = DB.max() + 1
    plt.figure()
    plt.scatter(pos[:, 2]-1, pos[:, 1]-1, color='black', s=3)  # 2， 1： longitude, latitude; 经度，纬度
    for num in range(num_group):
        if num >= 6:  break
        for i, item in enumerate(pos):
            if DB[i] == num:
                pos0.append(item)
        pos0 = np.array(pos0)
        plt.scatter(pos0[:, 2]-1, pos0[:, 1]-1, color=color[num], s=3)
        print(pos0[:, 1:]-1, '\n')
        print('\n')
        pos0 = []
    print(DB)
    plt.show()
    plt.close()

'''
for i in range(100):
    user = np.array(traj[i] - 1)
    locs = poi[:, 0]
    pos = []
    for item in user[:, 1]:
        pos.append(poi[np.where(locs==item)][0])
    pos = np.array(pos)
    print(user)
    print(pos)
    plt.scatter(poi[:, 2], poi[:, 1])  # 2， 1： longitude, latitude; 经度，纬度
    plt.scatter(pos[:, 2], pos[:, 1], color='red')
    plt.show()
    plt.close()
'''
'''
for i in range(100):
    user = data[np.where(data[:, 0] == i + 1)][:, 1]
    loc = []
    for item in poi:
        if item[0] + 1 in user:
            loc.append(item)
            print('time', user)
            plt.scatter(poi[:, 2], poi[:, 1])  # 2， 1： longitude, latitude; 经度，纬度
            plt.scatter(item[2], item[1], color='red')
            plt.show()
            plt.close()
    loc = np.array(loc)
    print(user)

    plt.scatter(poi[:, 2], poi[:, 1])  # 2， 1： longitude, latitude; 经度，纬度
    plt.scatter(loc[:, 2], loc[:, 1], color='red')
    plt.show()
    plt.close()
'''




