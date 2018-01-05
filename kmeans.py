import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import random
from random import sample
from math import sqrt
from numpy import mean
import copy
from sklearn import metrics
import time
import seaborn as sns
from scipy.stats import norm

total_distances = []

size = 500

set1 = np.random.normal(loc = 1, scale = 0.1, size = size)
set2 = np.random.normal(loc = 1.5, scale = 0.1, size = size)
set3 = np.random.normal(loc = 2, scale = 0.2, size = size)

p1 = 0.25
p2 = 0.5
p3 = 0.25

dset = np.array(random.sample(list(set1), int(p1*size)) + random.sample(list(set2), int(p2*size)) + random.sample(list(set3), int(p3*size)))

set4 = np.random.normal(loc = 1, scale = 0.3, size = size)
set5 = np.random.normal(loc = 1.5, scale = 0.4, size = size)
set6 = np.random.normal(loc = 2, scale = 0.3, size = size)
dset2 = np.array(random.sample(list(set4), int(p1*size)) + random.sample(list(set5), int(p2*size)) + random.sample(list(set6), int(p3*size)))

start_time = time.time()

def initialize_centers(df, k):
    random_indices = sample(range(size), k)
    centers = []
    for id in random_indices:
        centers.append(df[id])
    print("Random Indices : " + str(random_indices))
    return centers

def compute_center(df, k, cluster_labels):
    cluster_centers = list()
    data_points = list()
    for i in range(k):
        for idx, val in enumerate(cluster_labels):
            if val == i:
                data_points.append([df[idx]])
        cluster_centers.append(map(mean, zip(*data_points)))
    return cluster_centers

def euclidean_distance(x, y):
    summ = 0
    for i in range(len(x)):
        term = (x[i] - y[i]) ** 2
        summ += term
    return sqrt(summ)

def assign_cluster(df, cluster_centers):
    cluster_assigned = list()
    for i in range(size):
        distances = []
        distances2 = []
        for center in cluster_centers:
            distance = euclidean_distance([df[i]], [center])
            distance2 = distance ** 2
            distances.append(distance)
            distances2.append(distance2)
        total_distance = sum(distances2)/size
        min_dist, idx = min((val, idx) for (idx, val) in enumerate(distances))
        cluster_assigned.append(idx)
    total_distances.append(total_distance)
    return cluster_assigned


def kmeans(df, k):
    cluster_centers = initialize_centers(df, k)
    curr = 0
    while curr < MAX_ITER:
        cluster_labels = assign_cluster(df, cluster_centers)
        cluster_centers = compute_center(df, k, cluster_labels)
        curr += 1
    return cluster_labels, cluster_centers


k = 3
MAX_ITER = 15

labels, centers = kmeans(dset, k)
print (time.time() - start_time)

plt.plot(range(MAX_ITER), total_distances)
plt.show()
