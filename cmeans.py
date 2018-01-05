import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import random
from random import sample
from math import sqrt
from numpy import mean
import operator
import math
from sklearn import metrics
import time

size = 500

set1 = np.random.normal(loc = 1, scale = 0.1, size = size)
set2 = np.random.normal(loc = 1.5, scale = 0.1, size = size)
set3 = np.random.normal(loc = 2, scale = 0.2, size = size)

p1 = 0.25
p2 = 0.5
p3 = 0.25

df = np.array(random.sample(list(set1), int(p1*size)) + random.sample(list(set2), int(p2*size)) + random.sample(list(set3), int(p3*size)))
df_labels = np.array([0]*int(p1*size) + [1]*int(p2*size) + [2]*int(p3*size)) #true cluster labels

set4 = np.random.normal(loc = 1, scale = 0.3, size = size)
set5 = np.random.normal(loc = 1.5, scale = 0.4, size = size)
set6 = np.random.normal(loc = 2, scale = 0.3, size = size)
df2 = np.array(random.sample(list(set4), int(p1*size)) + random.sample(list(set5), int(p2*size)) + random.sample(list(set6), int(p3*size)))

k = 3
MAX_ITER = 100
m = 2.00
start_time = time.time()

def initialize_membership_matrix():
    membership_mat = list()
    for i in range(size):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        membership_mat.append(temp_list)
    return membership_mat

def calculate_cluster_center(membership_mat):
    cluster_mem_val = zip(*membership_mat)
    cluster_centers = list()
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [e ** m for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(size):
            data_point = [df[i]]
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers


def update_membership_value(membership_mat, cluster_centers):
    p = float(2/(m-1))
    for i in range(size):
        x = [df[i]]
        distances = [np.linalg.norm(map(operator.sub, x, cluster_centers[j])) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)
    return membership_mat


def get_clusters(membership_mat):
    cluster_labels = list()
    for i in range(size):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fuzzy_c_means_clustering():
    membership_mat = initialize_membership_matrix()
    curr = 0
    while curr <= MAX_ITER:
        cluster_centers = calculate_cluster_center(membership_mat)
        membership_mat = update_membership_value(membership_mat, cluster_centers)
        cluster_labels = get_clusters(membership_mat)
        curr += 1
    return cluster_labels, cluster_centers

labels, centers = fuzzy_c_means_clustering()
print (time.time() - start_time)
print ("Silhouette Coefficient: %0.5f", metrics.silhouette_score(df.reshape(-1, 1), labels))
print labels
print df_labels
print (metrics.accuracy_score(df_labels, labels))