import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,MiniBatchKMeans

import argparse
import pickle

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('-k', type=int,default=1000,
                    help='number of clusters')

parser.add_argument('-d', type=int,default=2,
                    help='number of dimensions')
parser.add_argument('-p', type=int,default=50000,
                    help='number of sampled points')
parser.add_argument('-f', type=str,default="",
                    help='directory in which to put the files')
parser.add_argument('-s', type=int,default=1,
                    help='seed number')
parser.add_argument('-t', type=str,default="",
                    help='a tag to help identify (e.g. run number)')
parser.add_argument('-j', type=int,default=-1,
                    help='the number of parallel jobs for KMeans')
parser.add_argument('-l', type=str,default="",
                    help='prefix to batch files')
parser.add_argument('-b', type=int ,default=0,
                    help='batch size for batch-based k-means')

args = parser.parse_args()

k = args.k
DIM = args.d
NumberOfSampedPoints = args.p

np.random.seed(args.s)

X = np.random.rand(NumberOfSampedPoints,DIM)


if args.b > 0:
    k_means = MiniBatchKMeans(init='k-means++', n_clusters=k, n_init=1, verbose=1,batch_size=args.b)
    j=0
    for i in range(0,args.p,args.b):
        X = pickle.load(open(args.l+str(j),"rb"))
        k_means.partial_fit(X)
        j+=1
else:
    k_means = KMeans(init='k-means++', n_clusters=k, n_init=1, n_jobs=args.j, verbose=1)
    k_means.fit(X)

k_means_cluster_centers = k_means.cluster_centers_
args.f = args.f if args.f=="" else args.f+"/"
filename = args.f+'centroids_' + str(k) + '_' + str(DIM) + '%s.dat'%(args.t)

with open(filename, 'w') as f:
    for p in k_means_cluster_centers:
       for item in p:
            f.write(str(item) + ' ')
       f.write('\n')


    
