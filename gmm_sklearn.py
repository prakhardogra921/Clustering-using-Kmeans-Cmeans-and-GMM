import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
from scipy import linalg
import itertools
import matplotlib as mpl
import random

size = 500

set1 = np.random.normal(loc = 1, scale = 0.1, size = size)
set2 = np.random.normal(loc = 1.5, scale = 0.1, size = size)
set3 = np.random.normal(loc = 2, scale = 0.2, size = size)

p1 = 0.25
p2 = 0.5
p3 = 0.25
#print(set1)
dset = np.array(random.sample(list(set1), int(p1*size)) + random.sample(list(set2), int(p2*size)) + random.sample(list(set3), int(p3*size)))


set4 = np.random.normal(loc = 1, scale = 0.3, size = size)
set5 = np.random.normal(loc = 1.5, scale = 0.4, size = size)
set6 = np.random.normal(loc = 2, scale = 0.3, size = size)
dset2 = np.array(random.sample(list(set4), int(p1*size)) + random.sample(list(set5), int(p2*size)) + random.sample(list(set6), int(p3*size)))


def generate_gmm(set):
    gmm = mixture.GaussianMixture(n_components=3)
    gmm.fit(set.reshape(-1, 1))
    y = gmm.predict(set.reshape(-1, 1))
    plt.scatter(set, y)
    plt.title('GMM')
    plt.show()

#generate_gmm(set1)
#generate_gmm(set2)
#generate_gmm(set3)
#generate_gmm(np.array(random.sample(list(set1), int(p1*len(set1))) + random.sample(list(set2), int(p2*len(set2))) + random.sample(list(set3), int(p3*len(set3)))))
generate_gmm(dset)
"""
gmm = mixture.GaussianMixture(n_components=2)
gmm.fit(set1.reshape(-1, 1))
y1 = gmm.predict(set1.reshape(-1, 1))
splot = plt.subplot(2, 1, 2)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
for i, (mean, cov, color) in enumerate(zip(gmm.means_, gmm.covariances_, color_iter)):
    v, w = linalg.eigh(cov)
    if not np.any(y1 == i):
        continue
    plt.scatter(set1[y1 == i, 0], set1[y1 == i, 1], .8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(.5)
    splot.add_artist(ell)
plt.xticks(())
plt.yticks(())
plt.title('GMM')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.show()
"""


