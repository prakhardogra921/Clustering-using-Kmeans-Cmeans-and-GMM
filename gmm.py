import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
sns.set_style("white")
from scipy.stats import norm
import time
from math import sqrt, log, exp, pi
from random import uniform

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

set4.sort()
plt.plot(set4, norm.pdf(set4,1,0.3))

set5.sort()
plt.plot(set5, norm.pdf(set5,1.5,0.4))

set6.sort()
plt.plot(set6, norm.pdf(set6,2,0.3))
plt.show()

sns.distplot(dset)
plt.show()

data = dset2

class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, datum):
        u = (datum - self.mu) / abs(self.sigma)
        y = (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
        return y

    def __repr__(self):
        return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)

best_single = Gaussian(np.mean(data), np.std(data))
x = np.linspace(-6, 8, 200)
g_single = stats.norm(best_single.mu, best_single.sigma).pdf(x)

class GaussianMixture:
    def __init__(self, data, mu_min=min(data), mu_max=max(data), sigma_min=.1, sigma_max=1, mix1=.25, mix2=.5):
        self.data = data
        self.one = Gaussian(uniform(mu_min, mu_max),
                            uniform(sigma_min, sigma_max))
        self.two = Gaussian(uniform(mu_min, mu_max),
                            uniform(sigma_min, sigma_max))
        self.three = Gaussian(uniform(mu_min, mu_max),
                            uniform(sigma_min, sigma_max))

        self.mix1 = mix1
        self.mix2 = mix2 #mix3 can be calculated as 1-(mix1+mix2)

    #calculates Expectation
    def Estep(self):
        self.loglike = 0.
        for datum in self.data:
            wp1 = self.one.pdf(datum) * (self.mix1)
            wp2 = self.two.pdf(datum) * (self.mix2)
            wp3 = self.three.pdf(datum) * (1 - self.mix1 - self.mix2)
            den = wp1 + wp2 + wp3
            wp1 /= den
            wp2 /= den
            wp3 /= den
            self.loglike += log(wp1 + wp2 + wp3)
            yield (wp1, wp2, wp3)

    #performs Maximization
    def Mstep(self, weights):
        (left, mid, rigt) = zip(*weights)
        one_den = sum(left)
        two_den = sum(mid)
        three_den = sum(rigt)
        self.one.mu = sum(w * d / one_den for (w, d) in zip(left, data))
        self.two.mu = sum(w * d / two_den for (w, d) in zip(mid, data))
        self.three.mu = sum(w * d / three_den for (w, d) in zip(rigt, data))
        self.one.sigma = sqrt(sum(w * ((d - self.one.mu) ** 2)
                                  for (w, d) in zip(left, data)) / one_den)
        self.two.sigma = sqrt(sum(w * ((d - self.two.mu) ** 2)
                                  for (w, d) in zip(mid, data)) / two_den)
        self.three.sigma = sqrt(sum(w * ((d - self.three.mu) ** 2)
                                  for (w, d) in zip(rigt, data)) / three_den)
        self.mix1 = one_den / len(data)
        self.mix2 = two_den / len(data)

    def iterate(self, N=1, verbose=False):
        mix.Mstep(mix.Estep())

    def pdf(self, x):
        return (self.mix1) * self.one.pdf(x) + (self.mix2) * self.two.pdf(x) + (1 - self.mix1 - self.mix2) * self.three.pdf(x)

    def __repr__(self):
        return 'GaussianMixture({0}, {1}, {2}, {3}, {4}, {5})'.format(self.one, self.two, self.three, self.mix1, self.mix2, 1 - self.mix1 - self.mix2)

    def __str__(self):
        return 'Mixture: {0}, {1}, {2}, {3}, {4}, {5})'.format(self.one, self.two, self.three, self.mix1, self.mix2, 1 - self.mix1 - self.mix2)


start_time = time.time()
n_iterations = 5
best_mix = None
best_loglike = float('-inf')
mix = GaussianMixture(data)

for _ in range(n_iterations):
    mix.iterate(verbose=True)
    if mix.loglike > best_loglike:
        best_loglike = mix.loglike
        best_mix = mix

n_iterations = 40
n_random_restarts = 500
best_mix = None
best_loglike = float('-inf')

for _ in range(n_random_restarts):
    mix = GaussianMixture(data)
    for _ in range(n_iterations):
        mix.iterate()
        if mix.loglike > best_loglike:
            best_loglike = mix.loglike
            best_mix = mix

print (time.time() - start_time)
print (best_loglike)
sns.distplot(data, bins=20, kde=False, norm_hist=True)
g_both = [best_mix.pdf(e) for e in x]
plt.plot(x, g_both, label='gaussian mixture');
plt.show()
print (best_mix)
