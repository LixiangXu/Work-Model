from matplotlib import pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# plt.close('all')
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
lamb = 0.5  # lambda
sigma = 0.1  # sigma
N_basis = 12*12  # number of basis
N_train = 20  # nubmer of training samples
eta_basis = 1e-3  # basis learning rate
eta_coef = 1e-3  # coef learning rate
epochs = 100  # number of epoch
rounds = 100  # number of rounds


# generate image
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255)
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    plt.draw()
    # plt.pause(0.1)
    return plt


# reconstruct func
def reconstruct(coef, basis_func):
    return coef.dot(basis_func)


# s' function
def ds(x):
    dsx = np.zeros([N_basis])
    for kk in range(N_basis):
        dsx[kk] = 2*x[kk]/(1+x[kk]**2)
    return dsx


# update coefficients
def update_coef(coef, basis_func, img):
    b = np.matmul(basis_func, img)
    C = basis_func.dot(basis_func.T)
    delta_coef = b - C.dot(coef) - lamb/sigma * ds(coef/sigma)
    coef = coef + delta_coef * eta_coef
    return coef


# update basis func
def update_basis(coef, basis_func, img):
    delta_basis = (img - reconstruct(coef, basis_func)).mean()
    basis_func = basis_func + delta_basis * eta_basis
    return basis_func


# load in data, and initialize coef and basis func
I, _ = mnist.test.next_batch(N_train)
Coef = np.random.normal(loc=0.0, scale=1.0, size=([N_train, N_basis]))
basis_func = np.random.normal(loc=0.0, scale=1.0, size=([N_basis, 28*28]))

for _ in range(epochs):
    for kk in range(N_train):
        coef = Coef[kk]
        img = I[kk]
        for _ in range(rounds):
            coef = update_coef(coef, basis_func, img)
            basis_func = update_basis(coef, basis_func, img)

plt.figure()
f1 = gen_image(reconstruct(coef, basis_func))
plt.show(f1)
plt.figure()
f2 = plt.hist(coef)
plt.show(f2)
