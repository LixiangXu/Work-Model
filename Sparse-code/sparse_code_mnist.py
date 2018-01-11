from matplotlib import pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time


# plt.close('all')
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
lamb = 0.15  # lambda
sigma = 10 * lamb  # sigma
N_basis = 28 * 28  # number of basis
N_train = 2000  # nubmer of training samples
eta_basis = 1e-3  # basis learning rate
eta_coef = 1e-2  # coef learning rate
epochs = 10  # number of epoch
rounds = 10  # number of rounds
start_time = time.time()

# generate image
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255)
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    plt.draw()
    plt.axis('off')
#==============================================================================
#     plt.pause(0.1)
#==============================================================================
    return plt


# rect func
def rect(x):
    return (x + np.abs(x))/2

# reconstruct func
def reconstruct(coef, basis_func):
    return rect(coef.dot(basis_func))


# s' function
def ds(x):
    dsx = x * np.exp(-x**2)
    return dsx


# update coefficients
def update_coef(coef, basis_func, img):
    b = np.matmul(img,basis_func.T)
    C = basis_func.dot(basis_func.T)
    delta_coef = b - C.dot(coef) - lamb/sigma * ds(coef/sigma)
#==============================================================================
#     if np.abs(delta_coef).sum()>1e6:
#         delta_coef = np.random.rand(len(b))
#==============================================================================
    coef = coef + delta_coef * eta_coef
    return coef


# update basis func
def update_basis(coef, Coef, basis_func, I):
    delta_basis =coef * ((I - reconstruct(Coef, basis_func)).mean(0))/N_basis
    basis_func = basis_func + delta_basis * eta_basis
    basis_func = basis_func / np.abs(basis_func).max()
    return basis_func


# load in data, and initialize coef and basis func
I, _ = mnist.test.next_batch(N_train)
Coef = np.random.normal(loc=0.0, scale=0.5, size=([N_train, N_basis])).astype('float64')
basis_func = np.random.normal(loc=0.0, scale=0.5, size=([N_basis, 28*28])).astype('float64')
mse = []

# show the distribution of initial coef
f1 = plt.figure()
plt.hist(np.ndarray.flatten(Coef), bins = 20, label = 'normal initial coef')

epoch = 0
err = 1
while True:
    
    if (err < 1e-2) and (epoch > epochs -1):
        break
    if epoch > 50 - 1:
        break
    train_seq = np.arange(N_train)
    np.random.shuffle(train_seq)
    for kk in train_seq:
        coef = Coef[kk]
        img = I[kk]
        for _ in range(rounds):
            coef = update_coef(coef, basis_func, img)
            basis_func = update_basis(coef, Coef, basis_func, I)
        Coef[kk] = coef
    err = (((I - reconstruct(Coef, basis_func))**2).mean(0)).mean()
    mse.append(err)
    epoch = epoch +1
    print("epoch: %d\t with MSE: %f"% (epoch, err))

# show the distribution of coef after training
plt.hist(np.ndarray.flatten(Coef), bins = 20, label = 'coef after training')
plt.title("epoch # = %d, round # = %d, train # = %d"% (epoch, rounds, N_train))
plt.legend()
plt.show(f1)
f1.savefig('sparse_code_mnist_coef_distribution.pdf')

# compare the input img and reconstruct img
f5 = plt.figure()
N_plot = np.min([N_train, 8])
if N_plot < 3/4 * N_train:
    plot_seq = np.random.randint(0, N_train, N_plot)
else:
    plot_seq = np.arange(N_plot)
    np.random.shuffle(plot_seq)
for kk in range(N_plot):
    plt.subplot(2, N_plot, kk + 1)
    gen_image(I[plot_seq[kk]])
    plt.subplot(2, N_plot, kk + 1 + N_plot)
    gen_image(reconstruct(Coef[plot_seq[kk]], basis_func))
plt.show(f5)
f5.savefig('sparse_code_mnist_result.pdf')

# show the MSE v.s. epochs
f6 = plt.figure()
plt.plot(mse)
plt.xlim(0, epoch)
plt.xlabel('epoch')
plt.ylabel('Mean Square Error')
f6.savefig('sparse_code_mnist_mse.pdf')

np.savetxt('coef.txt', Coef)
np.savetxt('basisfunc.txt', basis_func)

end_time = time.time()
print("time elapase: %d s"%  (end_time - start_time))
