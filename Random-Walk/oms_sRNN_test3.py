# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:41:55 2018

@author: Lixiang
"""


import numpy as np
from matplotlib import pylab as plt
import random
import time
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
from IPython import display
from matplotlib import animation
import webbrowser
import os

# Global config 
num_bp = 5 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 10
input_size = 3
output_size = 1
state_size = 8
h1_size = 2

ser_size = 10
num_mov = 6
max_len_mov = 4
Epochs = 400
per_epoch = 50


#generate data
def gen_data(size=600):
    X = np.ones(shape=(1,size))
    Y = np.zeros(shape=(1,size))
    n = int(num_mov/100*size)
    idx = 0
    for _ in range(n):
        d_idx = np.random.poisson(lam=10)
        len_mov = np.random.poisson(lam=max_len_mov)
        idx += d_idx
        if np.random.rand()>0.5:
            X[0,idx:idx+len_mov] = 2
        else:
            X[0,idx:idx+len_mov] = 0
        idx += len_mov
        if idx>size:
            break
    for idx in range(size):
        Y[0,idx] = Y[0,idx-1] + X[0,idx] - 1
    return X,Y


size = 100
X_, Y_ = gen_data(size)
f1 = plt.figure()
plt.subplot(2,1,1)
plt.plot(list(range(size)), X_.T)
plt.subplot(2,1,2)
plt.plot(list(range(size)), Y_.T)
plt.show()



def genBatch(data, batch_size, num_bp):
    data_x, data_y = data
    length = len(data_y[0])
    data_x_batch = np.reshape(data_x[0], [batch_size,-1])
    data_y_batch = np.reshape(data_y[0], [batch_size, -1])
    data_x_step = []
    data_y_step = []
    for ii in range(int(length/num_bp/batch_size)):
        data_x_step.append(data_x_batch[:,ii*num_bp:(ii+1)*num_bp])
        data_y_step.append(data_y_batch[:,ii*num_bp:(ii+1)*num_bp])
    #for ii in range(int(length/batch_size)-num_bp):
     #   data_x_step.append(data_x_batch[:,ii:ii+num_bp])
      #  data_y_step.append(data_y_batch[:,ii:ii+num_bp])
    x,y = list(data_x_step), list(data_y_step)
    
    return x,y
#data = gen_data() x,y = genBatch(data, batch_size, num_bp)

#print(len(x[0]))

#print(y)



trainX = list()
trainY = list()
for i in range(num_bp):
    trainX.append(tf.placeholder(tf.float32, shape=(batch_size, input_size)))
    
    trainY.append(tf.placeholder(tf.float32, shape=(batch_size, output_size)))
state_first = tf.placeholder(tf.float32, shape=(batch_size, state_size), name='1st_state')
out_first = tf.placeholder(tf.float32, shape=(batch_size, output_size), name='1st_output')

#RNN layer variables
U = tf.Variable(tf.truncated_normal([input_size+state_size, state_size], -0.1, 0.1))
V = tf.Variable(tf.truncated_normal([input_size+state_size+output_size, state_size], -0.1, 0.1))
b = tf.Variable(tf.zeros([1, state_size]))


W1 = tf.Variable(tf.truncated_normal([state_size, h1_size], -0.1, 0.1))
b1 = tf.Variable(tf.zeros([batch_size, h1_size]))
#b1_ = tf.Variable(tf.zeros([1, h1_size]))
#Ib1_ = tf.constant(1, tf.float32, shape=[batch_size, 1])
#b1 = tf.matmul(Ib1_, b1_)


Wo = tf.Variable(tf.truncated_normal([h1_size, output_size], -0.1, 0.1))
bo = tf.Variable(tf.zeros([batch_size, output_size]))
#bo_ = tf.Variable(tf.zeros([1, output_size]))
#Ibo_ = tf.constant(1, tf.float32, shape=[batch_size, 1])
#bo = tf.matmul(Ibo_, bo_)

#model
def RNN(x, state_pre, out_pre):
    #a = tf.concat([tf.matmul(x, U), tf.matmul(state_pre, W)], axis=1) + b
    #a = tf.matmul(tf.concat([x, state_pre], axis=1), U) + b
    a = tf.matmul(tf.concat([x, state_pre, out_pre], axis=1), V) + b
    state = tf.nn.softmax(a)
    

    #o_out = tf.matmul(state, V) + c

    
    
    h1 = tf.matmul(state, W1) + b1
    o_out = tf.matmul(tf.nn.softmax(h1), Wo) + bo
    
    
    
    return state, o_out


for i in range(num_bp):
    if i == 0:
        outputs = list()
        state_after, output_after = RNN(trainX[i], state_first, out_first)
    else:
        state_after, output_after = RNN(trainX[i], state_pre, out_pre)
    state_pre = state_after
    out_pre = output_after
    outputs.append(output_after)


#train

#log likelihood loss
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.concat(outputs,0),labels=tf.concat(trainY,0)))
loss = tf.losses.mean_squared_error(labels=tf.concat(trainY,0), predictions = tf.concat(outputs,0))
#learning_rate = tf.placeholder(tf.float32, shape=[])

#optimizer
global_epoch = tf.Variable(0)
global_sets = tf.Variable(0)

learning_rate = tf.train.exponential_decay(
    learning_rate=0.5, global_step=global_epoch, decay_steps=5000, decay_rate=0.9, staircase=True)
#learning_rate = 1.0

#optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

gradients,var=zip(*optimizer.compute_gradients(loss))
gradients_clipped, _ = tf.clip_by_global_norm(gradients, 0.2)
opt=optimizer.apply_gradients(zip(gradients_clipped,var),global_step=global_epoch)






#add init op to the graph
sess = tf.Session()
saver = tf.train.Saver()
cwd = os.getcwd()
train_filename = "oms_sRNN_statesize"+str(state_size)+"_h1size_"+str(h1_size)+".ckpt"

if os.path.isfile(train_filename+".index"):
    saver.restore(sess, train_filename)
    print("Model restored from: %s"%(train_filename))
    print('Global sets: %s'%(global_sets.eval(sess)))
else:
    init = tf.global_variables_initializer()
    sess.run(init)
    print("Initialize all Variables")







for ser in range(ser_size):
    size = 200
    data = gen_data(size=size)
    batch_x, batch_y = genBatch(data, batch_size, num_bp)
    steps = len(batch_y)
    average_loss_series = list()
    outputs_series = list()
    plot_y_series = list()
    global_epoch = tf.Variable(0)
    pbar = tqdm(total=Epochs)
    for epoch in range(Epochs):
        if epoch%per_epoch == 0:
            plot_y = np.zeros([batch_size, int(size/batch_size)])
        for step in range(steps):

            average_loss = list()
            #initialize the output
            if step == 0:
                if epoch == 0:
                    output_pass = np.zeros([batch_size, state_size], dtype=np.float32)
                    out_pass = np.zeros([batch_size, output_size], dtype=np.float32)
                else:
                    output_pass = np.zeros([batch_size, state_size], dtype=np.float32)
                    output_pass[1:batch_size,:] = np.asarray(state_after_[0:batch_size-1][:])
                    out_pass = np.zeros([batch_size, output_size], dtype=np.float32)
                    out_pass[1:batch_size,:] = np.asarray(output_after_[0:batch_size-1][:])
            else:
                output_pass = state_after_
                out_pass = output_after_
            feed_dict={state_first: output_pass, out_first: out_pass}


            #trains x
            for i in range(num_bp):
                batch_x_i = batch_x[step][:,i].astype(int)
                batch_x_i_onehot = np.zeros((len(batch_x_i), input_size))
                batch_x_i_onehot[range(len(batch_x_i)), batch_x_i] = 1
                feed_dict[trainX[i]] = batch_x_i_onehot

                batch_y_i = np.reshape(batch_y[step][:,i],[batch_size,1])
                feed_dict[trainY[i]] = batch_y_i

            #feed_dict[learning_rate] = 0.1

            state_after_, l ,outputs_, output_after_, opt_ = sess.run([state_after, loss, outputs, output_after, opt], feed_dict=feed_dict)
            outputs_series.append(outputs_)
            average_loss.append(l)
            ave_loss = sum(average_loss)/float(len(average_loss))

            if epoch%per_epoch == 0:
                plot_y[:, step*num_bp:(step+1)*num_bp] = np.asarray(outputs_).flatten().reshape([-1,batch_size]).T


        pbar.update(1)
        if epoch%per_epoch == 0:        
            plot_y_series.append(plot_y.reshape([-1]))
            display.clear_output(wait=True)
            f2 = plt.figure()
            plt.plot(data[1][0],label='Goal')
            plt.plot(plot_y.reshape([-1]),label='Preiction')
            plt.legend()
            display.display(plt.gcf())
            display.display(pbar)
            print ('Serial #: ' +str(ser) +'Epoch: ' +str(epoch) + ' Average Loss: '+str(ave_loss))
            time.sleep(0.01)

        average_loss_series.append(ave_loss)
        global_epoch += 1
        if ave_loss<0.8:
            break
    pbar.close()
    global_epoch = 0
    global_sets +=1
    save_train_path = saver.save(sess, cwd+"/"+train_filename)
    print("Model saved in file: %s" % save_train_path)
print('completed!')



#generate anim to show simulation results

#%%
f3 = plt.figure()
ax = plt.axes(xlim=(0,size), ylim=(min(data[1][0])-3, max(data[1][0])+3))
line1, = ax.plot([], [], lw=2)
line2, = ax.plot([],[], lw=2)

def init():
    line1.set_data([],[])
    line2.set_data([],[])
    return line1,line2,

def animate(i):
    x = np.linspace(0,size,size)
    y1 = data[1][0]
    y2 = plot_y_series[i]
    line1.set_data(x, y1)
    line1.set_label('Goal')
    line2.set_data(x, y2)
    line2.set_label('Prediction /w epochs= %d'%(i*per_epoch))
    plt.legend()
    return line1, line2,

anim = animation.FuncAnimation(f3, animate, init_func=init, frames=len(plot_y_series), interval=1, blit=True)

filename = 'random_walk_size_'+str(size)+'_Epochs_'+str(per_epoch*len(plot_y_series))+'.html'

anim.save(filename, fps=2, extra_args=['-vcodec', 'libx264'])


#open and run anim on chrome
url = str(filename)
chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
webbrowser.get(chrome_path).open(url)


#%%
# generate eval data
x_eva, y_eva = gen_data(size=200)
batch_x_eva = x_eva.reshape([batch_size,-1])


outputs_eva_series = list()

steps = len(batch_x_eva[0])
plot_y_eva = np.zeros([batch_size, steps])

evaX = tf.placeholder(tf.float32, shape=(batch_size, input_size), name='evaX')
state_first_eva = tf.placeholder(tf.float32, shape=(batch_size, state_size), name='1st_state_eva')
out_first_eva = tf.placeholder(tf.float32, shape=(batch_size, output_size), name='1st_output_eva')

# eval network
for i in range(steps):
    if i == 0:
        outputs_eva = list()
        state_after_eva, output_after_eva = RNN(evaX, state_first_eva, out_first_eva)
    else:
        state_after_eva, output_after_eva = RNN(evaX, state_pre_eva, out_pre_eva)
    state_pre_eva = state_after_eva
    out_pre_eva = output_after_eva
    outputs.append(output_after_eva)
    
# eval by steps
for step in range(steps):
    average_loss_eva = list()
    #initialize the output
    if step == 0:
        output_pass_eva = np.zeros([batch_size, state_size], dtype=np.float32)
        out_pass_eva = np.zeros([batch_size, output_size], dtype=np.float32)
    else:
        output_pass_eva = state_after_eva_
        out_pass_eva = outputs_eva_
    
    step_x_eva = batch_x_eva[:,step].astype(int)
    x_eva_onehot = np.zeros((batch_size, input_size))
    x_eva_onehot[range(batch_size), step_x_eva] = 1
    feed_dict_eva={state_first_eva: output_pass_eva, out_first_eva: out_pass_eva, evaX:x_eva_onehot}

    state_after_eva_,outputs_eva_ = sess.run([state_after_eva, output_after_eva], feed_dict=feed_dict_eva)
    outputs_eva_series.append(outputs_eva_)

    plot_y_eva[:, step] = np.asarray(outputs_eva_).flatten().reshape([batch_size]).T 
    
plt.show()
display.clear_output(wait=True)

f4 = plt.figure()
plt.subplot(2,1,1)
plt.plot(data[1][0],label='train Goal')
plt.plot(plot_y.reshape([-1]), label='train Prediction')
plt.legend()
plt.subplot(2,1,2)
plt.plot(y_eva[0],label='eval Goal')
plt.plot(plot_y_eva.reshape([-1]),label='eval Preiction')
plt.legend()
plt.show(f4)
#ave_loss_eva = 
#print(ave_loss_eva)
