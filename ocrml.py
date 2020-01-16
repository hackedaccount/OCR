import numpy as np
import urllib.request
import gzip
import os

def load_dataset():
    def download(filename, source ="http://yann.lecun.com/exdb/mnist/"):
        print("downloading", filename)
        urllib.request.urlretrieve(source+filename,filename)


    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        f = gzip.open(filename,'r')
        f.read(16)
        buf = f.read(28 * 28 * 60000)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        f = gzip.open(filename,'r')
        data = np.frombuffer(f.read(),np.uint8,offset=8)
        return data


    x_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')


    return x_train , x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_dataset()

import matplotlib.pyplot as plt
image = np.asarray(x_train[3][0])
plt.imshow(image)
plt.show()


# print(x_train[1][0])
# temp=[]
# a=[]
# final=[]
# for b in range(59999):
#     z=784*b
#     for i in range(28):
#         for j in range(28):
#             # print(x_train[z],end='')
#             a.append(x_train[z])
#             z+=1
#         # print()
#         temp.append(a)
#         a=[]
#     final.append(temp)
#     temp=[]
# # print(temp)
# # print((x_train[:784]))
# plt.show(plt.imshow(final[0]))


import lasagne
import theano
import theano.tensor as T

def build_NN(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None,1,28,28),input_var=input_var)
    
    l_in_drop = lasagne.layers.DropoutLayer(l_in,p=0.2)

    l_hid1 = lasagne.layers.DenseLayer(l_in_drop,num_units=800,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())

    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1,p=0.5)

    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop,num_units=800,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2,p=0.5)

    l_out = lasagne.layers.DenseLayer(l_hid2_drop,num_units=10,nonlinearity=lasagne.nonlinearities.softmax)

    return l_out


input_var = T.tensor4('inputs')
target_var = T.ivector('targers')
network =   build_NN(input_var)

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction,target_var)

loss = loss.mean()


params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum = 0.9)

train_fn = theano.function([input_var, target_var],loss, updates=updates )


num_training_steps = 10

for step in range(num_training_steps):
    train_err = train_fn(x_train, y_train)
    print('current step is '+str(step))

test_prediction = lasagne.layers.get_output(network)
val_fn = theano.function([input_var],test_prediction)

print(val_fn([x_test[0]]))
print(y_test[0])

test_prediction = lasagne.layers.get_output(network,deterministic = True)
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),target_var),dtype=theano.config.floatX)

acc_fn = theano.function([input_var, target_var],test_acc)

print(acc_fn(x_test,y_test))
