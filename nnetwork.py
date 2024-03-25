import tensorflow as tfv
import numpy as np
import random, os

tf = tfv.compat.v1
tf.compat.v1.disable_eager_execution()
sigmoid = tf.keras.activations.hard_sigmoid

P1 = 0.315
P2 = 0.815

# P1 = 0.3653
# P2 = 0.7653
# P1 = 0.4974
# P2 = 0.6974
# P1 = 0.5659
# P2 = 0.6659

# L=32,  (0.1,0.9)  (0.36585,0.76585) (0.5001,0.7001) (0.5641,0.6641)

prob = 0.01
LENGTH = 16
#num_sample = 1000
data_path = '2D_percolation/'+str(LENGTH)+'/'
files_list = sorted(os.listdir(data_path))


Format = '.npy' 
learningrate = 1e-4
Epoch = 1000
support_set1 = [round(prob*i, 2) for i in range(0, int(P1/prob))]
support_set2 = [round(prob*i, 2) for i in range(int(P2/prob), int(1/prob))]
print(support_set1)
print(support_set2)
support_set = support_set1 + support_set2

# support_point1 = support_set1[0]
#support_point2 = support_set2[-1]
support_point = [support_set1[0], support_set2[-1], 0.4, 0.64, 0.8]
#print()

def Network(net):
    net = tf.layers.flatten(net)
    net = tf.layers.Dense(units=100, activation=None)(net)
    net = tf.nn.swish(tf.layers.BatchNormalization()(net))
    net = tf.layers.Dense(units=50, activation=None)(net)
    return net

def Class(net):
    net = tf.layers.BatchNormalization()(net)
    net = tf.layers.Dense(units=50, activation=None)(net)
    net = tf.nn.swish(tf.layers.BatchNormalization()(net))
    net = tf.layers.Dense(units=1, activation=None)(net)
    net = tf.nn.sigmoid(tf.layers.BatchNormalization()(net))
    return net

config_point1 = np.load(data_path+str(support_point[0])+Format)
#config_point2 = np.load(data_path+str(support_point2)+Format)
shape = np.shape(config_point1)[1:]
#TIME = np.shape(config_point)[-1]
num_sample = np.shape(config_point1)[0]
#result = tf.math.reduce_mean(network(config_point),0)

positive = np.ones([num_sample, 1])#[[1] for i in range(num_sample)]
#tf.ones([num_sample,1])
negative = np.zeros([num_sample, 1])# [[0] for i in range(num_sample)]
#tf.zeros([num_sample,1])

Input2 = tf.placeholder(tf.float32, shape=(None,)+shape, name = 'input2')
Input1 = tf.placeholder(tf.float32, shape=(None,)+shape, name = 'input1')
label = tf.placeholder(tf.float32, shape=[None, 1], name = 'labels')


output1 = Network(Input1)
output2 = Network(Input2)
result = Class(tf.math.abs(output1-output2))
#loss = tf.reduce_mean(tf.reduce_sum((result-label)**2/0.001, -1), 0)
loss = tf.reduce_mean(- tf.reduce_sum( label*tf.log(result+1e-8) + (1-label)*tf.log(1-result+1e-8), -1), 0)

solver = tf.train.AdamOptimizer(learningrate).minimize(loss)
sess = tf.Session(config=tf.ConfigProto(
                                        log_device_placement=False))

sess.run(tf.global_variables_initializer())
sess.graph.finalize()

for epoch in range(Epoch):
    list1  = random.choice([support_set1, support_set2])
    list2  = random.choice([support_set1, support_set2])
    P_train1 = random.choice(list1)
    P_train2 = random.choice(list2)
    print('epoch:', epoch)
    print(P_train1, P_train2)
    if list2 == list1:
        label_train = positive 
        print('positive')
    else:
        label_train = negative
        print('negative')
    #P_train1 = random.choice(support_set)
    #P_train2 = random.choice(support_set)
    #if  P_train1 < P1 and  P_train2 > P2:
    #    label_train = negative
    #    print(P_train1, P_train2)
    #    print('negative')
    #    print()
    #elif  P_train1 > P2 and P_train2 < P1:
    #    label_train = negative
    #    print(P_train1, P_train2)
    #    print('negative')
    #    print()
    #else:
    #    print(P_train1, P_train2)
    #    label_train = positive 
    #    print('positive')
    #    print()

    data_P1 = np.load(data_path+str(P_train1)+Format)
    data_P2 = np.load(data_path+str(P_train2)+Format)

    np.random.shuffle(data_P1)
    np.random.shuffle(data_P2)

    feed = {label:label_train, Input1:data_P1, Input2:data_P2}
    _, loss_np = sess.run([solver, loss], feed)
    print(loss_np)
    print()

Result = []

for files in files_list:
    Final = []
    Final.append(float(files.split(".npy")[0]))
    for point in support_point:
        config_point = np.load(data_path+str(point)+Format)
        config = np.load(data_path+files)
        feed = {Input1:config, Input2:config_point}
        final = sess.run(result, feed)
    #   feed2 = {Input1:config, Input2:config_point2}
    #   final2= sess.run(result, feed2)
        # print(files.split('.npy')[0])
        Final.append(np.mean(final))
   # print([float(files.split('.npy')[0]), np.mean(final1), np.mean(final2)])
    print(Final)
    Result.append(Final)
   # Result.append([float(files.split('.npy')[0]), np.mean(final1), np.mean(final2)])
np.savetxt('2D_percolation/'+str(LENGTH)+'_'+str(P1)+'_'+str(P2)+'_7'+'.dat', Result)
# np.savetxt('2D_percolation/'+str(LENGTH)+'_'+str(P1)+'_'+str(P2) +'.dat', Result)
