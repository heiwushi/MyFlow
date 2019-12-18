import myflow as mf

import numpy as np

hidden_size = 50
batch_size = 10

x_val = np.random.random(size=[batch_size, 3])
y_val = np.asarray([[1], [0], [0], [1], [1], [0], [0], [0], [1], [1]])

x = mf.placeholder(shape=[batch_size, 3], tensor_name="x")
y = mf.placeholder(shape=[batch_size, 1], tensor_name="y")

v1 = mf.variable(init_value=1 - 2 * np.random.random([3, hidden_size]), tensor_name="v1")
b1 = mf.variable(init_value=1 - 2 * np.random.random([1, hidden_size]), tensor_name="b1")
l1 = mf.matmul(x, v1) + mf.broadcast(b1, shape=[batch_size, hidden_size], axis=0, tensor_name="b1_broadcast")
h1 = mf.relu(l1)

v2 = mf.variable(init_value=1 - 2 * np.random.random([hidden_size, 1]), tensor_name="v2")
b2 = mf.variable(init_value=1 - 2 * np.random.random([1, 1]), tensor_name="b2")
l2 = mf.matmul(h1, v2) + mf.broadcast(b2, shape=[10, 1], axis=0, tensor_name="b2_broadcast")
y_pred = mf.sigmoid(l2)

loss = mf.losses.binary_cross_entropy(y, y_pred)
train_step = mf.optimizer.Optimizer(learn_rate=0.01).minimize(loss)

sess = mf.Session()
for i in range(2000):
    _, loss_val, y_pred_val = sess.run([train_step, loss, y_pred], feed_dict={x: x_val, y: y_val})
    print(i, loss_val, y_pred_val)
