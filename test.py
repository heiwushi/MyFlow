import myflow as mf
from optimizer import Optimizer
import numpy as np


v1 = mf.variable(shape=[2, 3], init_value=np.asarray([[1, 2, 3], [2, 2, 2]]), node_name="v1")
v2 = mf.placeholder(shape=[3, 2], node_name="v2")
v3 = mf.matmul(v1, v2, node_name="v3")
v4 = mf.reduce_sum(v3, axis=0, node_name="v4")
loss = mf.reduce_sum(v4, axis=1, node_name="loss")
train = Optimizer(learn_rate=0.01).minimize(loss)
sess = mf.Session()
for i in range(10):
    _, loss_val = sess.run([train, loss], feed_dict={v2: np.asarray([[4, 5], [6, 1], [1, 1]])})
    print(i,loss_val)