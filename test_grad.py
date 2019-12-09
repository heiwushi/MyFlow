import myflow as mf
from myflow.optimizer import Optimizer
import numpy as np

x_val=np.random.random(size=[10, 3])
y_val=np.asarray([[1],[0],[0],[1],[1],[0], [0], [0],[1],[1]])


x = mf.placeholder(shape=[10,3], tensor_name="x")
y = mf.placeholder(shape=[10,1], tensor_name="y")


v1 = mf.variable(shape=[3, 5], init_value=np.random.random([3, 5]), tensor_name="v1")
b1 = mf.variable(shape=[1, 5], init_value=np.random.random([1, 5]), tensor_name="b1")
l1 = mf.matmul(x,v1)+mf.broadcast(b1,shape=[10,5], axis=0, tensor_name="b1_broadcast")
h1 = mf.sigmoid(l1)


v2 = mf.variable(shape=[5, 1], init_value=np.random.random([5, 1]), tensor_name="v2")
b2 = mf.variable(shape=[1, 1], init_value=np.random.random([1, 1]), tensor_name="b2")
l2 = mf.matmul(h1, v2)+mf.broadcast(b2,shape=[10,1], axis=0, tensor_name="b2_broadcast")
h2 = mf.sigmoid(l2)


sess = mf.Session()
for i in range(10):
    loss_val = sess.run([h2], feed_dict={x: x_val, y: y_val})
    print(loss_val[0])
