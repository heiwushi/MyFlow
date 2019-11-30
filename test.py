import myflow as mf
import numpy as np


v1 = mf.placeholder(shape=[2, 3], node_name="v1")
v2 = mf.placeholder(shape=[3, 2], node_name="v2")
v3 = mf.matmul(v1, v2, node_name="v3")
v4 = mf.reduce_sum(v3, axis=0, node_name="v4")
v5 = mf.reduce_sum(v4, axis=1, node_name="v5")


gradient = mf.compute_gradient(final_node=v5, target_node=v1, name="v5_v1_gradient")
sess = mf.Session()
v4_val, v5_val,  gradient_val = sess.run([v4, v5, gradient],
                                                feed_dict={v1: np.asarray([[1, 2, 3], [2, 2, 2]]),
                                                           v2: np.asarray([[4, 5], [6, 1], [1, 1]])})
print(v4_val)
print(v5_val)
print(gradient_val)
