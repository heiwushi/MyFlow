from myflow.ops import add,broadcast
import tensorflow as tf

tf.nn.bias_add()
def add_bias(x,bias):
    return  x + broadcast(bias, shape=[10, 1], axis=0, tensor_name="b2_broadcast")

