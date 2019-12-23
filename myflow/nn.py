from myflow.ops import add, broadcast, reshape


def add_bias(x, bias, tensor_name):
    assert len(bias.shape) == 1 and bias.shape[0] == x.shape[1]
    bias = reshape(bias, shape=[1, bias.shape[0]])
    return add(x, broadcast(bias, shape=x.shape, axis=0, tensor_name="b2_broadcast"),
               tensor_name=tensor_name)
