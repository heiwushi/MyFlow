import numpy as np
import abc


class Node(object):
    def __init__(self):
        self.input_nodes = []  # 该节点连接的输入节点
        self.out_nodes = []  # 该节点连接的输出节点
        self.name = ""  # 该节点的名称
        self.shape = []  # 该节点的矩阵形状
        self.op = None  # 该节点是由哪个操作OP生成的
        self.params = {}  # 存放其余参数

    def __str__(self):
        return self.name


class Op(abc.ABC):

    def __call__(self, node_name):
        '''
        调用该操作，返回一个新节点
        :return:
        '''
        new_node = Node()
        new_node.op = self
        new_node.name = self.op_name + ":" + node_name
        return new_node

    @abc.abstractmethod
    def compute(self, node: Node):
        '''
        在Session.run()期间被调用，根据输入值, 计算node节点的前向输出值
        :return:
        '''
        pass

    @abc.abstractmethod
    def gradient(self, node: Node, output_grad: Node):
        '''
        在构建图期间计算梯度，返回后续节点关于node节点的输入节点input_nodes的梯度，用于反向传播。
        注意给出的是梯度计算的拓扑图，真正的梯度值是在Session.run()期间才被计算出的
        :param node: 要计算梯度的节点。
        :param output_grad: 后续节点关于node节点的梯度
        :return:
        '''
        pass


class Variable(Op):
    def __init__(self, name=None):
        pass


class PlaceHolder(Op):
    def __init__(self):
        self.op_name = "PlaceHolder"

    def __call__(self, shape, node_name=''):
        new_node = super().__call__(node_name)
        new_node.shape = list(shape)
        return new_node

    def compute(self, node: Node, inputs_vals):
        pass

    def gradient(self, node: Node, output_grad: Node):
        pass


class Add(Op):
    '''
    矩阵逐元素相加
    '''

    def __init__(self):
        self.op_name = "Add"

    def __call__(self, input1: Node, input2: Node, node_name=''):
        assert input1.shape == input2.shape
        new_node = super().__call__(node_name)
        new_node.input_nodes = [input1, input2]
        input1.out_nodes.append(new_node)
        input2.out_nodes.append(new_node)
        new_node.shape = list(input1.shape)
        return new_node

    def compute(self, node: Node, inputs_vals):
        return inputs_vals[0] + inputs_vals[1]

    def gradient(self, node: Node, output_grad: Node):
        return [output_grad, output_grad]


class Mul(Op):
    '''
    矩阵逐元素相乘
    '''

    def __init__(self):
        self.op_name = "Mul"

    def __call__(self, input1: Node, input2: Node, node_name=''):
        assert input1.shape == input2.shape
        new_node = super().__call__(node_name)
        new_node.input_nodes = [input1, input2]
        input1.out_nodes.append(new_node)
        input2.out_nodes.append(new_node)
        new_node.shape = input1.shape[0]
        return new_node

    def compute(self, node: Node, inputs_vals):
        return np.multiply(inputs_vals[0], inputs_vals[1])

    def gradient(self, node: Node, output_grad: Node):
        return [mul(output_grad, node.input_nodes[1]), mul(output_grad, node.input_nodes[0])]


class MatMul(Op):

    def __init__(self):
        self.op_name = "MatMul"

    def __call__(self, input1: Node, input2: Node, node_name=''):
        assert input1.shape[1] == input2.shape[0]
        new_node = super().__call__(node_name)
        new_node.input_nodes = [input1, input2]
        input1.out_nodes.append(new_node)
        input2.out_nodes.append(new_node)
        new_node.shape = [input1.shape[0], input2.shape[1]]
        return new_node

    def compute(self, node: Node, input_vals):
        return np.matmul(input_vals[0], input_vals[1])

    def gradient(self, node: Node, output_grad:Node):
        return [matmul(output_grad, transpose(node.input_nodes[1])), matmul(node.input_nodes[0], output_grad)]


class Transpose(Op):

    def __init__(self):
        self.op_name = "Transpose"

    def __call__(self, input: Node, node_name=''):
        new_node = super().__call__(node_name)
        new_node.input_nodes = [input]
        input.out_nodes.append(new_node)
        new_node.shape = [input.shape[1], input.shape[0]]
        return new_node

    def compute(self, node: Node, input_vals):
        return np.transpose(input_vals[0])

    def gradient(self, node: Node, output_grad:Node):
        return [transpose(output_grad)]


class OnesLike(Op):

    def __init__(self):
        self.op_name = "OnesLike"

    def __call__(self, input: Node, node_name=''):
        new_node = super().__call__(node_name)
        new_node.input_nodes = [input]
        # input.out_nodes.append(new_node)
        new_node.shape = list(input.shape)
        return new_node

    def compute(self, node: Node, input_vals):
        return np.ones(input_vals[0].shape)

    def gradient(self, node: Node, output_grad: Node):
        return [zeroslike(node.input_nodes[0])]


class ZerosLike(Op):

    def __init__(self):
        self.op_name = "ZerosLike"

    def __call__(self, input: Node, node_name=''):
        new_node = super().__call__(node_name)
        new_node.input_nodes = [input]
        # input.out_nodes.append(new_node)
        new_node.shape = list(input.shape)
        return new_node

    def compute(self, node: Node, input_vals):
        return np.zeros(input_vals[0].shape)

    def gradient(self, node: Node, output_grad: Node):
        return [zeroslike(node.input_nodes[0])]


class ReduceSum(Op):

    def __init__(self):
        self.op_name = "ReduceSum"

    def __call__(self, input: Node, axis, node_name=''):
        new_node = super().__call__(node_name)
        new_node.input_nodes = [input]
        input.out_nodes.append(new_node)
        new_node.params["axis"] = axis
        new_shape = list(input.shape)
        new_shape[axis] = 1
        new_node.shape = new_shape
        return new_node

    def compute(self, node: Node, input_vals):
        return np.reshape(np.sum(input_vals[0], axis=node.params["axis"]), node.shape)

    def gradient(self, node: Node, output_grad: Node):
        return [broadcast(output_grad, shape=node.input_nodes[0].shape, axis=node.params["axis"])]


class Broadcast(Op):
    def __init__(self):
        self.op_name = "Broadcast"

    def __call__(self, input: Node, shape, axis, node_name=''):
        assert input.shape[axis] == 1
        new_node = super().__call__(node_name)
        new_node.input_nodes = [input]
        input.out_nodes.append(new_node)
        new_node.shape = list(shape)
        new_node.params["axis"] = axis
        return new_node

    def compute(self, node: Node, input_vals):
        return np.broadcast_to(input_vals[0], node.shape)

    def gradient(self, node: Node, output_grad: Node):
        return reduce_sum(output_grad, axis=node.params["axis"])


add = Add()
mul = Mul()
matmul = MatMul()
transpose = Transpose()
reduce_sum = ReduceSum()
broadcast = Broadcast()

placeholder = PlaceHolder()
oneslike = OnesLike()
zeroslike = ZerosLike()

gradient_dict = {}


def compute_gradient(final_node: Node, target_node: Node, name=None):
    assert final_node.shape[0] == 1 and final_node.shape[1] == 1
    gradient_dict[final_node] = oneslike(final_node)
    if gradient_dict.get(target_node):
        return gradient_dict[target_node]
    else:
        result = zeroslike(target_node)
        for output_n in target_node.out_nodes:
            output_grad = compute_gradient(final_node, output_n)
            order = output_n.input_nodes.index(target_node)
            target_node_g = output_n.op.gradient(output_n, output_grad)[order]
            result = add(result, target_node_g)
        gradient_dict[target_node] = result
        return result


class Session(object):

    def __init__(self):
        pass

    def compute_node_val(self, node: Node, feed_dict):
        if isinstance(node.op, PlaceHolder):
            return feed_dict[node]
        else:
            inputs_vals = []
            for input_n in node.input_nodes:
                inputs_vals.append(self.compute_node_val(input_n, feed_dict))
            return node.op.compute(node, inputs_vals)

    def run(self, eval_list, feed_dict=None):
        result_vals = []
        for node in eval_list:
            result_vals.append(self.compute_node_val(node, feed_dict=feed_dict))
        return result_vals


v1 = placeholder(shape=[2, 3], node_name="v1")
v2 = placeholder(shape=[2, 3], node_name="v2")
v3 = add(v1, v2, node_name="v3")
v4 = add(v1, v3, node_name="v4")
v5 = reduce_sum(v4, axis=0, node_name="v5")
v6 = reduce_sum(v5, axis=1, node_name="v6")
gradient = compute_gradient(v6, v6, "v6_v1_gradient")
sess = Session()
v4_val, v5_val, v6_val, gradient_val = sess.run([v4, v5, v6, gradient],
                                                feed_dict={v1: np.asarray([[1, 2, 3], [2, 2, 2]]),
                                                           v2: np.asarray([[4, 5, 6], [1, 1, 1]])})
print(v4_val)
print(v5_val)
print(v6_val)
print(gradient_val)
