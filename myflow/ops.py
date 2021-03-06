import numpy as np
import abc
from myflow.common import _GradientMode
from myflow.graph import Graph


class Tensor(object):
    def __init__(self):
        self.input_tensors = []  # 该节点连接的输入节点
        self.out_tensors = []  # 该节点连接的输出节点
        self.name = ""  # 该节点的名称
        self.shape = []  # 该节点的矩阵形状
        self.op = None  # 该节点是由哪个操作OP生成的
        self.params = {}  # 存放其余参数

    def __str__(self):
        return self.name

    def __neg__(self):
        return minus(self)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return add(self, -other)

    def __rsub__(self, other):
        return add(other, -self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __truediv__(self, other):
        return divide(self, other)

    def __rtruediv__(self, other):
        return divide(other, self)


class Op(abc.ABC):

    def __call__(self, tensor_name):
        '''
        调用该操作，返回一个新节点
        :return:
        '''
        new_tensor = Tensor()
        new_tensor.op = self
        new_tensor.name = self.op_name + ":" + tensor_name
        return new_tensor

    @abc.abstractmethod
    def compute(self, tensor: Tensor, input_vals):
        '''
        在Session.run()期间被调用，根据输入值, 计算tensor节点的前向输出值
        :return:
        '''
        pass

    @abc.abstractmethod
    def gradient(self, tensor: Tensor, output_grad: Tensor):
        '''
        在构建图期间计算梯度，返回后续节点关于tensor节点的输入节点input_tensors的梯度，用于反向传播。
        注意给出的是梯度计算的拓扑图，真正的梯度值是在Session.run()期间才被计算出的
        :param tensor: 要计算梯度的节点。
        :param output_grad: 后续节点关于tensor节点的梯度
        :return:
        '''
        pass


class Variable(Op):
    def __init__(self):
        self.op_name = "Variable"

    def __call__(self, init_value: np.ndarray, tensor_name=None):
        new_tensor = super().__call__(tensor_name)
        new_tensor.shape = list(init_value.shape)
        new_tensor.value = init_value
        Graph.get_default_graph().TRAIN_VARS_COLLECTIONS.append(new_tensor)
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return tensor.value

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        pass


class Constant(Op):
    def __init__(self):
        self.op_name = "Constant"

    def __call__(self, const_value: np.ndarray, tensor_name=""):
        new_tensor = super().__call__(tensor_name)
        new_tensor.shape = list(const_value.shape)
        new_tensor.value = const_value
        Graph.get_default_graph().CONSTANTS_COLLECTIONS.append(new_tensor)
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return tensor.value

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        pass


class PlaceHolder(Op):
    def __init__(self):
        self.op_name = "PlaceHolder"

    def __call__(self, shape, tensor_name=''):
        new_tensor = super().__call__(tensor_name)
        new_tensor.shape = list(shape)
        Graph.get_default_graph().PLACEHOLDER_COLLECTIONS.append(new_tensor)
        return new_tensor

    def compute(self, tensor: Tensor, inputs_vals):
        pass

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        pass


class SingleInputOp(Op):
    def __call__(self, input: Tensor, tensor_name=''):
        new_tensor = super().__call__(tensor_name)
        new_tensor.input_tensors = [input]
        if not _GradientMode.is_gradient_mode():
            input.out_tensors.append(new_tensor)
        Graph.get_default_graph().OPERATION_TENSORS_COLLECTIONS.append(new_tensor)
        return new_tensor


class DoubleInputOp(Op):
    def __call__(self, input1: Tensor, input2: Tensor, tensor_name=''):
        new_tensor = super().__call__(tensor_name)
        new_tensor.input_tensors = [input1, input2]
        if not _GradientMode.is_gradient_mode():
            input1.out_tensors.append(new_tensor)
            input2.out_tensors.append(new_tensor)
        Graph.get_default_graph().OPERATION_TENSORS_COLLECTIONS.append(new_tensor)
        return new_tensor


class Add(DoubleInputOp):
    '''
    矩阵逐元素相加
    '''

    def __init__(self):
        self.op_name = "Add"

    def __call__(self, input1: Tensor, input2: Tensor, tensor_name=''):
        input1, input2 = convert_py_numeric(input1, input2)
        assert input1.shape == input2.shape
        new_tensor = super().__call__(input1, input2, tensor_name)
        new_tensor.shape = list(input1.shape)
        return new_tensor

    def compute(self, tensor: Tensor, inputs_vals):
        return inputs_vals[0] + inputs_vals[1]

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return [output_grad, output_grad]


class Mul(DoubleInputOp):
    '''
    矩阵逐元素相乘
    '''

    def __init__(self):
        self.op_name = "Mul"

    def __call__(self, input1: Tensor, input2: Tensor, tensor_name=''):
        input1, input2 = convert_py_numeric(input1, input2)
        assert input1.shape == input2.shape
        new_tensor = super().__call__(input1, input2, tensor_name)
        new_tensor.shape = input1.shape
        return new_tensor

    def compute(self, tensor: Tensor, inputs_vals):
        return np.multiply(inputs_vals[0], inputs_vals[1])

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return [output_grad * tensor.input_tensors[1], output_grad * tensor.input_tensors[0]]


class MatMul(DoubleInputOp):

    def __init__(self):
        self.op_name = "MatMul"

    def __call__(self, input1: Tensor, input2: Tensor, tensor_name=''):
        assert input1.shape[1] == input2.shape[0]
        new_tensor = super().__call__(input1, input2, tensor_name)
        new_tensor.shape = [input1.shape[0], input2.shape[1]]
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return np.matmul(input_vals[0], input_vals[1])

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return [matmul(output_grad, transpose(tensor.input_tensors[1])),
                matmul(transpose(tensor.input_tensors[0]), output_grad)]


class Divide(DoubleInputOp):
    def __init__(self):
        self.op_name = "Divide"

    def __call__(self, input1: Tensor, input2: Tensor, tensor_name=''):
        input1, input2 = convert_py_numeric(input1, input2)
        assert input1.shape == input2.shape
        new_tensor = super().__call__(input1, input2, tensor_name)
        new_tensor.shape = list(input1.shape)
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return np.divide(input_vals[0], input_vals[1])

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return [output_grad / tensor.input_tensors[1],
                -output_grad * tensor.input_tensors[0] / (tensor.input_tensors[1] * tensor.input_tensors[1])]


class Transpose(SingleInputOp):

    def __init__(self):
        self.op_name = "Transpose"

    def __call__(self, input: Tensor, tensor_name=''):
        new_tensor = super().__call__(input, tensor_name)
        new_tensor.shape = [input.shape[1], input.shape[0]]
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return np.transpose(input_vals[0])

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return [transpose(output_grad)]


class Ones(Op):

    def __init__(self):
        self.op_name = "Ones"

    def __call__(self, shape, tensor_name=''):
        new_tensor = super().__call__(tensor_name)
        new_tensor.shape = list(shape)
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return np.ones(tensor.shape)

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return [zeros(tensor.shape)]


class Zeros(Op):

    def __init__(self):
        self.op_name = "Zeros"

    def __call__(self, shape, tensor_name=''):
        new_tensor = super().__call__(tensor_name)
        new_tensor.shape = list(shape)
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return np.zeros(tensor.shape)

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return [zeros(tensor.shape)]


class ReduceSum(SingleInputOp):

    def __init__(self):
        self.op_name = "ReduceSum"

    def __call__(self, input: Tensor, axis, tensor_name=''):
        new_tensor = super().__call__(input, tensor_name)
        new_tensor.params["axis"] = axis
        new_shape = list(input.shape)
        new_shape.pop(axis)
        new_tensor.shape = new_shape
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return np.sum(input_vals[0], axis=tensor.params["axis"])

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        # 因为reduce_sum后，axis指定的维度会被消减掉，而broadcast要求广播的维度是1
        # 所以反向链式求导时，需要把output_grad在axis上的维度恢复为1
        output_shape = list(tensor.shape)
        output_shape.insert(tensor.params["axis"], 1)
        new_output_grad = reshape(output_grad, shape=output_shape)
        return [broadcast(new_output_grad, shape=tensor.input_tensors[0].shape, axis=tensor.params["axis"])]


class Broadcast(SingleInputOp):
    def __init__(self):
        self.op_name = "Broadcast"

    def __call__(self, input: Tensor, shape, axis, tensor_name=''):
        assert input.shape[axis] == 1
        new_tensor = super().__call__(input, tensor_name)
        new_shape = list(input.shape)
        new_shape[axis] = shape[axis]
        new_tensor.shape = new_shape
        new_tensor.params["axis"] = axis
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return np.broadcast_to(input_vals[0], tensor.shape)

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        # 因为reduce_sum操作会消减掉axis维度，而broadcast操作的输入在axis上维度应该为1，所以还需要reshape一下
        return [reshape(reduce_sum(output_grad, axis=tensor.params["axis"]), tensor.input_tensors[0].shape)]


class Sigmoid(SingleInputOp):
    def __init__(self):
        self.op_name = "Sigmoid"

    def __call__(self, input: Tensor, tensor_name=''):
        new_tensor = super().__call__(input, tensor_name)
        new_tensor.shape = list(input.shape)
        Graph.get_default_graph().OPERATION_TENSORS_COLLECTIONS.append(new_tensor)
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return 1 / (1 + np.exp(-input_vals[0]))

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        x = tensor.input_tensors[0]
        return [sigmoid(x) * (1 - sigmoid(x)) * output_grad]


class Relu(SingleInputOp):
    def __init__(self):
        self.op_name = "Relu"

    def __call__(self, input: Tensor, tensor_name=''):
        new_tensor = super().__call__(input, tensor_name)
        new_tensor.shape = list(input.shape)
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return (np.abs(input_vals[0]) + input_vals[0]) / 2

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return [stepfunc(tensor.input_tensors[0]) * output_grad]


class StepFunc(SingleInputOp):

    def __init__(self):
        self.op_name = "Relu"

    def __call__(self, input: Tensor, tensor_name=''):
        new_tensor = super().__call__(input, tensor_name)
        new_tensor.shape = list(input.shape)
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return (1 + np.sign(input_vals[0])) / 2

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        pass


class Minus(SingleInputOp):
    def __init__(self):
        self.op_name = "Minus"

    def __call__(self, input: Tensor, tensor_name=''):
        new_tensor = super().__call__(input, tensor_name)
        new_tensor.shape = list(input.shape)
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return -input_vals[0]

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return [-output_grad]


class Exp(SingleInputOp):
    def __init__(self):
        self.op_name = "Exp"

    def __call__(self, input: Tensor, tensor_name=''):
        new_tensor = super().__call__(input, tensor_name)
        new_tensor.shape = list(input.shape)
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return np.exp(input_vals[0])

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return [exp(tensor.input_tensors[0]) * output_grad]


class Log(SingleInputOp):
    def __init__(self):
        self.op_name = "Log"

    def __call__(self, input: Tensor, tensor_name=''):
        new_tensor = super().__call__(input, tensor_name)
        new_tensor.shape = list(input.shape)
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return np.log(input_vals[0])

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return [output_grad / tensor.input_tensors[0]]


class Reshape(SingleInputOp):
    def __init__(self):
        self.op_name = "Reshape"

    def __call__(self, input: Tensor, shape, tensor_name=''):
        new_tensor = super().__call__(input, tensor_name)
        new_tensor.shape = list(shape)
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        return np.reshape(input_vals[0], tensor.shape)

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return [reshape(output_grad, tensor.input_tensors[0].shape)]


class Dropout(SingleInputOp):
    def __init__(self):
        self.op_name = "Dropout"

    def __call__(self, input: Tensor, keep_prob, tensor_name=''):
        new_tensor = super().__call__(input, tensor_name)
        new_tensor.shape = list(input.shape)
        new_tensor.params["keep_prob"] = keep_prob
        return new_tensor

    def compute(self, tensor: Tensor, input_vals):
        print("dropout:", tensor)
        return input_vals[0] * np.random.binomial(1, tensor.params["keep_prob"], size=tensor.shape)

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return [dropout(output_grad, tensor.params["keep_prob"])]


def convert_py_numeric(x, y):
    '''
    将python中的字面常量数值转化为constant的tensor节点
    :param x:
    :param y:
    :return:
    '''
    assert type(x) in [int, float, Tensor]
    assert type(y) in [int, float, Tensor]
    if isinstance(x, int) or isinstance(x, float):
        x = constant(const_value=x * np.ones(shape=y.shape))
    if isinstance(y, int) or isinstance(y, float):
        y = constant(const_value=y * np.ones(shape=x.shape))
    return x, y


add = Add()
minus = Minus()
mul = Mul()
matmul = MatMul()
divide = Divide()
transpose = Transpose()
reduce_sum = ReduceSum()
broadcast = Broadcast()
exp = Exp()
log = Log()
sigmoid = Sigmoid()
relu = Relu()
stepfunc = StepFunc()
reshape = Reshape()
dropout = Dropout()
ones = Ones()
zeros = Zeros()

placeholder = PlaceHolder()
variable = Variable()
constant = Constant()

