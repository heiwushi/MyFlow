import abc
import numpy as np
import functools

from myflow.ops import Tensor, Op, oneslike, zeroslike, add
from myflow.common import _GradientMode
from myflow.graph import Graph


class ApplyGradient(Op):

    def __init__(self, compute_var_delta):
        self.op_name = "ApplyGradient"
        self.compute_var_delta = compute_var_delta

    def __call__(self, vars_gradients: dict, tensor_name=""):
        new_tensor = super().__call__(tensor_name)
        vars = list(vars_gradients.keys())
        gradients = [vars_gradients[var] for var in vars]
        new_tensor.input_tensors.extend(vars)
        new_tensor.input_tensors.extend(gradients)
        return new_tensor

    def compute(self, tensor, input_vals):
        var_vals = input_vals[0:int(len(input_vals) / 2)]
        gradient_vals = input_vals[int(len(input_vals) / 2):]
        var_delta = self.compute_var_delta(gradient_vals)
        for i, var in enumerate(tensor.input_tensors[0:int(len(input_vals) / 2)]):
            var.value = np.add(var_vals[i], var_delta[i])

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return None


class Optimizer(abc.ABC):
    def __init__(self):
        self.apply_gradient = ApplyGradient(self.compute_var_delta)

    @abc.abstractmethod
    def compute_var_delta(self, grad_list):
        pass

    def compute_gradient(self, loss: Tensor, var_list=None, name=None):
        '''
        给出因变量loss关于自变量var的梯度的计算图
        :param loss: 所求梯度的因变量
        :param var: 所求梯度的自变量
        :param name:节点名
        :return:
        '''
        # 考虑简单的情况，这里要求必须是[]形状，即标量。实际上机器学习中的损失loss一般都是标量。
        assert loss.shape == []
        if var_list is None:
            var_list = Graph.get_default_graph().TRAIN_VARS_COLLECTIONS
        vars_gradients = {}
        with _GradientMode():
            for var in var_list:
                # 如果求的是关于自身的梯度，则直接返回一个形状与var一样、元素全为1的矩阵
                if var == loss:
                    result = oneslike(var)
                else:
                    # 根据多元复合函数求导法则，loss关于var的导数，应该为loss对var的所有输出节点所在路径分别求导，之后求和
                    result = zeroslike(var)
                    for output_n in var.out_tensors:
                        # 对于每条输出路径，先对输出节点求导
                        output_grad = self.compute_gradient(loss, [output_n])[output_n]
                        # 之后根据该节点的操作的gradient函数，计算该条路径对var的导数
                        order = output_n.input_tensors.index(var)
                        var_g = output_n.op.gradient(output_n, output_grad)[order]
                        # 与之前各条路径的结果累加
                        result = add(result, var_g)
                vars_gradients[var] = result
            return vars_gradients

    def minimize(self, loss, var_list=None):
        assert loss.shape == []
        vars_gradients = self.compute_gradient(loss, var_list)
        return self.apply_gradient(vars_gradients)


class SGD(Optimizer):

    def __init__(self, learn_rate=0.001):
        super().__init__()
        self.learn_rate = learn_rate

    def compute_var_delta(self, grad_vals_list):
        for i in range(len(grad_vals_list)):
            grad_vals_list[i] = -self.learn_rate * grad_vals_list[i]
        return grad_vals_list


class RMSProp(Optimizer):

    def __init__(self, learn_rate=0.001, decay=0.9):
        super().__init__()
        self.learn_rate = learn_rate
        self.decay = decay
        self.r = 0

    def compute_var_delta(self, grad_vals_list):
        grad_vals_shape_list = list(map(lambda x: x.shape, grad_vals_list))
        grad_vals_flatten_list = list(map(lambda x: x.flatten(), grad_vals_list))
        grad_vals_concat = np.concatenate(grad_vals_flatten_list)
        self.r = self.r * self.decay + (1 - self.decay) * grad_vals_concat * grad_vals_concat
        var_delta_concat = - self.learn_rate * (1 / np.sqrt(self.r + 1e-7)) * grad_vals_concat
        var_delta = []
        tmp_pos = 0
        for i in range(len(grad_vals_list)):
            shape = grad_vals_shape_list[i]
            length = functools.reduce(lambda x, y: x * y, shape)
            var_delta.append(np.reshape(var_delta_concat[tmp_pos:tmp_pos + length], shape))
            tmp_pos += length
        return var_delta


class Adam(Optimizer):

    def __init__(self):
        super().__init__()
        # TODO


    def compute_var_delta(self, grad_vals_list):
        # TODO
        pass
