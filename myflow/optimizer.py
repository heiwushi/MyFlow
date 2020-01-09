import abc
import numpy as np
import functools

from myflow.ops import Tensor, Op, ones, zeros, add
from myflow.common import _GradientMode
from myflow.graph import Graph


class ApplyGradient(Op):
    '''
    用计算好的梯度更新Variable
    '''

    def __init__(self, compute_var_delta):
        '''
        :param compute_var_delta: 根据梯度计算Variable的更新值，由不同的优化器来做具体实现
        '''
        self.op_name = "ApplyGradient"
        self.compute_var_delta = compute_var_delta

    def __call__(self, vars_gradients: dict, tensor_name=""):
        '''
        将应用梯度这一操作作为一个计算图tensor节点返回，
        该节点的输入节点为要训练的若干个Variable及其对应的梯度
        :param vars_gradients:一个字典，key为要训练的Variable节点，value为Variable节点的梯度
        :param tensor_name:
        :return:
        '''
        new_tensor = super().__call__(tensor_name)
        vars = list(vars_gradients.keys())
        gradients = [vars_gradients[var] for var in vars]
        new_tensor.input_tensors.extend(vars)
        new_tensor.input_tensors.extend(gradients)
        return new_tensor

    def compute(self, tensor, input_vals):
        '''
        在Session.run()期间调用，对Variable节点的实际值进行更新
        :param tensor:
        :param input_vals:
        :return:
        '''
        var_vals = input_vals[0:int(len(input_vals) / 2)]
        gradient_vals = input_vals[int(len(input_vals) / 2):]
        var_delta = self.compute_var_delta(gradient_vals)
        for i, var in enumerate(tensor.input_tensors[0:int(len(input_vals) / 2)]):
            var.value = np.add(var_vals[i], var_delta[i])

    def gradient(self, tensor: Tensor, output_grad: Tensor):
        return None


class Optimizer(abc.ABC):
    '''
    所有优化器的基类
    优化器实现了三个图操作：
    compute_gradient：给出梯度的计算图节点
    apply_gradient：给出梯度更新的计算图节点
    minimize:等价于先调用compute_gradient，再调用apply_gradient
    所有派生于Optimizer的子类应该实现compute_var_delta方法。该方法决定不同优化器在计算时，如何根据梯度算出Variable节点的变化值
    '''

    def __init__(self):
        self.apply_gradient = ApplyGradient(self.compute_var_delta)

    @abc.abstractmethod
    def compute_var_delta(self, grad_vals_list):
        '''
        所有派生于Optimizer的子类应该实现compute_var_delta方法。该方法决定不同优化器在计算时，如何根据梯度算出Variable节点的变化值
        :param grad_vals_list: 计算出的梯度值list
        :return:
        '''
        pass

    def compute_gradient(self, loss: Tensor, var_list=None, cache_dict: dict = None, name=None):
        '''
        给出因变量loss关于自变量var_list的梯度的计算图
        :param loss: 所求梯度的因变量节点
        :param var_list: 所求梯度的自变量节点list
        :param name:节点名
        :return:
        '''
        # 这里要求必须是[]形状，即标量。实际上机器学习中的损失loss一般都是标量。
        assert loss.shape == []
        if var_list is None:
            var_list = Graph.get_default_graph().TRAIN_VARS_COLLECTIONS
        if cache_dict is None:
            cache_dict = dict()
        vars_gradients = {}
        with _GradientMode():
            for var in var_list:
                result = cache_dict.get(var)
                if result is None:
                    # 如果求的是关于自身的梯度，则直接返回一个形状与var一样、元素全为1的矩阵
                    if var == loss:
                        result = ones(var.shape)
                    else:
                        # 根据多元复合函数求导法则，loss关于var的导数，应该为loss对var的所有输出节点所在路径分别求导，之后求和
                        result = zeros(var.shape)
                        for output_n in var.out_tensors:
                            # 对于每条输出路径，先对输出节点求导
                            output_grad = self.compute_gradient(loss, [output_n], cache_dict)[output_n]
                            # 之后根据该节点的操作的gradient函数，计算该条路径对var的导数
                            order = output_n.input_tensors.index(var)
                            var_g = output_n.op.gradient(output_n, output_grad)[order]
                            # 与之前各条路径的结果累加
                            result = add(result, var_g)
                    cache_dict[var] = result
                vars_gradients[var] = result
            return vars_gradients

    def minimize(self, loss, var_list=None):
        '''
        :param loss:所求梯度的因变量节点
        :param var_list:所求梯度的自变量节点list
        :return:
        '''
        assert loss.shape == []
        vars_gradients = self.compute_gradient(loss, var_list)
        return self.apply_gradient(vars_gradients)


class SGD(Optimizer):
    '''
    基本的随机梯度下降算法
    '''

    def __init__(self, learn_rate=0.001):
        '''
        :param learn_rate: 学习率
        '''
        super().__init__()
        self.learn_rate = learn_rate

    def compute_var_delta(self, grad_vals_list):
        for i in range(len(grad_vals_list)):
            grad_vals_list[i] = -self.learn_rate * grad_vals_list[i]
        return grad_vals_list


class RMSProp(Optimizer):
    '''
    Root Mean Square Prop优化算法
    动态调整每个参数的学习率，使其反比于该参数的历史偏导数平方值总和的平方根。
    而不同于AdaGrad的是,历史偏导数平方值的总和会以decay的速率衰减
    '''
    def __init__(self, learn_rate=0.001, decay=0.9):
        '''
        :param learn_rate: 初始学习率
        :param decay: 衰减率
        '''
        super().__init__()
        self.learn_rate = learn_rate
        self.decay = decay
        self.r = 0

    def compute_var_delta(self, grad_vals_list):
        grad_vals_shape_list = list(map(lambda x: x.shape, grad_vals_list))
        grad_vals_flatten_list = list(map(lambda x: x.flatten(), grad_vals_list))
        grad_vals_concat = np.concatenate(grad_vals_flatten_list)
        self.r = self.r * self.decay + (1 - self.decay) * grad_vals_concat * grad_vals_concat
        var_delta_concat = - self.learn_rate * (1 / np.sqrt(self.r + 1e-8)) * grad_vals_concat
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
