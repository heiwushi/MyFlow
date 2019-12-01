from myflow import Op,Node,_GradientMode,add,oneslike, zeroslike,TRAIN_VARS_COLLECTIONS
import numpy as np
class Optimizer(object):
    class ApplyGradient(Op):

        def __init__(self, learn_rate=0.001):
            self.op_name="ApplyGradient"
            self.learn_rate = learn_rate

        def __call__(self, vars_gradients: dict, node_name=""):
            new_node = super().__call__(node_name)
            vars = list(vars_gradients.keys())
            gradients = [vars_gradients[var] for var in vars]
            new_node.input_nodes.extend(vars)
            new_node.input_nodes.extend(gradients)
            return new_node

        def compute(self, node, input_vals):
            var_vals = input_vals[0:int(len(input_vals)/2)]
            gradient_vals = input_vals[int(len(input_vals)/2):]
            for i, var in enumerate(node.input_nodes[0:int(len(input_vals)/2)]):
                var.value = np.add(var_vals[i], -self.learn_rate*gradient_vals[i])

        def gradient(self):
            pass

    def __init__(self, learn_rate=0.001):
        self.learn_rate = learn_rate
        self.apply_gradient = Optimizer.ApplyGradient(learn_rate)

    def compute_gradient(self, loss: Node, var_list, name=None):
        '''
        给出因变量loss关于自变量var的梯度的计算图
        :param loss: 所求梯度的因变量
        :param var: 所求梯度的自变量
        :param name:节点名
        :return:
        '''
        # 考虑简单的情况，这里要求必须是[1,1]形状，即标量。实际上机器学习中的损失loss一般都是标量。
        assert loss.shape[0] == 1 and loss.shape[1] == 1
        vars_gradients={}
        with _GradientMode():
            for var in var_list:
                # 如果求的是关于自身的梯度，则直接返回一个形状与var一样、元素全为1的矩阵
                if var == loss:
                    result=oneslike(var)
                else:
                    # 根据多元复合函数求导法则，loss关于var的导数，应该为loss对var的所有输出节点所在路径分别求导，之后求和
                    result = zeroslike(var)
                    for output_n in var.out_nodes:
                        # 对于每条输出路径，先对输出节点求导
                        output_grad = self.compute_gradient(loss, [output_n])[output_n]
                        # 之后根据该节点的操作的gradient函数，计算该条路径对var的导数
                        order = output_n.input_nodes.index(var)
                        var_g = output_n.op.gradient(output_n, output_grad)[order]
                        # 与之前各条路径的结果累加
                        result = add(result, var_g)
                vars_gradients[var]=result
            return vars_gradients

    def minimize(self, loss, var_list=TRAIN_VARS_COLLECTIONS):
        assert loss.shape[0]==1 and loss.shape[1]==1
        vars_gradients = self.compute_gradient(loss,  var_list)
        return self.apply_gradient(vars_gradients)