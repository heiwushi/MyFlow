from myflow.ops import Tensor, PlaceHolder,Dropout

class Session(object):
    '''
    仿照tensorflow1.x, 通过Session对象执行图的运算
    '''

    def __init__(self):
        pass

    def compute_tensor_val(self, tensor: Tensor, feed_dict, cache_dict):
        '''
        计算tensor节点的实际值
        :param tensor: 要计算的节点
        :param feed_dict: place_holder节点的feed值
        :param cache_dict: 缓存，防止重复计算
        :return:
        '''

        val = cache_dict.get(tensor)
        if val is None:
            if isinstance(tensor.op, PlaceHolder):
                val = feed_dict[tensor]
            else:
                inputs_vals = []
                for input_n in tensor.input_tensors:
                    input_n_val = self.compute_tensor_val(input_n, feed_dict, cache_dict)
                    inputs_vals.append(input_n_val)
                val = tensor.op.compute(tensor, inputs_vals)
            cache_dict[tensor] = val
        return val

    def run(self, eval_list, feed_dict=None):
        '''
        外部调用run来计算eval_list中的tensor值
        :param eval_list: 要计算的tensor list
        :param feed_dict: place_holder节点的feed值
        :return:
        '''
        result_vals = []
        for tensor in eval_list:
            cache_dict=dict()
            result_vals.append(self.compute_tensor_val(tensor, feed_dict=feed_dict, cache_dict=cache_dict))
        return result_vals

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        pass