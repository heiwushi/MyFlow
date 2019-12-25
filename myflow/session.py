from myflow.ops import Tensor, PlaceHolder

class Session(object):

    def __init__(self):
        pass

    def compute_tensor_val(self, tensor: Tensor, feed_dict):
        if isinstance(tensor.op, PlaceHolder):
            return feed_dict[tensor]
        else:
            inputs_vals = []
            for input_n in tensor.input_tensors:
                input_n_val = self.compute_tensor_val(input_n, feed_dict)
                inputs_vals.append(input_n_val)
            return tensor.op.compute(tensor, inputs_vals)

    def run(self, eval_list, feed_dict=None):
        result_vals = []
        for tensor in eval_list:
            result_vals.append(self.compute_tensor_val(tensor, feed_dict=feed_dict))
        return result_vals