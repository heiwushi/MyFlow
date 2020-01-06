from myflow.ops import Tensor, PlaceHolder

class Session(object):

    def __init__(self):
        pass

    def compute_tensor_val(self, tensor: Tensor, feed_dict):
        if isinstance(tensor.op, PlaceHolder):
            val = feed_dict[tensor]
        else:
            inputs_vals = []
            for input_n in tensor.input_tensors:
                input_n_val = self.compute_tensor_val(input_n, feed_dict)
                inputs_vals.append(input_n_val)
            val = tensor.op.compute(tensor, inputs_vals)
        return val

    def run(self, eval_list, feed_dict=None):
        result_vals = []
        for tensor in eval_list:
            result_vals.append(self.compute_tensor_val(tensor, feed_dict=feed_dict))
        return result_vals

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        pass