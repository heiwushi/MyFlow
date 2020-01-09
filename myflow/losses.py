from myflow.ops import Tensor,log, reduce_sum
def binary_cross_entropy(y_true:Tensor, y_pred:Tensor):
    '''
    交叉熵损失
    :param y_true: shape=[batch_size, class_num]
    :param y_prob: shape=[batch_size, class_num]
    :return:
    '''
    return -reduce_sum(reduce_sum(y_true * log(y_pred+1e-10) + (1-y_true) * log(1-y_pred+1e-10), axis=1), axis=0)