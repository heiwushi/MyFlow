class _GradientMode(object):
    '''
    该类用于标记是否在计算梯度。当处于梯度计算过程时，每个节点的output_tensors是不变的
    该类不是线程安全的
    '''
    __gradient_mode = False
    __enter_counter = 0

    def __enter__(self):
        _GradientMode.__enter_counter += 1
        _GradientMode.__gradient_mode = True

    def __exit__(self, exc_type, exc_val, exc_tb):

        _GradientMode.__enter_counter -= 1
        if _GradientMode.__enter_counter ==0:
            _GradientMode.__gradient_mode = False

    @classmethod
    def is_gradient_mode(cls):
        return cls.__gradient_mode


