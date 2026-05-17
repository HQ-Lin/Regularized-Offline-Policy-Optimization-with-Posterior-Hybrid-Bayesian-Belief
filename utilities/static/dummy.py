import numpy as np

class StaticFns:
    """
    为固定周期的环境 (如 HIV, Currency) 提供静态函数。
    """
    @staticmethod
    def _dummy_done_func(state, action, next_state):
        """
        对于固定周期的环境, 终止由 rollout 长度控制。
        此函数假定在 rollout 过程中的任何状态都不是终止状态。
        """
        return np.zeros((state.shape[0], 1), dtype=bool)

    # 将函数赋值给类变量 termination_fn 以便统一调用
    termination_fn = _dummy_done_func