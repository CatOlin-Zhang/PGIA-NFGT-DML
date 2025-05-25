import torch
from torch import Tensor
from typing import List, Optional
import torch.nn as nn
from sklearn.utils.extmath import randomized_svd
# 定义 required 对象
required = object()

__all__ = ['SGD', 'sgd']


class SGD:
    def __init__(self, params, lr=required, momentum=0, dampening=0,weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None,differentiable=False):
        if lr is required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,weight_decay=weight_decay, nesterov=nesterov,maximize=maximize, foreach=foreach,differentiable=differentiable)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.param_groups = [{'params': list(params), **defaults}]
        self.state = {}

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = {}

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,d_p_list,momentum_buffer_list,weight_decay=group['weight_decay'],momentum=group['momentum'],lr=group['lr'],dampening=group['dampening'],nesterov=group['nesterov'],maximize=group['maximize'],has_sparse_grad=has_sparse_grad,foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def sgd(params: List[Tensor],d_p_list: List[Tensor],momentum_buffer_list: List[Optional[Tensor]], has_sparse_grad: bool = None,foreach: bool = None, *,weight_decay: float,momentum: float,lr: float,dampening: float,nesterov: bool,maximize: bool):
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

    func(params,d_p_list,momentum_buffer_list,weight_decay=weight_decay,momentum=momentum,lr=lr,dampening=dampening,nesterov=nesterov,has_sparse_grad=has_sparse_grad,maximize=maximize)


def _single_tensor_sgd(params: List[Tensor],d_p_list: List[Tensor],momentum_buffer_list: List[Optional[Tensor]],*,weight_decay: float,momentum: float,lr: float,dampening: float,nesterov: bool,maximize: bool,has_sparse_grad: bool):
    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        # 使用 add 而不是 add_
        param.data = param.data.add(d_p, alpha=-lr)


def _multi_tensor_sgd(params: List[Tensor],grads: List[Tensor],momentum_buffer_list: List[Optional[Tensor]],*,weight_decay: float,momentum: float,lr: float,dampening: float,nesterov: bool,maximize: bool,has_sparse_grad: bool):
    if len(params) == 0:
        return

    if has_sparse_grad is None:
        has_sparse_grad = any(grad.is_sparse for grad in grads)

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    if momentum != 0:
        bufs = []

        all_states_with_momentum_buffer = True
        for i in range(len(momentum_buffer_list)):
            if momentum_buffer_list[i] is None:
                all_states_with_momentum_buffer = False
                break
            else:
                bufs.append(momentum_buffer_list[i])

        if all_states_with_momentum_buffer:
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads, alpha=1 - dampening)
        else:
            bufs = []
            for i in range(len(momentum_buffer_list)):
                if momentum_buffer_list[i] is None:
                    buf = momentum_buffer_list[i] = torch.clone(grads[i]).detach()
                else:
                    buf = momentum_buffer_list[i]
                    buf.mul_(momentum).add_(grads[i], alpha=1 - dampening)

                bufs.append(buf)

        if nesterov:
            torch._foreach_add_(grads, bufs, alpha=momentum)
        else:
            grads = bufs

    if not has_sparse_grad:
        # 使用 add 而不是 add_
        torch._foreach_sub_(params, grads, alpha=lr)
    else:
        # foreach APIs dont support sparse
        for i in range(len(params)):
            params[i].data = params[i].data.add(grads[i], alpha=-lr)

#实验定义pca方案
import torch
from typing import List, Optional
import random


def pca_sgd_incremental(
        params: List[torch.Tensor],
        d_p_list: List[torch.Tensor],
        momentum_buffer_list: List[Optional[torch.Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        has_sparse_grad: bool,
        energy_threshold: float = 0.9,
        projection_threshold: float = 1,
):
    """修改后的PCA-SGD函数，支持不同维度的参数"""

    with torch.no_grad():
        # 对每个参数单独处理
        for i, (param, d_p) in enumerate(zip(params, d_p_list)):
            if d_p is None:
                continue

            # 获取当前参数的梯度
            grad = d_p.flatten().unsqueeze(0)  # [1, param_dim]

            # 计算历史梯度均值（简化版PCA）
            if not hasattr(param, '_grad_history'):
                param._grad_history = grad.clone()
            else:
                # 更新历史梯度均值（指数移动平均）
                param._grad_history = 0.9 * param._grad_history + 0.1 * grad

            # 计算投影（简化版PCA）
            projection = torch.mm(grad, param._grad_history.T)
            update_flag = (projection.abs() > projection_threshold).any()

            # 仅当满足条件时更新参数
            if update_flag:
                d_p = d_p if not maximize else -d_p

                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)

                if momentum != 0:
                    buf = momentum_buffer_list[i]
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = buf if not nesterov else d_p.add(buf, alpha=momentum)

                param.data.add_(d_p, alpha=-lr)