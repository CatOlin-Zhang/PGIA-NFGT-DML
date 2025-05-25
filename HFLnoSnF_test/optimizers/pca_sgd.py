import torch
import torch.nn as nn


class PCASGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *,
                 pca_components=3, energy_threshold=0.9,
                 update_threshold=0.5):
        # 标准SGD参数初始化
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        pca_components=pca_components,
                        energy_threshold=energy_threshold,
                        update_threshold=update_threshold)
        super().__init__(params, defaults)

        # PCA相关初始化
        self.activation_buffers = {}
        self.pca_models = {}

    def register_activation(self, layer_name, activation):
        """注册各层的激活值（应在forward hook中调用）"""
        if layer_name not in self.activation_buffers:
            self.activation_buffers[layer_name] = []
        self.activation_buffers[layer_name].append(activation.detach().clone())

        # 保持固定长度的历史记录
        if len(self.activation_buffers[layer_name]) > 100:
            self.activation_buffers[layer_name].pop(0)

    def _compute_pca(self, activations):
        """计算激活值的PCA模型"""
        flat_acts = torch.cat([act.flatten(1) for act in activations], dim=0)
        centered = flat_acts - flat_acts.mean(dim=0)
        U, S, V = torch.pca_lowrank(centered, q=self.defaults['pca_components'])
        return V[:, :self.defaults['pca_components']]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # 获取层名称（需要与模型hook配合）
                layer_name = self._get_layer_name(p)
                if layer_name not in self.pca_models:
                    if layer_name in self.activation_buffers:
                        self.pca_models[layer_name] = self._compute_pca(
                            self.activation_buffers[layer_name]
                        )
                    else:
                        # 没有激活记录时使用标准SGD
                        self._standard_sgd_step(p, group)
                        continue

                # 计算梯度在主成分上的投影
                pca_basis = self.pca_models[layer_name]
                grad_flat = p.grad.flatten()
                projection = torch.matmul(pca_basis.T, grad_flat)

                # 根据投影强度决定更新
                if projection.norm() > group['update_threshold']:
                    self._standard_sgd_step(p, group)

        return loss

    def _standard_sgd_step(self, param, group):
        """标准SGD更新步骤"""
        d_p = param.grad
        if group['weight_decay'] != 0:
            d_p = d_p.add(param, alpha=group['weight_decay'])

        if group['momentum'] != 0:
            param_state = self.state[param]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(group['momentum']).add_(d_p, alpha=1 - group['dampening'])

            if group['nesterov']:
                d_p = d_p.add(buf, alpha=group['momentum'])
            else:
                d_p = buf

        param.add_(d_p, alpha=-group['lr'])

    def _get_layer_name(self, param):
        """辅助方法：通过参数指针获取层名称"""
        for name, module in param._model.named_modules():
            if any(p is param for p in module.parameters()):
                return name
        return None