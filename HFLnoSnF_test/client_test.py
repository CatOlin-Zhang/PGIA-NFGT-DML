import copy
import torch
import torch.nn as nn
from fedml.simulation.sp.fedavg.client import Client
from optimizers.pca_sgd import PCASGD  # 新增导入


class HFLClient(Client):
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model,
                 model_trainer):
        super().__init__(client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                         model_trainer)
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.args = args
        self.device = device
        self.model = model
        self.model_trainer = model_trainer
        self.criterion = nn.CrossEntropyLoss().to(device)

        # 绑定优化器到模型（用于hook访问）
        self.model.optimizer = None

    def train(self, global_round_idx, group_round_idx, w):
        self.model.load_state_dict(w)
        self.model.to(self.device)

        # 初始化优化器
        if self.args.client_optimizer == "sgd":
            optimizer = PCASGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=getattr(self.args, 'momentum', 0.9),  # 安全获取
                weight_decay=getattr(self.args, 'wd', 0.0),
                pca_components=getattr(self.args, 'pca_components', 3),
                update_threshold=getattr(self.args, 'update_threshold', 0.5)
            )
            # 绑定优化器实例到模型
            self.model.optimizer = optimizer
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.wd,
                amsgrad=True,
            )

        w_list = []
        for epoch in range(self.args.epochs):
            # 清空PCA激活缓冲区（每epoch清空）
            if isinstance(optimizer, PCASGD):
                optimizer.activation_buffers.clear()

            for x, labels in self.local_training_data:
                x, labels = x.to(self.device), labels.to(self.device)

                # 前向传播（自动记录激活值）
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 参数更新
                optimizer.step()

            # 记录模型状态
            global_epoch = (global_round_idx * self.args.group_comm_round * self.args.epochs +
                            group_round_idx * self.args.epochs + epoch)
            if global_epoch % self.args.frequency_of_the_test == 0 or epoch == self.args.epochs - 1:
                w_list.append((global_epoch, copy.deepcopy(self.model.state_dict())))

        return w_list