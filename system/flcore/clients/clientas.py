import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from utils.privacy import *
from torch.autograd import grad
from torchviz import make_dot
import torch.optim as optim




class clientAS(Client):

    # model shape
    ######################################################################################################
    # (
    #   (base): FedAvgCNN(
    #     (conv1): Sequential(
    #       (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))
    #       (1): ReLU(inplace=True)
    #       (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    #     )
    #     (conv2): Sequential(
    #       (0): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
    #       (1): ReLU(inplace=True)
    #       (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    #     )
    #     (fc1): Sequential(
    #       (0): Linear(in_features=1600, out_features=512, bias=True)
    #       (1): ReLU(inplace=True)
    #     )
    #     (fc): Identity()
    #   )
    #   (head): Linear(in_features=512, out_features=100, bias=True)
    # )
    ######################################################################################################

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        self.fim_trace_history = []
        self.yxy_hyper = args.yxy_hyper
        self.wo_local = args.wo_local
        self.drift_aware = args.drift_aware
        self.drift_fim_weight = args.drift_fim_weight
        self.drift_proto_weight = args.drift_proto_weight
        self.drift_loss_weight = args.drift_loss_weight
        self.drift_clip = args.drift_clip
        self.drift_norm_eps = args.drift_norm_eps
        self.personalization_min = args.personalization_min
        self.personalization_max = args.personalization_max

        self.prev_fim = None
        self.prev_train_loss = None
        self.prev_prototypes = None
        self.current_drift_score = 0.0
        self.drift_score_history = []

    def _extract_mean_prototypes(self, batch_size=None):
        if batch_size is None:
            batch_size = self.args.prototype_batch_size if hasattr(self.args, "prototype_batch_size") else 16

        local_prototypes = [[] for _ in range(self.num_classes)]
        trainloader = self.load_train_data(batch_size=batch_size)
        self.model.eval()

        with torch.no_grad():
            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                proto_batch = self.model.base(x_batch)

                for proto, y in zip(proto_batch, y_batch):
                    local_prototypes[y.item()].append(proto.detach().clone())

        mean_prototypes = []
        for class_prototypes in local_prototypes:
            if class_prototypes:
                mean_prototypes.append(torch.mean(torch.stack(class_prototypes), dim=0))
            else:
                mean_prototypes.append(None)

        return mean_prototypes

    def _prototype_shift(self, current_prototypes):
        if self.prev_prototypes is None:
            return 0.0

        total_shift = 0.0
        count = 0
        for current_proto, prev_proto in zip(current_prototypes, self.prev_prototypes):
            if current_proto is not None and prev_proto is not None:
                total_shift += torch.norm(current_proto - prev_proto, p=2).item()
                count += 1

        return total_shift / max(count, 1)

    def _compute_drift_score(self, fim_trace_value, train_loss_value, current_prototypes):
        if not self.drift_aware:
            return 0.0

        if self.prev_fim is None:
            fim_change = 0.0
        else:
            fim_change = abs(fim_trace_value - self.prev_fim) / max(abs(self.prev_fim), self.drift_norm_eps)

        proto_shift = self._prototype_shift(current_prototypes)

        if self.prev_train_loss is None:
            loss_spike = 0.0
        else:
            loss_spike = max(0.0, train_loss_value - self.prev_train_loss) / max(abs(self.prev_train_loss), self.drift_norm_eps)

        drift_score = (
            self.drift_fim_weight * fim_change +
            self.drift_proto_weight * proto_shift +
            self.drift_loss_weight * loss_spike
        )

        return float(np.clip(drift_score, 0.0, self.drift_clip))

    def _update_drift_state(self, fim_trace_value, train_loss_value, current_prototypes, drift_score):
        self.prev_fim = float(fim_trace_value)
        self.prev_train_loss = float(train_loss_value)
        self.prev_prototypes = [
            proto.detach().clone() if proto is not None else None
            for proto in current_prototypes
        ]
        self.current_drift_score = float(drift_score)
        self.drift_score_history.append(float(drift_score))

    def train(self, is_selected):
        trainloader = self.load_train_data()

        if is_selected:
            self.model.train()

            # differential privacy
            if self.privacy:
                self.model, self.optimizer, trainloader, privacy_engine = \
                    initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
            start_time = time.time()

            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            total_loss = 0.0
            total_batches = 0

            for step in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    total_batches += 1

            # self.model.cpu()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time

            if self.privacy:
                eps, DELTA = get_dp_params(privacy_engine)
                print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
            train_loss_value = total_loss / max(total_batches, 1)

        else:
            self.model.eval()
            train_loss_value = self.prev_train_loss if self.prev_train_loss is not None else 0.0

        self.model.eval()

        # Compute FIM and its trace after training
        fim_trace_sum = 0
        eval_loader = self.load_train_data()
        for i, (x, y) in enumerate(eval_loader):
            # Forward pass
            x = x.to(self.device)
            y = y.to(self.device)
            outputs = self.model(x)
            # Negative log likelihood as our loss
            nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(y)), y].mean()

            # Compute gradient of the negative log likelihood w.r.t. model parameters
            grads = grad(nll, self.model.parameters())

            # Compute and accumulate the trace of the Fisher Information Matrix
            for g in grads:
                fim_trace_sum += torch.sum(g ** 2).detach()

        fim_trace_value = fim_trace_sum.item()
        self.fim_trace_history.append(fim_trace_value)

        current_prototypes = self._extract_mean_prototypes()
        drift_score = self._compute_drift_score(
            fim_trace_value=fim_trace_value,
            train_loss_value=train_loss_value,
            current_prototypes=current_prototypes,
        )
        self._update_drift_state(
            fim_trace_value=fim_trace_value,
            train_loss_value=train_loss_value,
            current_prototypes=current_prototypes,
            drift_score=drift_score,
        )

    def evaluate(self):
        testloader = self.load_test_data()
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        accuracy = 100. * correct / total
        return accuracy
    
    # def set_parameters(self, model, progress):
        # # Substitute the parameters of the base, enabling personalization
        # for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
        #     old_param.data = new_param.data.clone()

    def set_parameters(self, model, progress):

        # print(f'client{self.id}')
        if self.wo_local:
            # Substitute the parameters of the base, enabling vanilla personalization
            for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
                old_param.data = new_param.data.clone()
        else:
            local_prototypes = self._extract_mean_prototypes()
            trainloader = self.load_train_data(batch_size=self.args.prototype_batch_size)

            # Align global model's prototype with the local prototype
            alignment_optimizer = torch.optim.SGD(model.base.parameters(), lr=0.01)  # Adjust learning rate and optimizer as needed
            alignment_loss_fn = torch.nn.MSELoss()

            # print(f'client{self.id}')
            for _ in range(1):  # Iterate for 1 epochs; adjust as needed
                for x_batch, y_batch in trainloader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    global_proto_batch = model.base(x_batch)
                    loss = 0
                    for label in y_batch.unique():
                        if local_prototypes[label.item()] is not None:
                            loss += alignment_loss_fn(global_proto_batch[y_batch == label], local_prototypes[label.item()])
                    alignment_optimizer.zero_grad()
                    loss.backward()
                    alignment_optimizer.step()

            if self.drift_aware:
                alpha = np.clip(
                    self.personalization_min + self.current_drift_score,
                    self.personalization_min,
                    self.personalization_max
                )
            else:
                alpha = 0.0

            # Substitute the parameters of the base, enabling personalization
            for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
                old_param.data = (1 - alpha) * new_param.data.clone() + alpha * old_param.data.clone()


            # end
