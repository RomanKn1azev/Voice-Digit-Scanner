import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import codes.models.metrics as met


import torch.optim.lr_scheduler as scheduler

from tqdm import tqdm
from codes.utils.utils import evaluate


class CNN:
    def __init__(self, arch, config: dict, device, steps_per_epoch=1):
        self.arch = arch
        self.config = config
        self.epochs = self.config.get('epochs')
        self.device = device
        self.steps_per_epoch = steps_per_epoch
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.loss = self._build_loss()
        self.metrics = self._build_metrics()
        
    def _build_model(self):        
        if self.arch == 'lite':
            return self._lite()
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")
    
    def _build_optimizer(self):
        optim_param = self.config.get('optimizer')

        name = optim_param.pop('name')
        
        if name == 'AdamW':
            return optim.AdamW(self.model.parameters(), **optim_param)
        if name == "SGD":
            return optim.SGD(self.model.parameters(), **optim_param)
        elif name == "RMSprop":
            return optim.RMSprop(self.model.parameters(), **optim_param)
        elif name == "Adam":
            return optim.Adam(self.model.parameters(), **optim_param)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")
    
    def _build_scheduler(self):
        scheduler_param = self.config.get('scheduler')

        name = scheduler_param.pop('name')
        
        if name == 'OneCycleLR':
            return scheduler.OneCycleLR(
                self.optimizer,
                epochs=self.epochs,
                steps_per_epoch = self.steps_per_epoch,
                **scheduler_param
                )
        else:
            raise ValueError(f"Unsupported scheduler: {name}")
    
    def _build_loss(self):
        loss_name = self.config.get('loss')

        if loss_name == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
        
    def _build_metrics(self) -> dict:
        metrics = self.config.get('metrics')

        metrics_dict = {}

        for metric in metrics:
            if metric == 'accuracy':
                metrics_dict['accuracy'] = met.create_accuracy_metric()
            elif metric == "f1":
                metrics_dict["f1"] = met.create_f1_metric()
            elif metric == "roc_auc":
                metrics_dict["roc_auc"] = met.create_roc_auc_metric()
            elif metric == "precision":
                metrics_dict["precision"] = met.create_precision_metric()
            elif metric == "recall":
                metrics_dict["recall"] = met.create_recall_metric()
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        return metrics_dict
    
    def _get_model(self, sequential):
        class CNN2DAudioClassifier(nn.Module):
            def __init__(self, sequential):
                super().__init__()
                self.sequential = sequential
            
            def forward(self, x):
                return self.sequential(x)
        
        return CNN2DAudioClassifier(sequential).to(self.device)


    def _lite(self):
        layers = []
        for layer_param in self.config.get('layers'):
            type_layer = layer_param.pop('type')
            if type_layer == "Conv2d":
                layers.append(nn.Conv2d(**layer_param))
            elif type_layer == "MaxPool2d":
                layers.append(nn.MaxPool2d(**layer_param))
            elif type_layer == "ReLU":
                layers.append(nn.ReLU())
            elif type_layer == "Flatten":
                layers.append(nn.Flatten())
            elif type_layer == "BatchNorm2d":
                layers.append(nn.BatchNorm2d(**layer_param))
            elif type_layer == "AdaptiveAvgPool2d":
                layers.append(nn.AdaptiveAvgPool2d(**layer_param))
            elif type_layer == "Linear":
                layers.append(nn.Linear(**layer_param))

        return self._get_model(nn.Sequential(*layers))
    
    def train(
            self,
            train_dl,
            val_dl,
              ):
        losses = []
        val_lossses = []
        accuracy = []
        val_accuracy = []

        for epoch in range(self.epochs):
            running_loss = 0.0
            correct_prediction = 0
            total_prediction = 0
            
            for X, y in tqdm(train_dl):
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.loss(outputs, y.long())

                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()

                _, prediction = torch.max(outputs, 1)
                correct_prediction += (prediction == y).sum().item()
                total_prediction += prediction.shape[0]

            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction

            v_acc, v_loss = evaluate(self.model, val_dl, self.device, self._build_loss())
                
            print("Epoch: %d, Loss: %.4f, Train Accuracy: %.2f, Val. Loss: %.4f, Val. Accuracy: %.2f" % (
                epoch + 1, avg_loss, acc, v_loss, v_acc
                ))

            losses.append(avg_loss)
            val_lossses.append(v_loss)

            accuracy.append(acc)
            val_accuracy.append(v_acc)
        
        return losses, val_lossses, accuracy, val_accuracy