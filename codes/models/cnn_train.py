import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import metrics as met
import tqdm

import torch.optim.lr_scheduler as scheduler


class CNN_Trainer:
    def __init__(self, config: dict, device):
        self.config = config
        self.epochs = self.config.get('epochs')
        self.device = device
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.loss = self._build_loss()
        self.metrics = self._build_metrics()
        
    def _build_model(self):
        arch = self.config.get('arch').get('name')
        
        if arch == 'lite':
            return self._lite()
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
    
    def _build_optimizer(self):
        optim_param = self.config.get('optimizer')

        name = optim_param.pop('name')
        
        if name == 'AdamW':
            return optim.AdamW(self.model.parameters(), **optim_param)
        if name == "SGD":
            return optim.SGD(self.model.parameters(), **optim_param)
        elif name == "RMSprop":
            return optim.RMSprop(self.model.parameters(), **optim_param)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")
    
    def _build_scheduler(self):
        scheduler_param = self.config.get('scheduler')

        name = scheduler_param.pop('name')
        
        if name == 'OneCycleLR':
            return scheduler.OneCycleLR(
                self.optimizer,
                epochs=self.epochs,
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

    def _lite(self):
        layers = []
        for layer_param in self.config.get('layers'):
            type_layer = layer_param.pop('type')
            if type_layer == "Conv2d":
                layers.append(nn.Conv2d(**layer_param))
            elif type_layer == "ReLU":
                layers.append(nn.ReLU())
            elif type_layer == "Flatten":
                layers.append(nn.Flatten())
            elif type_layer == "BatchNorm2d":
                layers.append(nn.BatchNorm2d(**layer_param))
            elif type_layer == "Linear":
                layers.append(nn.Linear(**layer_param))
        return nn.Sequential(*layers).to(self.device)

    
    def train(
            self,
            train_dl,
            val_dl,
            device
              ):
        losses = []
        val_lossses = []

        for epoch in range(self.epochs):
            running_loss = 0.0
            correct_prediction = 0
            total_prediction = 0

            for X, y in tqdm(train_dl):
                X, y = X.to(device), y.to(device)
                prediction = self.model(X)
                loss = self.loss(prediction, y.long())

                running_loss += loss.item()
                total_prediction += prediction.shape[0]

                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                ...