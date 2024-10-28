import lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


class SoftmaxMLPClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, learning_rate=0.001):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Additional hidden layer
        self.fc3 = nn.Linear(hidden_dim, 2)  # Output layer with 2 units for softmax
        self.learning_rate = learning_rate


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Activation for additional hidden layer
        x = self.fc3(x)  # No sigmoid activation, logits are expected by CrossEntropyLoss
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.long())  # Use cross_entropy, which includes softmax
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.long())  # Use cross_entropy, which includes softmax
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)