from pyexpat import model
import torch
from agent.networks import CNN

class BCAgent:
    """ BCAgent """
    def __init__(self, learning_rate=0.01):
        # TODO: Define network, loss function, optimizer
        self.net = CNN(n_classes=4)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), learning_rate, momentum=0.9)

    def update(self, X_batch, y_batch):
    #     # TODO: transform input to tensors
        # print(X_batch.shape)    # torch.Size([64, 1, 96, 96])
    #     # TODO: forward + backward + optimize
        loss = 0
        self.optimizer.zero_grad()
        output_train = self.net(X_batch)
        loss_train = self.loss(output_train, y_batch)

        loss_train.backward()
        self.optimizer.step()
        loss = loss_train.item()

        return loss

    def predict(self, X):
        # TODO: forward pass
        outputs = self.net(X)
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
