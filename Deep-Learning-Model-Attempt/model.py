import torch
from torch import nn

device = 'cuda'


class NeuralNetwork(nn.Module):
    def __init__(self, states):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(18, 250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.ReLU(),
            nn.Linear(250, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(model, inputs, actual, loss_fn, learning_rate, epochs):
    inputs = inputs.to(device)
    actual = actual.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        totalLoss = 0

        # Compute the loss function
        pred = model(inputs)
        loss = loss_fn(pred, actual)

        totalLoss += loss.item()

        # Back Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(totalLoss)


def test_loop(model, inputs, actual, loss_fn):
    with torch.no_grad():
        y_pred = model(inputs)
        print("Test Prediction: " + str(y_pred))
        loss = loss_fn(y_pred, actual)
        print('Test Loss: {:.4f}'.format(loss))

