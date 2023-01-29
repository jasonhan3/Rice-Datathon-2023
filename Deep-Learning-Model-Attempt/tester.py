import torch
from torch import nn
import pandas as pd
import model, data_prep
from model import NeuralNetwork


device = 'cuda'

training_data = pd.read_csv("./data/flipped.csv")
test_data = pd.read_csv("./data/2020flipped.csv")

states = training_data['State'].unique()

inputs, outputs = data_prep.prepare_train_data(training_data, states)

test_inputs, test_outputs = data_prep.prepare_test_data(test_data)

norms = outputs.reshape(5, 50).norm(dim=0).to(device)
print("NORMS")
print(norms)


test_inputs = test_inputs.float().cuda()
test_outputs = test_outputs.float().cuda()

inputs = torch.nn.functional.normalize(inputs, dim=1)
outputs = torch.nn.functional.normalize(outputs, dim=0)
print(outputs)

print(inputs.shape)
print(outputs.shape)

nn_model = NeuralNetwork(len(states)).to(device)
nn_model = nn_model.float()

test_inputs = torch.nn.functional.normalize(test_inputs, dim=1)

epochs = 500
l_rate = 0.01
loss_fn = torch.nn.MSELoss()

model.train_loop(nn_model, inputs, outputs, loss_fn, l_rate, epochs)

print("PREDICTIONS:")
predictions = nn_model(test_inputs)
print(predictions)

print(predictions.shape)
print(norms.shape)
print(predictions.flatten() * norms)
