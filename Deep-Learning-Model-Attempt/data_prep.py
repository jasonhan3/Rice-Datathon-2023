import torch
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'


"""
Helper function used to prepare the training data. 

args:
    - overall_data: the csv file read using pandas
    - states: an array of states to aggregate data from
    
returns: two pytorch tensors - the input_data and the actual values
    - input_data: a (5 * len(states)) * 19 tensor
    - actual: 5 * len(states) tensor
"""


def prepare_train_data(overall_data, states):
    input_data = torch.ones(5 * len(states), 18, device=device)
    actual = torch.ones(5 * len(states), device=device)

    states_data = overall_data.loc[(overall_data['State']).isin(states)]
    actual = torch.from_numpy(states_data["assistance"].unique()).float()

    states_data = states_data.drop('State', axis=1)
    states_data = states_data.drop('Year', axis=1)
    states_data = states_data.drop('CLPRB', axis=1)
    states_data = states_data.drop('emissions', axis=1)
    states_data = states_data.drop('assistance', axis=1)
    states_data = states_data.drop('numInvestments', axis=1)

    input_data = torch.from_numpy(states_data.to_numpy()).float()
    actual = actual
    return input_data, actual


def prepare_test_data(overall_data):
    input_data = torch.ones(50, 18, device=device)
    actual = torch.ones(50, device=device)

    actual = torch.from_numpy(overall_data["assistance"].to_numpy())

    overall_data = overall_data.drop('State', axis=1)
    overall_data = overall_data.drop('CLPRB', axis=1)
    overall_data = overall_data.drop('Year', axis=1)
    overall_data = overall_data.drop('emissions', axis=1)
    overall_data = overall_data.drop('assistance', axis=1)
    overall_data = overall_data.drop('numInvestments', axis=1)

    input_data = torch.from_numpy(overall_data.to_numpy())
    return input_data, actual



