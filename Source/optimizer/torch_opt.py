from torch import nn, optim


def torch_adam(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)
