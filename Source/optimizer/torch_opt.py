from torch import nn, optim


def torch_adam(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)


def torch_adamax(model, learning_rate):
    return optim.Adamax(model.parameters(), lr=learning_rate)


def torch_sgd(model, learning_rate):
    return optim.SGD(model.parameters(), lr=learning_rate)
