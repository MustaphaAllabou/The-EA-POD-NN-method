"""PyTorch backend.

Same fully-connected architecture as the Keras backend: ``hidden_layers``
Dense layers with ReLU, then a linear output layer, trained with Adam on
the MSE loss. Computations run on the CPU, which is the fastest choice for
networks of this size.
"""
import copy

import numpy as np
import torch
from torch import nn

EXTENSION = ".pt"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def build(input_dim, output_dim, neurons, hidden_layers):
    layers, size = [], input_dim
    for _ in range(hidden_layers):
        layers += [nn.Linear(size, neurons), nn.ReLU()]
        size = neurons
    layers.append(nn.Linear(size, output_dim))
    model = nn.Sequential(*layers)
    # Remember the architecture so that save() / load() can rebuild it.
    model.architecture = dict(input_dim=input_dim, output_dim=output_dim,
                              neurons=neurons, hidden_layers=hidden_layers)
    return model


def _tensor(array):
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


def train(X_train, Y_train, X_val, Y_val, neurons, config):
    model = build(X_train.shape[1], Y_train.shape[1], neurons, config.hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(_tensor(X_train), _tensor(Y_train)),
        batch_size=config.batch_size, shuffle=True)
    X_val, Y_val = _tensor(X_val), _tensor(Y_val)

    history = {"loss": [], "val_loss": []}
    best_loss, best_state, epochs_without_progress = np.inf, None, 0

    for epoch in range(1, config.epochs + 1):
        # One pass over the training data
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / len(loader.dataset)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), Y_val).item()

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if config.verbose and (epoch == 1 or epoch % 50 == 0 or epoch == config.epochs):
            print(f"Epoch {epoch}/{config.epochs} - loss: {train_loss:.4e} "
                  f"- val_loss: {val_loss:.4e}")

        # Early stopping, keeping the best weights
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
            epochs_without_progress = 0
        else:
            epochs_without_progress += 1
            if config.patience and epochs_without_progress >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if config.patience and best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(_tensor(X)).numpy()


def save(model, path):
    torch.save({"architecture": model.architecture, "state_dict": model.state_dict()},
               path + EXTENSION)


def load(path):
    checkpoint = torch.load(path + EXTENSION, weights_only=True)
    model = build(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
