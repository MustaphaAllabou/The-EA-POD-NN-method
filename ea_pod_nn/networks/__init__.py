"""Neural-network backends of EA-POD-NN.

Each backend module provides the same five functions, so the rest of the
package never imports TensorFlow or PyTorch directly:

    set_seed(seed)
    train(X_train, Y_train, X_val, Y_val, neurons, config) -> (model, history)
        history = {"loss": [...], "val_loss": [...]}
    predict(model, X) -> np.ndarray
    save(model, path)          # path without extension
    load(path) -> model        # path without extension
"""

BACKENDS = ("keras", "torch")


def get_backend(name):
    """Return the backend module for "keras" (TensorFlow) or "torch" (PyTorch)."""
    if name == "keras":
        from . import keras_backend as backend
    elif name == "torch":
        from . import torch_backend as backend
    else:
        raise ValueError(f"Unknown backend {name!r}, choose among {BACKENDS}")
    return backend
