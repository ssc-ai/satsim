"""Types for typing functions signatures."""

from typing import Union, Callable, List

import importlib
import numpy as np
import tensorflow as tf

# TODO: Remove once https://github.com/tensorflow/tensorflow/issues/44613 is resolved
# TensorFlow and Keras frequently move the internal location of
# ``keras_tensor``.  Try a list of known locations so that we work across
# multiple TensorFlow/Keras versions (e.g. TF 2.5 through 2.16+).
_KERAS_TENSOR_IMPORT_PATHS = [
    "keras.src.engine.keras_tensor",  # TF >= 2.13
    "keras.engine.keras_tensor",      # TF < 2.13
    "keras.src.backend.keras_tensor",  # Keras 3.x
    "tensorflow.python.keras.engine.keras_tensor",
]

keras_tensor = None
for _path in _KERAS_TENSOR_IMPORT_PATHS:
    try:
        keras_tensor = importlib.import_module(_path)
        break
    except Exception:
        continue

if keras_tensor is None:
    # Fall back to KerasTensor class exposed via tf.keras.__internal__
    try:
        from tensorflow.keras.__internal__ import KerasTensor as keras_tensor
    except Exception:  # pragma: no cover - extremely old TF
        from tensorflow.python.keras.engine import keras_tensor  # type: ignore


Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

Initializer = Union[None, dict, str, Callable, tf.keras.initializers.Initializer]
Regularizer = Union[None, dict, str, Callable, tf.keras.regularizers.Regularizer]
Constraint = Union[None, dict, str, Callable, tf.keras.constraints.Constraint]
Activation = Union[None, str, Callable]
if importlib.util.find_spec("tensorflow.keras.optimizers.legacy") is not None:
    Optimizer = Union[
        tf.keras.optimizers.Optimizer, tf.keras.optimizers.legacy.Optimizer, str
    ]
else:
    Optimizer = Union[tf.keras.optimizers.Optimizer, str]

TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
    keras_tensor.KerasTensor,
]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]
