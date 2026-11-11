"""Released SchemaGAN loading and RNG initialization; see THIRD_PARTY.md."""

import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/schemaGAN.h5"


def load_model(seed):
    import h5py
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    started = time.perf_counter()
    model = tf.keras.models.load_model(MODEL, compile=False)
    load_seconds = time.perf_counter() - started
    if model.input_shape[1:] != (32, 512, 1) or model.output_shape[1:] != (32, 512, 1):
        raise ValueError("Unexpected model input or output shape")
    environment = {
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "device": "CPU",
        "threads": 1,
        "op_determinism": True,
        "oneDNN": True,
    }
    return model, load_seconds, environment
