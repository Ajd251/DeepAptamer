"""The paper's three classifiers, rebuilt layer for layer from the original script/model.py.

They use tf_keras (Keras 2) rather than Keras 3 so that the authors' Keras 2.4 weight files load
unchanged and the Attention layer behaves as it did when those weights were trained.
"""
import tf_keras as keras
from tf_keras import layers

SEQ_INPUT = (35, 4)
SHAPE_INPUT = (126, 1)


def build_cnn():
    model = keras.Sequential([
        keras.Input(SEQ_INPUT),
        layers.Conv1D(32, 12),
        layers.MaxPooling1D(4),
        layers.Flatten(),
        layers.Dense(16, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_bilstm(units=62, dropout=0.1):
    model = keras.Sequential([
        keras.Input(SEQ_INPUT),
        layers.Bidirectional(layers.LSTM(units, return_sequences=True)),
        layers.Dropout(dropout),
        layers.Bidirectional(layers.LSTM(units)),
        layers.Dropout(dropout),
        layers.Dense(2, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_deepaptamer(lstm_units=100, dropout=0.1, batch_attention=True):
    seq_in = keras.Input(SEQ_INPUT, name="md_1")
    x = layers.Conv1D(12, 1)(seq_in)
    x = layers.MaxPooling1D(1)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(4, activation="softmax")(x)  # (35, 4)

    shape_in = keras.Input(SHAPE_INPUT, name="md_2")
    s = layers.Conv1D(1, 100)(shape_in)
    s = layers.MaxPooling1D(20)(s)
    s = layers.Dense(4, activation="relu")(s)  # (1, 4): shape becomes a 36th "position"

    h = layers.Concatenate(axis=1)([x, s])
    h = layers.Bidirectional(layers.LSTM(100, return_sequences=True))(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Bidirectional(layers.LSTM(lstm_units))(h)
    h = layers.Dropout(dropout)(h)
    # On this 2-D tensor, Keras 2 attention mixes each sample with the others in its batch, so a
    # sequence's score depends on its batch-mates. With one sequence per batch it is the identity,
    # which is what batch_attention=False builds (it has no weights, so weight files still load).
    if batch_attention:
        h = layers.Attention()([h, h])
    out = layers.Dense(2, activation="sigmoid")(h)

    model = keras.Model([seq_in, shape_in], out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Subdirectory name (as used under model/) -> (display name, builder)
MODELS = {
    "cnn": ("CNN", build_cnn),
    "bilstm": ("BiLSTM", build_bilstm),
    "cnn_bilstm": ("DeepAptamer", build_deepaptamer),
}


def inputs_for(model, onehot, shape):
    return [onehot, shape] if len(model.inputs) == 2 else onehot


def load(name, weights_path):
    """Build model `name` for scoring and load its weights; every sequence is scored independently."""
    build = MODELS[name][1]
    model = build(batch_attention=False) if name == "cnn_bilstm" else build()
    model.load_weights(weights_path)
    return model


def predict(model, onehot, shape):
    """Probability of the high-affinity class for each sequence."""
    return model.predict(inputs_for(model, onehot, shape), batch_size=256, verbose=0)[:, 1]
