"""
model.py — Build & train the MELD emotion classifier (Keras)

Exports:
- create_emotion_model(...)
- compile_model(model, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=None)
- default_callbacks(...)
- train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, callbacks=None, verbose=1)
"""

from __future__ import annotations
from typing import Sequence, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def create_emotion_model(
    vocab_size: int,
    embedding_dim: int = 128,
    max_len: int = 128,
    num_classes: int = 7,
    lstm1_units: int = 64,
    lstm2_units: int = 32,
    dropout_lstm: float = 0.3,
    recurrent_dropout_lstm: float = 0.3,
    dense_units: Sequence[int] = (128, 64, 32),
    dense_dropouts: Sequence[float] = (0.5, 0.4, 0.3),
) -> tf.keras.Model:
    """
    Create the BiLSTM-based emotion classification model.

    Mirrors your improved architecture from the notebook:
      Embedding -> BiLSTM(64, return_sequences) -> BiLSTM(32)
      -> Dense(128)->Dropout(0.5) -> Dense(64)->Dropout(0.4) -> Dense(32)->Dropout(0.3)
      -> Dense(num_classes, softmax)

    Args:
        vocab_size: size of tokenizer vocabulary (cap). Use len(word_index)+1 capped.
        embedding_dim: embedding dimension.
        max_len: maximum sequence length (pads/truncates to this).
        num_classes: number of output classes (7 for MELD).
        lstm1_units: units in first BiLSTM.
        lstm2_units: units in second BiLSTM.
        dropout_lstm: dropout for LSTM inputs.
        recurrent_dropout_lstm: recurrent dropout inside LSTM.
        dense_units: sizes of the Dense “funnel” layers.
        dense_dropouts: dropouts after each Dense layer (same length as dense_units).

    Returns:
        A compiled (structure only) tf.keras.Model (you still need to compile()).
    """
    assert len(dense_units) == len(dense_dropouts), \
        "dense_units and dense_dropouts must be the same length"

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        input_length=max_len,
                        mask_zero=True))

    # Recurrent stack
    model.add(Bidirectional(LSTM(lstm1_units, return_sequences=True,
                                 dropout=dropout_lstm,
                                 recurrent_dropout=recurrent_dropout_lstm)))
    model.add(Bidirectional(LSTM(lstm2_units, return_sequences=False,
                                 dropout=dropout_lstm,
                                 recurrent_dropout=recurrent_dropout_lstm)))

    # Dense funnel
    for units, dr in zip(dense_units, dense_dropouts):
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(dr))

    # Output
    model.add(Dense(num_classes, activation="softmax"))
    return model


def compile_model(
    model: tf.keras.Model,
    optimizer: str | tf.keras.optimizers.Optimizer = "adam",
    loss: str = "sparse_categorical_crossentropy",
    metrics: Optional[Sequence] = None
) -> tf.keras.Model:
    """
    Compile the model with sensible defaults from your notebook.
    """
    if metrics is None:
        metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def default_callbacks(
    monitor_acc: str = "val_accuracy",
    patience_es: int = 5,
    monitor_lr: str = "val_loss",
    factor: float = 0.5,
    patience_rlrop: int = 3,
    min_lr: float = 1e-7,
    verbose: int = 1
):
    """
    Returns the EarlyStopping and ReduceLROnPlateau callbacks you used.
    """
    return [
        EarlyStopping(monitor=monitor_acc, patience=patience_es,
                      restore_best_weights=True, verbose=verbose),
        ReduceLROnPlateau(monitor=monitor_lr, factor=factor,
                          patience=patience_rlrop, min_lr=min_lr, verbose=verbose),
    ]


def train_model(
    model: tf.keras.Model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 20,
    batch_size: int = 32,
    callbacks: Optional[Sequence] = None,
    verbose: int = 1
):
    """
    Fit the model using your training block (without any evaluation/printing).
    """
    if callbacks is None:
        callbacks = default_callbacks(verbose=verbose)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    return history
