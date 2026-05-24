"""
Custom Layer & Loss buat TensorFlow — dipakai saat load model di FastAPI.
"""

import keras
from keras.layers import Dense
import tensorflow as tf

_original_dense_init = Dense.__init__


def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _original_dense_init(self, *args, **kwargs)


Dense.__init__ = _patched_dense_init


# CUSTOM LAYER: SCALED DENSE
@tf.keras.utils.register_keras_serializable()
class ScaledDenseLayer(tf.keras.layers.Layer):
    """Dense + learnable scale factor (s). Output = s * (W . input + b)."""

    def __init__(self, units, initial_scale=10.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.initial_scale = initial_scale

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_scale),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        logits = tf.matmul(inputs, self.kernel) + self.bias
        return logits * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'initial_scale': self.initial_scale,
        })
        return config


# CUSTOM LOSS: FOCAL LOSS
@tf.keras.utils.register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss untuk handle class imbalance.
    FL(p) = -alpha * (1-p)^gamma * log(p)."""

    def __init__(self, gamma=2.0, alpha=0.25, num_classes=25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        y_true = tf.cast(
            tf.one_hot(tf.cast(y_true, tf.int32), self.num_classes),
            tf.float32
        )
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        loss = focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'num_classes': self.num_classes,
        })
        return config


# CUSTOM CALLBACK: TRAINING LOGGER
class TrainingLogger(tf.keras.callbacks.Callback):
    """Callback yang ngeprint log per epoch + auto-stop kalau val_accuracy
    lewat target."""

    def __init__(self, target_val_acc=0.92):
        super().__init__()
        self.target_val_acc = target_val_acc
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        msg = (
            f'[Epoch {epoch+1:3d}] '
            f'loss={loss:.4f} acc={acc:.4f} | '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}'
        )
        if val_acc >= self.target_val_acc:
            msg += f' >>> TARGET ({self.target_val_acc}) TERCAPAI!'
            self.model.stop_training = True
        print(msg)
        self.history.append({'epoch': epoch + 1, **logs})


# Dipakai saat load model
CUSTOM_OBJECTS = {
    'ScaledDenseLayer': ScaledDenseLayer,
    'FocalLoss': FocalLoss,
}