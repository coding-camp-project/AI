"""
    <>Custom Layer sma loss tensorflow<>
"""

import keras
from keras.layers import Dense
import tensorflow as tf

_original_dense_init = Dense.__init__


def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _original_dense_init(self, *args, **kwargs)


Dense.__init__ = _patched_dense_init


# Custom layer
@tf.keras.utils.register_keras_serializable()
class L2NormalizationLayer(tf.keras.layers.Layer):
    """Layer noermalisasi l2 (arsitektur lama)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)
    
    def get_config(self):
        return super().get_config()

# Custom layer 2
@tf.keras.utils.register_keras_serializable()
class ScaledDenseLayer(tf.keras.layers.Layer):
    """dense layer + learnabel scale factor (arsitketur baru)"""
    
    def __init__(self, units, initial_scale=3.0, **kwargs): #diganti
        super().__init__(**kwargs)
        self.units = units
        self.initial_scale = initial_scale
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
        
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_scale),
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        logits = tf.matmul(inputs, self.kernel) + self.bias
        return logits * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "initial_scale": self.initial_scale
        })
        return config
    

# Custom loss function
@tf.keras.utils.register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    """Focal loss -> bobot lebih kalo classnya kompleks"""
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
            'num_classes': self.num_classes
        })
        return config
        


# Dipakai saat laod model
CUSTOM_OBJECTS = {
    'L2NormalizationLayer': L2NormalizationLayer,
    'ScaledDenseLayer': ScaledDenseLayer,
    'FocalLoss': FocalLoss,
}