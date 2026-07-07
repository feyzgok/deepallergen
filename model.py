import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, concatenate, Conv1D, Conv1DTranspose, \
    BatchNormalization, Dropout, Dense, Flatten, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras import Input


def return_model(model_name):
    model_dic = {'DeepAllergen': DeepAllergen()}
    return model_dic[model_name]

@tf.keras.utils.register_keras_serializable()
class SqueezeExcitation1DLayer(tf.keras.Model):

    def __init__(self, out_dim, ratio, layer_name="se", **kwargs):
        if 'name' in kwargs:
            layer_name = kwargs.pop('name')
        super(SqueezeExcitation1DLayer, self).__init__(name=layer_name, **kwargs)
        self.out_dim = out_dim
        self.ratio = ratio
        self.squeeze = GlobalAveragePooling1D()
        self.excitation_a = Dense(units=int(out_dim / ratio), activation='relu')
        self.excitation_b = Dense(units=out_dim, activation='sigmoid')
        self.shape = [-1, 1, out_dim]

    def call(self, input_x):
        squeeze = self.squeeze(input_x)
        excitation = self.excitation_a(squeeze)
        excitation = self.excitation_b(excitation)
        scale = tf.reshape(excitation, self.shape)
        se = input_x * scale
        return se

    def get_config(self):
        config = super(SqueezeExcitation1DLayer, self).get_config()
        config.update({
            "out_dim": self.out_dim,
            "ratio": self.ratio,
            "layer_name": self.name,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.3, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
             Dense(embed_dim, kernel_regularizer=tf.keras.regularizers.L2(1e-4))]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class PositionEmbedding(Layer):
    def __init__(self, max_len, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.max_len = max_len

    def call(self, x):
        positions = tf.range(start=0, limit=111, delta=111 / self.max_len)
        positions = tf.math.sin(positions)
        return tf.math.multiply(x, tf.expand_dims(tf.cast(positions, dtype='float32'), axis=-1))

    def get_config(self):
        config = super(PositionEmbedding, self).get_config()
        config.update({"max_len": self.max_len})
        return config


def EncoderSeBlock(inputs, n_filters=15, kernel_size=7, dropout_prob=0.4, ratio=2, layer_name="",
                   condensing=True):
    conv = Conv1D(n_filters, kernel_size, activation='relu', padding='same',
                  kernel_initializer='HeNormal',
                  kernel_regularizer=tf.keras.regularizers.L2(1e-4))(inputs)
    se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name + "_0")(conv)
    bn = BatchNormalization()(se)
    conv = Conv1D(n_filters, kernel_size, activation='relu', padding='same',
                  kernel_initializer='HeNormal',
                  kernel_regularizer=tf.keras.regularizers.L2(1e-4))(bn)
    se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name + "_1")(conv)
    conv = BatchNormalization()(se)

    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    if condensing:
        next_layer = Conv1D(n_filters, kernel_size, activation='relu', strides=2, padding='same',
                            kernel_initializer='HeNormal',
                            kernel_regularizer=tf.keras.regularizers.L2(1e-4))(conv)
    else:
        next_layer = conv

    skip_connection = conv
    return next_layer, skip_connection


def DecoderSeBlock(prev_layer_input, skip_layer_input, kernel_size=7, ratio=2, layer_name="", n_filters=32):
    up = Conv1DTranspose(filters=n_filters, kernel_size=kernel_size, strides=2, padding='same',
                         kernel_regularizer=tf.keras.regularizers.L2(1e-4))(prev_layer_input)
    merge = concatenate([up, skip_layer_input], axis=-1)

    conv = Conv1D(n_filters, kernel_size, activation='relu', padding='same',
                  kernel_initializer='HeNormal',
                  kernel_regularizer=tf.keras.regularizers.L2(1e-4))(merge)
    se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name + "_0")(conv)
    bn = BatchNormalization()(se)
    conv = Conv1D(n_filters, kernel_size, activation='relu', padding='same',
                  kernel_initializer='HeNormal',
                  kernel_regularizer=tf.keras.regularizers.L2(1e-4))(bn)
    se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name + "_1")(conv)
    bn = BatchNormalization()(se)
    return bn


def DeepAllergen(input_size=(112, 1024), n_filters=128, n_head=8, kernel_size=11, n_ff_dims=128,
                 n_classes=1):
    inputs = Input(input_size)
    max_len, vocab_size = input_size
    max_len = 14

    # ==================== ENCODER ====================
    cblock1 = EncoderSeBlock(inputs, n_filters, kernel_size=kernel_size, dropout_prob=0.2,
                             layer_name="ecb_1_", condensing=True)
    cblock2 = EncoderSeBlock(cblock1[0], 128, kernel_size=kernel_size, dropout_prob=0.3,
                             layer_name="ecb_2_", condensing=True)
    cblock3 = EncoderSeBlock(cblock2[0], 256, kernel_size=kernel_size, dropout_prob=0.4,
                             layer_name="ecb_3_", condensing=True)
    cblock4 = EncoderSeBlock(cblock3[0], 256, kernel_size=kernel_size, dropout_prob=0.3,
                             layer_name="ecb_4_", condensing=False)

    # ==================== TRANSFORMER ====================
    embedding_layer_1 = PositionEmbedding(max_len)

    transformer_block_1 = TransformerBlock(embed_dim=256, num_heads=n_head, ff_dim=n_ff_dims, rate=0.3)
    transformer_block_2 = TransformerBlock(embed_dim=256, num_heads=n_head, ff_dim=n_ff_dims, rate=0.3)
    transformer_block_3 = TransformerBlock(embed_dim=256, num_heads=n_head, ff_dim=n_ff_dims, rate=0.3)
    transformer_block_4 = TransformerBlock(256, n_head, n_ff_dims, rate=0.3)
    transformer_block_5 = TransformerBlock(256, n_head, n_ff_dims, rate=0.3)
    transformer_block_6 = TransformerBlock(256, n_head, n_ff_dims, rate=0.3)
    transformer_block_7 = TransformerBlock(256, n_head, n_ff_dims, rate=0.3)
    transformer_block_8 = TransformerBlock(256, n_head, n_ff_dims, rate=0.3)

    x = embedding_layer_1(cblock4[0])
    x = transformer_block_1(x)
    x1 = transformer_block_2(x)
    x = transformer_block_3(x) + x1
    x2 = transformer_block_4(x)
    x = transformer_block_5(x) + x2
    x3 = transformer_block_6(x)
    x = transformer_block_7(x) + x3
    x = transformer_block_8(x)

    # ==================== DECODER ====================
    ublock6 = DecoderSeBlock(x, cblock3[1], kernel_size=kernel_size, n_filters=128, layer_name="dcb_1_")
    ublock7 = DecoderSeBlock(ublock6, cblock2[1], kernel_size=kernel_size, n_filters=128, layer_name="dcb_2_")
    ublock8 = DecoderSeBlock(ublock7, cblock1[1], kernel_size=kernel_size, n_filters=128, layer_name="dcb_3_")

    # ==================== SE BLOCK ====================
    conv9_0 = Conv1D(filters=128, kernel_size=kernel_size, activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.L2(1e-4))(ublock8)
    x = SqueezeExcitation1DLayer(out_dim=128, ratio=2, layer_name='se_0')(conv9_0)
    conv9_1 = Conv1D(filters=64, kernel_size=kernel_size, activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.L2(1e-4))(x)
    x = SqueezeExcitation1DLayer(out_dim=64, ratio=2, layer_name='se_1')(conv9_1)
    conv9_2 = Conv1D(filters=32, kernel_size=kernel_size, activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.L2(1e-4))(x)
    x = SqueezeExcitation1DLayer(out_dim=32, ratio=2, layer_name='se_2')(conv9_2)

    # ==================== MLP ====================
    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=tf.keras.regularizers.L2(1e-4))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    x = Dense(128, kernel_regularizer=tf.keras.regularizers.L2(1e-4))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.L2(1e-4))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, kernel_regularizer=tf.keras.regularizers.L2(1e-4))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L2(1e-4))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
