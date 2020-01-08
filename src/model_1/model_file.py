import numpy as np
import os
import sys
import keras
import keras.backend as K
from keras import Model, Sequential, layers
from keras.layers import Lambda
import tensorflow as tf
from keras.layers import Input, Embedding, Dot, Reshape, Add

domain_dimesnsions = [10,20,25,30]
num_domains = 4
embed_dim = 16

embedding_layer = []
bias_layer = []

input_layer = Input(
    shape=(num_domains,)
)
# =======================
# Values to be predicted
# =======================
n = num_domains*(num_domains-1)//2
y_true = Input(
    shape=(n,1,)
)
X_ij_max = np.ones([n,1])
# =======================
# Input record
# =======================
split_input_record = Lambda(
    lambda x:
    tf.split(
        x,
        num_or_size_splits=num_domains,
        axis=-1)
    ,
    name='split_layer'
)(input_layer)

for i in range(num_domains):
    emb_i = Embedding(
        input_dim = domain_dimesnsions[i],
        output_dim= embed_dim,
        embeddings_initializer='random_uniform',
    )(split_input_record[i])
    embedding_layer.append(emb_i)

    bias_i = Embedding(
        input_dim = domain_dimesnsions[i],
        output_dim=1,
        input_length=1,
        embeddings_initializer='random_uniform'
    )(split_input_record[i])
    bias_layer.append(bias_i)

y_pred = []

for i in range(num_domains):
    for j in range(i+1,num_domains):
        w_i__w_j = Dot(axes=-1)([
            embedding_layer[i],
            embedding_layer[j]
        ])
        w_i__w_j = Reshape(target_shape=(1,))(w_i__w_j)
        pred_logXij = Add()([w_i__w_j, bias_layer[i],bias_layer[j]])
        pred_logXij = Reshape(target_shape=(1,))(pred_logXij)
        y_pred.append(pred_logXij)

y_pred_stacked = Lambda(
    lambda x:
    tf.stack(
        y_pred,
        axis=1
    ),
    name='stack_layer'
)(y_pred)


def custom_loss_function(
        y_true,
        y_pred
):
    global X_ij_max
    global num_domains
    a = 0.8
    epsilon = 0.00001
    """
    This is GloVe's loss function
    :param y_true: The actual values, in our case the 'observed' X_ij co-occurrence values
    :param y_pred: The predicted (log-)co-occurrences from the model
    :return: The loss associated with this batch
    """

    print(y_pred.shape)
    print(y_true.shape)

    _err = K.square( y_pred - K.log(y_true + epsilon))
    print(_err.shape)
    _scale = K.pow(K.clip(y_true / X_ij_max, 0.0, 1.0), a)

    res = _scale * _err
    print(res.shape)

    return K.sum(
        res,
        axis=-1
    )


model = Model(
    [input_layer,y_true],
    y_pred_stacked
)

model.summary()
model.compile(
    loss = custom_loss_function,
    optimizer='adam'
)

