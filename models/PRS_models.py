from abc import ABC

import tensorflow as tf


# model definition
class myModel(tf.keras.Model):
    def __init__(self, n_phenotype, n_categorical_phenotype_feature: list = None):
        super(myModel, self).__init__()

        if n_categorical_phenotype_feature is None:
            n_categorical_phenotype_feature = []
        self.W = []
        # weight for numerical phenotype
        for i in range(n_phenotype-len(n_categorical_phenotype_feature)):
            self.W.append(tf.Variable(tf.random.normal(shape=(2, 1), dtype=tf.float64), trainable=True))
        # weight for categorical phenotype
        for n in n_categorical_phenotype_feature:
            self.W.append(tf.Variable(tf.random.normal(shape=(2, n), dtype=tf.float64), trainable=True))
        # confidence matrix for every features
        self.M = tf.Variable(tf.random.normal(shape=[2, n_phenotype], dtype=tf.float64), trainable=True)

    def forward(self, numerical_x: list, categorical_x: list):
        outputs = []
        # processing numerical data
        k = 0
        for x in numerical_x:
            out = tf.squeeze(x)
            out = tf.reshape(out, shape=[1, -1])
            out = tf.matmul(self.W[k], out)
            out = tf.sigmoid(out)
            out = tf.matmul(self.M[:, k], out)
            outputs.append(out)
            k += 1
        # processing categorical data
        for x in categorical_x:
            out = tf.squeeze(x)
            if out.ndim == 1:
                out = tf.reshape(x, shape=[-1, 1])
            out = tf.matmul(self.W[k], tf.transpose(out))
            out = tf.sigmoid(out)
            out = tf.matmul(self.M[:, k], out)
            outputs.append(out)
            k += 1

        f = tf.reduce_sum(outputs, axis=0)
        f = tf.exp(f)
        f_sum = tf.reduce_sum(f, 0)
        prob = f / f_sum
        return prob


# Logistic model definition
class PRS_model(tf.keras.Model, ABC):
    def __init__(self):
        super(PRS_model, self).__init__()

        self.w = tf.Variable(tf.random.normal(shape=[1]))
        self.b = tf.Variable(tf.random.normal(shape=[1]))

    def forward(self, score):
        logit = 1 / (1 + tf.exp(-(self.w * score + self.b)))
        return logit
