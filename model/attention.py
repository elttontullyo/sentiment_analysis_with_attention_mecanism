import tensorflow as tf
from tensorflow.keras.layers import Layer

# The Attention Mechanism is a key component used in various deep learning models, especially in natural language processing (NLP) tasks, 
# to help the model focus on relevant parts of the input data while making predictions. 
# It was first introduced in the context of machine translation tasks and has since become popular in many other sequence-to-sequence tasks.
# The basic idea behind the Attention Mechanism is to compute a set of attention weights 
# that indicate the importance of each input element (e.g., words in a sentence) concerning the output at a given time step. 
# These attention weights are used to compute a weighted sum of the input elements, 
# which acts as a context vector that helps the model make predictions.
# In this implementation, we create an Attention class that inherits from tensorflow.keras.layers.Layer. 
# The class contains the necessary components for the attention mechanism:
#  the query, key, and value matrices (represented by W_query, W_key, and V, respectively).
# The build method is responsible for creating and initializing the learnable weights. 
# The call method performs the actual attention computation, including calculating the attention scores and weights, 
# and computing the context vector as a weighted sum of the inputs.

# Clear all previously registered custom objects
tf.keras.saving.get_custom_objects().clear()
# Upon registration, you can optionally specify a package or a name.
@tf.keras.saving.register_keras_serializable(package="AttentionMecanism")
class AttentionMecanism(Layer):
    def __init__(self, **kwargs):
        super(AttentionMecanism, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_query = self.add_weight(name='W_query',
                                       shape=(input_shape[-1], input_shape[-1]),
                                       initializer='glorot_uniform',
                                       trainable=True)
        self.W_key = self.add_weight(name='W_key',
                                     shape=(input_shape[-1], input_shape[-1]),
                                     initializer='glorot_uniform',
                                     trainable=True)
        self.V = self.add_weight(name='V',
                                 shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(AttentionMecanism, self).build(input_shape)

    def call(self, inputs):
        query = tf.matmul(inputs, self.W_query)
        key = tf.matmul(inputs, self.W_key)

        score = tf.matmul(tf.nn.tanh(query + key), self.V)
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1]) 

    def get_config(self):
        return super(AttentionMecanism,self).get_config() 