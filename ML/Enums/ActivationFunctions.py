from enum import Enum
from .BaseEnum import BaseEnum

class ActivationFunctions(BaseEnum):
    # All related information found on https://keras.io/api/layers/activations/

    # linear unit activation function
    relu = 1

    # sigmoid function sigmoid(value) = 1 / (1 + exp(-value))
    sigmoid = 2

    # soft max conver values to probability distubition, output vetor range between 0 and 1
    # exp(value) / tf.reduce_sum(exp(value))
    softmax = 3

    # softplus(value) = log(exp(value) + 1)
    softplus = 4

    # softsign(value)
    softsign = 5

    # Hyperbolic activation function
    #  tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))
    tanh = 6

    #  Scaled exponential linear unit
    # if x < 0, return scale * x
    # if x > 0, return scale * alpha * (exp(x) - 1)
    # where alpha and scale are constant parremeters as  alpha=1.67326324 and scale=1.05070098

    selu = 7

    # exponential linear unit  same as selu but not scaled
    # alpha > 0
    # if x > 0, alpha * (expo(x) - 1)
    #  if x < 0 ELU parameter alpha controls value to make negative inputs
    elu = 8

    # exponential actiation function
    # simpl value = exp(x)
    exponential = 9
