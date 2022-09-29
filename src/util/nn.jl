#This script contains relevant utility functions about neural network,
#in particular, activation functions and their high-order derivatives.

using NNlib: softplus, sigmoid

@reexport using NNlib: softplus, sigmoid

export sigmoid_deri, sigmoid_sec_deri
export softplus_dict4, softplus_tuple4

#todo"Improve the overflow of exponential functions."

"""
The first-order derivative of sigmoid.
"""
sigmoid_deri(x) = exp(-max(x,-200))*sigmoid(x)^2

"""
The second-order derivative of sigmoid.
"""
sigmoid_sec_deri(x) = -exp(-max(x,-200))*(1-exp(-max(x,-200)))*sigmoid(x)^3

"""
A dictionary that contains all derivatives of softplus function
"""
softplus_dict4 = Dict(:activation=>softplus,
    :σderi=>sigmoid,
    :σsec_deri => sigmoid_deri,
    :σth_deri => sigmoid_sec_deri)

softplus_tuple4 = (softplus, sigmoid, sigmoid_deri, sigmoid_sec_deri)
