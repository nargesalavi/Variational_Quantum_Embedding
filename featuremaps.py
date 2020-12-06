"""
Feature maps
************

This module contains feature maps. Each feature map function
takes an input vector x and weights, and constructs a circuit that maps
these two to a quantum state. The feature map function can be called in a qnode.

A feature map has the following positional arguments: weights, x, wires. It can have optional
keyword arguments.

Each feature map comes with a function that generates initial parameters
for that particular feature map.
"""
import numpy as np
import pennylane as qml


def _entanglerZ(w_, w1, w2):
    qml.CNOT(wires=[w2, w1])
    qml.RZ(2*w_, wires=w1)
    qml.CNOT(wires=[w2, w1])


def qaoa(weights, x, wires, n_layers=1):
    """
    1-d Ising-coupling QAOA feature map, according to arXiv1812.11075.

    Example one layer, 4 wires, 2 inputs:

       |0> - R_x(x1) - |^| -------- |_| - R_y(w7)  -
       |0> - R_x(x2) - |_|-|^| ---------- R_y(w8)  -
       |0> - R_x(w1) ------|_|-|^| ------ R_y(w9)  -
       |0> - R_x(w2) ----------|_| -|^| - R_y(w10) -

    After the last layer, another block of R_x(x_i) rotations is applied.

    :param weights: trainable weights of shape 2*n_layers*n_wires
    :param 1d x: input, len(x) is <= len(wires)
    :param wires: list of wires on which the feature map acts
    :param n_layers: number of repetitions of the first layer
    """
    n_wires = len(wires)

    if n_wires == 1:
        n_weights_needed = n_layers
    elif n_wires == 2:
        n_weights_needed = 3 * n_layers
    else:
        n_weights_needed = 2 * n_wires * n_layers

    if len(x) > n_wires:
        raise ValueError("Feat map can encode at most {} features (which is the "
                         "number of wires), got {}.".format(n_wires, len(x)))

    if len(weights) != n_weights_needed:
        raise ValueError("Feat map needs {} weights, got {}."
                         .format(n_weights_needed, len(weights)))

    for l in range(n_layers):

        # inputs
        for i in range(n_wires):
            # Either feed in feature
            if i < len(x):
                qml.RX(x[i], wires=wires[i])
            # or a Hadamard
            else:
                qml.Hadamard(wires=wires[i])

        # 1-d nearest neighbour coupling
        if n_wires == 1:
            qml.RY(weights[l], wires=wires[0])
        elif n_wires == 2:
            _entanglerZ(weights[l * 3 + 2], wires[0], wires[1])
            # local fields
            for i in range(n_wires):
                qml.RY(weights[l * 3 + i], wires=wires[i])
        else:
            for i in range(n_wires):
                if i < n_wires-1:
                    _entanglerZ(weights[l * 2 * n_wires + i], wires[i], wires[i + 1])
                else:
                    # enforce periodic boundary condition
                    _entanglerZ(weights[l * 2 * n_wires + i], wires[i], wires[0])
            # local fields
            for i in range(n_wires):
                qml.RY(weights[l * 2 * n_wires + n_wires + i], wires=wires[i])

    # repeat feature encoding once more at the end
    for i in range(n_wires):
        # Either feed in feature
        if i < len(x):
            qml.RX(x[i], wires=wires[i])
        # or a Hadamard
        else:
            qml.Hadamard(wires=wires[i])


def qaoa_largedata(weights, x, wires, n_layers=1):
    """
    QAOA feature map for data with num of features>3
    Keeping 4 wires max and breaking the feature vector 
    so as to apply three features at a time. I am assuming,
    max number of features to be 9.   

    Example one layer, 4 wires, 5 inputs:

       |0> - R_x(x1) - R_x(x4) - |^| -------- |_| -  R_y(w5)  -
       |0> - R_x(x2) - R_x(x5) - |_|-|^| ----------  R_y(w6)  -
       |0> - R_x(x3) -    H    ------|_|-|^| ------  R_y(w7)  -
       |0> -    H    -    H    ----------|_| -|^| -  R_y(w8)  -

    After the last layer, another block of R_x(x_i) rotations is applied.

    :param weights: trainable weights of shape 3*n_layers*n_wires
    :param 1d x: input, len(x) is <= len(wires)
    :param wires: list of wires on which the feature map acts
    :param n_layers: number of repetitions of the first layer
    """

    dim = len(x)
    s = 3
    subx = np.split(x, range(s, dim, s))

    n_wires = len(wires)
    n_weights_needed = 2 * n_wires * n_layers

    if len(weights) != n_weights_needed:
        raise ValueError("Feat map needs {} weights, got {}."
                         .format(n_weights_needed, len(weights)))

    for l in range(n_layers):
        # inputs
        for sub in subx:
            for i in range(n_wires):
                # Either feed in feature
                if i < len(sub):
                    qml.RX(sub[i], wires=wires[i])
                # or a Hadamard
                else:
                    qml.Hadamard(wires=wires[i])

        # 1-d nearest neighbour coupling
        for i in range(n_wires):
            if i < n_wires-1:
                _entanglerZ(weights[l * 2 * n_wires + i], wires[i], wires[i + 1])
            else:
                # enforce periodic boundary condition
                _entanglerZ(weights[l * 2 * n_wires + i], wires[i], wires[0])
        # local fields
        for i in range(n_wires):
            qml.RY(weights[l * 2 * n_wires + n_wires + i], wires=wires[i])

    # repeat feature encoding once more at the end
    for sub in subx:
        for i in range(n_wires):
            # Either feed in feature
            if i < len(sub):
                qml.RX(sub[i], wires=wires[i])
            # or a Hadamard
            else:
                qml.Hadamard(wires=wires[i])


def pars_qaoa(n_wires, n_layers=1):
    """
    Initial weight generator for 1-d qaoa feature map
    :param n_wires: number of wires
    :param n_layers: number of layers
    :return: array of weights
    """
    if n_wires == 1:
        return 0.001*np.ones(n_layers)
    elif n_wires == 2:
        return 0.001 * np.ones(n_layers * 3)
    elif n_wires == 4:
        return 0.001 * np.ones(n_wires * n_layers * 2)
    return 0.001*np.ones(n_layers * n_wires * 2)
