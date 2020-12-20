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


def qaoa(weights, x, wires, n_layers=1, circuit_ID = 1):
    """
    1-d Ising-coupling QAOA feature map, according to arXiv1812.11075.

    Example one layer, 4 wires, 2 inputs:

       |0> - R_x(x1) - |^| -------- |_| - R_y(w7)  -
       |0> - R_x(x2) - |_|-|^| ---------- R_y(w8)  -
       |0> - ___H___ ------|_|-|^| ------ R_y(w9)  -
       |0> - ___H___ ----------|_| -|^| - R_y(w10) -

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
                if circuit_ID == 1:
                    qml.RX(x[i], wires=wires[i])
                elif circuit_ID == 2:
                    qml.RY(x[i], wires=wires[i])
            # or a Hadamard
            else:
                qml.Hadamard(wires=wires[i])

        # 1-d nearest neighbour coupling
        if n_wires == 1:
            if circuit_ID == 1:
                qml.RY(weights[l], wires=wires[0])
            elif circuit_ID == 2:
                qml.RX(weights[l], wires=wires[0])
            
        elif n_wires == 2:
            _entanglerZ(weights[l * 3 + 2], wires[0], wires[1])
            # local fields
            for i in range(n_wires):
                if circuit_ID == 1:
                    qml.RY(weights[l * 3 + i], wires=wires[i])
                elif circuit_ID == 2:
                    qml.RX(weights[l * 3 + i], wires=wires[i])
        else:
            for i in range(n_wires):
                if i < n_wires-1:
                    _entanglerZ(weights[l * 2 * n_wires + i], wires[i], wires[i + 1])
                else:
                    # enforce periodic boundary condition
                    _entanglerZ(weights[l * 2 * n_wires + i], wires[i], wires[0])
            # local fields
            for i in range(n_wires):
                if circuit_ID == 1:
                    qml.RY(weights[l * 2 * n_wires + n_wires + i], wires=wires[i])
                elif circuit_ID == 2:
                    qml.RX(weights[l * 2 * n_wires + n_wires + i], wires=wires[i])

    # repeat feature encoding once more at the end
    for i in range(n_wires):
        # Either feed in feature
        if i < len(x):
            if circuit_ID == 1:
                qml.RX(x[i], wires=wires[i])
            elif circuit_ID == 2:
                qml.RY(x[i], wires=wires[i])
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




def shallow_circuit(weights, x, wires, n_layers=1,circuit_ID=1):
    """
    Circuits are designed based on paper arXiv:1905.10876.

    Example one layer, 4 wires, 2 inputs:

       |0> - R_x(x1) - |^| -------- |_| - R_y(w5)  -
       |0> - R_x(x2) - |_|-|^| ---------- R_y(w6)  -
       |0> - ___H___ ------|_|-|^| ------ R_y(w7)  -
       |0> - ___H___ ----------|_| -|^| - R_y(w8) -

    After the last layer, another block of R_x(x_i) rotations is applied.

    :param weights: trainable weights of shape 2*n_layers*n_wires
    :param 1d x: input, len(x) is <= len(wires)
    :param wires: list of wires on which the feature map acts
    :param n_layers: number of repetitions of the first layer
    :param circuit_ID: the ID of the circuit based on 
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
                if circuit_ID == 18 or circuit_ID == 19:
                    qml.RX(x[i], wires=wires[i])
                else:
                    raise ValueError("Wrong circuit_ID: It should be between 1-19, got {}.".format(circuit_ID))
            else:
                qml.Hadamard(wires=wires[i])

        # 1-d nearest neighbour coupling
        if n_wires == 1:
            if circuit_ID == 18 or circuit_ID == 19:
                qml.RZ(weights[l], wires=wires[0])
            
        elif n_wires == 2:
            # local fields
            for i in range(n_wires):
                if circuit_ID == 18 or circuit_ID == 19:
                    qml.RZ(weights[l * 3 + i], wires=wires[i])
                else:
                    raise ValueError("Wrong circuit_ID: It should be between 1-19, got {}.".format(circuit_ID))
            if circuit_ID == 18:
                qml.CRZ(weights[l * 3 + 2], wires=[wires[1], wires[0]])
            elif circuit_ID == 19:
                qml.CRX(weights[l * 3 + 2], wires=[wires[1], wires[0]])
        else:
            # local fields
            for i in range(n_wires):
                if circuit_ID == 18 or circuit_ID == 19:
                    qml.RZ(weights[l * 2 * n_wires + i], wires=wires[i])

            for i in range(n_wires):
                if i == 0:
                    if  circuit_ID == 18:
                        qml.CRZ(weights[l * 2 * n_wires + n_wires + i], wires=[wires[n_wires-1], wires[0]])
                    elif  circuit_ID == 19:
                        qml.CRX(weights[l * 2 * n_wires + n_wires + i], wires=[wires[n_wires-1], wires[0]])
                elif i < n_wires-1:
                    if  circuit_ID == 18:
                        qml.CRZ(weights[l * 2 * n_wires + n_wires + i], wires=[wires[i], wires[i + 1]])
                    elif  circuit_ID == 19:
                        qml.CRX(weights[l * 2 * n_wires + n_wires + i], wires=[wires[i], wires[i + 1]])

    # repeat feature encoding once more at the end
    for i in range(n_wires):
        # Either feed in feature
        if i < len(x):
            if circuit_ID == 18 or circuit_ID == 19:
                qml.RX(x[i], wires=wires[i])
        # or a Hadamard
        else:
            qml.Hadamard(wires=wires[i])