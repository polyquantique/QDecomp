Tutorial
========

This tutorial provides a step-by-step guide to using the :mod:`QDecomp` package for quantum gate decomposition.

Single-Qubit Gate (SQG) Decompositions
--------------------------------------

SQG Decomposition Using :func:`qdecomp.decompositions.sqg_decomp`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the simplest decomposition the :mod:`QDecomp` package can perform is the decomposition of single-qubit gates into the Clifford+T universal subset.
This decomposition is carried out using the :func:`qdecomp.decompositions.sqg_decomp` function.
It takes as inputs the matrix representation of the gate and the desired tolerance (epsilon) for the decomposition.
It returns a sequence of Clifford+T gates and the global phase factor (alpha) that represents the single-qubit gate.

.. code-block:: python

    from qdecomp.decompositions import sqg_decomp
    from scipy.stats import unitary_group

    # Generate a random single-qubit gate (2x2 unitary matrix)
    sqg = unitary_group.rvs(2, random_state=42)

    # Decompose the single-qubit gate into Clifford+T gates with a tolerance of 0.001
    sequence, alpha = sqg_decomp(sqg, epsilon=0.001)

    # Print the resulting sequence of gates
    print(sequence)  # T H S T H S T [...] Z T H Z S H S

The decomposition carried out by :func:`qdecomp.decompositions.sqg_decomp` is done in two steps:

1. Decomposing the single-qubit gate into a sequence of ZYZ gates using :func:`qdecomp.decompositions.zyz_decomp`.
2. Decomposing each ZYZ gate into Clifford+T gates using :func:`qdecomp.decompositions.rz_decomp`.

The next subsections present a tutorial on how to use these two functions.


Partial SQG Decomposition Using :func:`qdecomp.decompositions.zyz_decomp`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any SQG can be decomposed as :math:`\alpha R_z(\theta_2) R_y(\theta_1) R_z(\theta_0)` where :math:`\alpha` is a global phase factor and :math:`\theta_i` are angles.
This can be done using the :func:`qdecomp.decompositions.zyz_decomp` function which takes a single-qubit gate (2x2 unitary matrix) as input.
It returns a tuple containing the three angles of the ZYZ decomposition and a global phase factor (alpha).

.. code-block:: python

    from qdecomp.decompositions import zyz_decomp
    from scipy.stats import unitary_group

    # Generate a random single-qubit gate (2x2 unitary matrix)
    sqg = unitary_group.rvs(2, random_state=42)

    # Decompose the single-qubit gate into ZYZ angles and global phase factor
    zyz_result = zyz_decomp(sqg)
    angles = zyz_result[:3]
    alpha = zyz_result[3]

    # Print the angles and global phase factor
    print("Angles:", angles)  # Angles: [-0.200, 2.519, 1.622]
    print("Global phase factor:", alpha)  # Global phase factor: 0.271...


Rz Decomposition Using :func:`qdecomp.decompositions.rz_decomp`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`qdecomp.decompositions.rz_decomp` function decomposes a rotation around the z axis into a sequence of Clifford+T gates.
As input, it takes the angle of rotation and a tolerance (epsilon) for the decomposition.
It returns a sequence of gates as a string.

.. code-block:: python
    
    from qdecomp.decompositions import rz_decomp

    # Define the angle of rotation and tolerance
    angle = 0.4  # Example angle in radians
    epsilon = 0.001  # Tolerance for decomposition

    # Decompose the rotation around the z axis into Clifford+T gates
    sequence = rz_decomp(angle=angle, epsilon=epsilon)

    # Print the resulting sequence of gates
    print(sequence)  # S T H S T H S [...] S H S T H


Two-Qubit Gate (TQG) Decompositions
-----------------------------------

The :mod:`QDecomp` package can also perform two-qubit gate (TQG) decompositions into a series of Clifford+T gates.
The decomposition is carried out using the :func:`qdecomp.decompositions.tqg_decomp` function.
It takes as inputs the matrix representation of the gate and the desired tolerance (epsilon) for the decomposition.
It returns a list of :class:`QGate` objects, each one containing the Clifford+T decomposition sequence and the qubit(s) on which it applies.

.. code-block:: python

    from qdecomp.decompositions import tqg_decomp
    from scipy.stats import unitary_group

    # Generate a random two-qubit gate (4x4 unitary matrix)
    tqg = unitary_group.rvs(4, random_state=42)

    # Decompose the two-qubit gate into Clifford+T gates with a tolerance of 0.001
    circuit = tqg_decomp(tqg, epsilon=0.001)

    # Print gates in the circuit
    for gate in circuit:
        print(f"Target: {gate.target} -  Sequence: {gate.sequence}")
    # Target: (0,) - Sequence: Z H T H T H T H T H S [...]
    # Target: (1,) - Sequence: S T H T H S T H T H T [...]
    # Target: (0, 1) - Sequence: CNOT1
    # [...]
    # Target: (1,) - Sequence: H T H S T H S T H S T [...]
