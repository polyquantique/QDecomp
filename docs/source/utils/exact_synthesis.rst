Exact Synthesis of Clifford+T Gates
===================================

The ``qdecomp.utils.exact_synthesis`` subpackage provides a set of tools for the exact synthesis of Clifford+T gates in their corresponding sequences.
Decomposable Clifford+T gates are represented as a matrix of the form:

.. math:: 
    
    U = \begin{bmatrix}
    z & -w^{\dagger}\omega^k  \\
    w & z^{\dagger} \omega^k
    \end{bmatrix},

where :math:`z` and :math:`w` are elements in :math:`\mathbb{D}[\omega]`, :math:`\omega = e^{i \frac{\pi}{4}}` and :math:`k \in \{ 0, ..., 7\}`. 


Exact Synthesis Module
----------------------
.. automodule:: qdecomp.utils.exact_synthesis.exact_synthesis
    :members:
    :undoc-members:
    :show-inheritance:

S3 Table Generator Module
-------------------------
.. automodule:: qdecomp.utils.exact_synthesis.s3_generator
    :members:
    :undoc-members:
    :show-inheritance: