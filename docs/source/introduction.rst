Introduction
============

Overview
--------

:mod:`QDecomp` is a standalone software package to perform multiple decompositions of single-qubit and two-qubit quantum gates.

The package primarily focuses on decomposing gates into the Clifford+T universal subset by implementing the algorithm proposed by Ross and Selinger :cite:`intro_ross`.

The package contains 4 main subpackages:

* :mod:`decompositions` : Proposes user-oriented functions for decomposing various quantum gates
* :mod:`utils` : Contains the core algorithms and additional utility functions
* :mod:`rings` : Implements classes for symbolic calculations in mathematical rings
* :mod:`plot` : Offers visualization tools of core parts of the algorithm used mainly for debugging

Below is a figure illustrating the subpackages (green) and their associated modules (yellow) and classes (orange). 

.. figure:: _static/package_structure.svg
   :alt: package_structure
   :width: 600px
   :align: center

   :mod:`QDecomp` package structure

Documentation
-------------

The complete API documentation is available and can be built locally using Sphinx:

.. code-block:: bash

    cd docs
    make html
    ./build/html/index.html

The documentation is generated in :file:`docs/build/html/`. Open :file:`docs/build/html/index.html` in a browser to view it.

License
-------

Released under the Apache License 2.0.

Collaborations
--------------

This package was made in collaboration with D-Wave and Polytechnique Montréal.

Citing This Package
-------------------

If you use :mod:`QDecomp` in your research or projects, please cite it using the following BibTeX entry:

.. code-block:: bibtex

    @software{qdecomp,
      author = {Romain, Olivier and Girouard, Vincent and Trudeau, Marius and Blais, Francis},
      title = {QDecomp},
      year = {2025},
      version = {1.0.0},
      url = {https://github.com/polyquantique/QDecomp}
    }

References
----------

.. bibliography::
   :filter: cited and docname in docnames
   :keyprefix: intro_

   exact_synthesis
   crooks
