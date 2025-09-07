Installation
============

The :mod:`QDecomp` package can be installed from source by following the following steps:

1. Clone the repository

.. code-block:: bash

    git clone https://github.com/polyquantique/QDecomp.git
    cd QDecomp

2. (Optional) Create and activate a virtual environment

- Linux / macOS:

    .. code-block:: bash

        python3 -m venv venv
        source venv/bin/activate

- Windows (Command Prompt):

    .. code-block:: bash
        
        python -m venv venv
        venv\Scripts\activate

3. Install the package and dependencies

- Standard installation:

    .. code-block:: bash
        
        pip install .

- Editable (developer) installation:

    .. code-block:: bash

        pip install -r requirements.txt -e .

4. (Optional) Run the tests

.. code-block:: bash

    pip install pytest
    pytest tests
