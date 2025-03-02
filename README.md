# Qudecomp
``Qucomp`` is a standalone software package for decomposing any quantum circuit into Clifford+T quantum gates.

## Installation

To install the CliffordPlusT Compiler, clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/CliffordPlusT.git
pip install -r requirements.txt
```

## Usage

The package takes as an input a specific data structure composed of tupples 
and strings representing the quantum circuit. Each gate should be specified in the order they appear in the circuit, followed by a tuple containing the two target qubits and finally a desired error $\varepsilon$. The input circuit should follow the structure showed below.

$$
\left[(Gate_1; qubits_{G_1}; \varepsilon_1), (Gate_2; qubits_{G_2}; \varepsilon_1), (Gate_3; qubits_{G_3}; \varepsilon_1), ... \right],
$$

For a 2 qubit circuit composed of an X gate on the first qubit followed by a CNOT, here is how the input circuit should be structured

```python
circuit = ((np.array([0,1],[1,0]), (1,1), 1e-3), np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]]), (1,2)). 
```
The decomp function will output a data structure similar to the one provided as input, but with gates replaced with their corresponding H,T sequences as shown below

$$ \left[(HTTTHTH; (1,1); \varepsilon_1), (CNOT; (2,1); 0), (HTHTTHT; (2,2); \varepsilon_3), ...\right]$$

Here is an example of how to use the CliffordPlusT Compiler:
```python
from cliffordplust import decomp

# Your quantum circuit code here
circuit = ...

compiled_circuit = decomp(circuit)
print(compiled_circuit)
```

## Contributing

Need to see whith Theodore

## License

Released under the Apache License 2.0. See LICENSE file.

## Contact

If you have any questions or feedback, please open an issue or contact us at
* [francis.blais@polymtl.ca](francis.blais@polymtl.ca). 
* [marius.trudeau@polymtl.ca](marius.trudeau@polymtl.ca). 
* [olivier.romain@polymtl.ca](olivier.romain@polymtl.ca). 
* [vincent-2.girouard@polymtl.ca](vincent-2.girouard@polymtl.ca). 

## Acknowledgements

Need to add Ross and Sellinger article
