# interactions-from-fluctuations

This is Python3 package to infer interaction matrices from count data via different inference methods that assume different kind of sampling noise.
The functionalities of the package can be accessed via the python API or via command line. Please refer to the test directory for an example of a basic implementation.

## Installation

To install the package use:
```sh
python setup.py install
```

## Usage

### Python API

You can use the `influ` package in your Python code as follows (example using the F2 method):

```python
from influ.F2 import F2

# Example usage
infer_F2= F2(counts, totcounts)
interaction_matrix = infer_F2.infer()
print(interaction_matrix)

```

## Contributing

If you would like to contribute to the development of this package, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your fork.
5. Create a pull request to the main repository.

## License

This project is licensed under the GNU General Public License. See the `LICENSE` file for more details.