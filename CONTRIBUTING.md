# Releasing a new version of QDecomp

When releasing a new version of QDecomp, make sure to follow the following steps.

### 1. Install the requirements

```bash
python -m pip install --upgrade pip
pip install build twine
```

### 2. Choose a new version number

Create a new version number following the **Semantic Versioning** (SemVer) style:
```md
MAJOR.MINOR.PATCH
```
where
- **PATCH**: bug fixes
- **MINOR**: new features, backward-compatible
- **MAJOR**: breaking changes


### 3. Update the version in the repository

The version of the package is defined in the global [\_\_init\_\_.py](src/qdecomp/__init__.py) file of the package:
```python
__version__ = "1.0.0"
```

Also, make sure to update the version displayed in the [README](README.md) file and in the [introduction](docs/source/introduction.rst) of the documentation.


### 4. Update the documentation

If new features were added to the package, update the **API documentation** in `\docs\source`.

### 5. Tests and formatting

Before creating the release, make sure that all **tests** are passing by running
```bash
pytest
```
in the root directory. Also make sure the repository has proper **formatting** by running 
```bash
isort .
black -l 100 .
```
from the root directory.

### 6. Create a Github release

When all modifications are done, **commit** and **push** all the modifications. Next, create and push a **Git tag** using
```bash
git tag v1.2.3
git push origin v1.2.3
```
where the version number `vX.Y.Z` corresponds to the one chosen in [step 2](#2-decide-a-new-version-number). This process can also be performed from the GitHub web UI.

Finally, in Github, create a new **draft release**. Select the created **tag**, and write a **changelog** with the description of the release. Publish the release on Github. 

### 7. Build the PyPI distributions

In the root directory, **build** the wheel and the source distribution using
```bash
rm -rf dist/
python -m build
```

This will create the following files.
```md
dist/
    qdecomp-1.2.3-py3-none-any.whl
    qdecomp-1.2.3.tar.gz
```

You can verify that the build is correct by installing the package locally in a virtual environnement.
```bash
python -m venv temp_env
source /temp_env/bin/activate
pip install dist/qdecomp-1.2.3-py3-none-any.whl
```

### 8. Upload to PyPI

It is first recommended to upload it on **TestPyPI**:
```bash
twine upload --repository testpypi dist/*
```

You can then install it from **TestPyPI** and verify that everything works fine using 
```bash
pip install --index-url https://test.pypi.org/simple qdecomp
```

Finally, upload it on **PyPI**.
```bash
twine upload dist/*
```

Be **careful**, the uploaded version is definitive and cannot be modified. To apply changes, a new version of the package must be uploaded.

### 9. Build the documentation

Go on *Read the Docs* and build the latest documentation. 