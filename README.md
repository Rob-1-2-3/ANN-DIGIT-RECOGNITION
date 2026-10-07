# Handwritten Digit Recognition with a From-Scratch ANN

This project uses a one-hidden-layer artificial neural network implemented with
NumPy to classify MNIST handwritten digits. The notebook includes baseline
training, learning-rate and batch-size experiments, and a check of 10 labeled
test images. It does not use a neural-network training library.

## Project files

- `ANN.ipynb` — notebook containing the ANN implementation and experiments.
- `Data/mnist_test.csv` — labeled MNIST test samples.
- `Data/mnist_train.csv` — required training data; excluded from this repository
  because it exceeds GitHub's 100 MiB file limit.

## Dataset setup

Place the MNIST training CSV at:

```text
Data/mnist_train.csv
```

The CSV must have a `label` column and 784 pixel columns. The notebook loads
`Data/mnist_test.csv` for its 10-image test section. Keep the training CSV out
of Git; `.gitignore` excludes it.

## Run the notebook

Install the notebook dependencies in your Python environment:

```bash
python -m pip install numpy pandas matplotlib jupyter
```

Open `ANN.ipynb` in Jupyter or VS Code and run the cells from top to bottom.
The experiments use fixed random seeds so their model initialization and
sample selection are reproducible.
