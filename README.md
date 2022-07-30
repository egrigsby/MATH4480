# MATH4480

Test and utility functions for the practical projects.

The module `final_utils` (containing the code for the final project) contains the following functions to load a cleaned version of the respective dataset:

- `load_breastcancer()`
- `load_pokerhands()`
- `load_spambase()`
- `load_iris()`
- `load_parkinsons()`
- `load_white_wine()`
- `load_auto_mpg()`
- `load_fashion_mnist()`
- `load_cifar10()`
- `load_horses_or_humans()`
- `load_rock_paper_scissors()`

The first seven commands load non-image datasets, while the last four load image datasets.

The non-image dataset commands each return a triple of the format `(df, x, y)`, where `df` is a pandas dataframe containing the dataset, and `x` and `y` are NumPy-arrays containing the features and labels, respectively.

The image dataset commands each return a quadruple of the format `x_train, y_train, x_test, y_test`, which are NumPy-arrays representing the training and test sets of features and labels, respectively.
For either of the arrays of `x_train` and `x_test`, one may call e.g. `plot_some_examples(x_train)` in order to display some of the images contained in that set.

**Note:** The iris-dataset as well as the auto-mpg-dataset may or may not require some further preprocessing such as one-hot-encoding some features and/or labels.