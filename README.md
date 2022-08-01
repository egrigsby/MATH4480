# MATH4480

Test and utility functions for the practical and final projects.

The modules can be imported by running e.g.

```
import os

if not os.path.isdir('MATH4480'):
    !git clone https://github.com/m-a-huber/MATH4480.git

from MATH4480.final_utils import final_utils
```

---

The module `final_utils` (containing the code for the final project) contains the following functions to load a cleaned version of the respective dataset:

- `load_breastcancer()`; see [description](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29).
- `load_pokerhands()`; see [description](https://archive.ics.uci.edu/ml/datasets/Poker+Hand).
- `load_spambase()`; see [description](https://archive.ics.uci.edu/ml/datasets/Spambase).
- `load_iris()`; see [description](https://archive.ics.uci.edu/ml/datasets/Iris).
- `load_parkinsons()`; see [description](https://archive.ics.uci.edu/ml/datasets/parkinsons).
- `load_white_wine()`; see [description](https://archive.ics.uci.edu/ml/datasets/wine+quality).
- `load_auto_mpg()`; see [description](https://archive.ics.uci.edu/ml/datasets/auto+mpg).
- `load_fashion_mnist()`; see [description](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data).
- `load_cifar10()`; see [description](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data).
- `load_horses_or_humans()`; see [description](https://www.tensorflow.org/datasets/catalog/horses_or_humans).
- `load_rock_paper_scissors()`; see [description](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors).

**Note:** The iris-dataset as well as the auto-mpg-dataset may or may not require some further preprocessing such as one-hot-encoding some features and/or labels.
Moreover, the wine-dataset contains _only the white wine data_.

---

The first seven of the above commands load non-image datasets, while the last four load image datasets.

The non-image dataset commands each return a triple of the format `(df, x, y)`, where `df` is a pandas dataframe containing the dataset, and `x` and `y` are NumPy-arrays containing the features and labels, respectively.
The dataframe `df` is arranged so that the last column contains the label `y`.
For a description of the datasets, refer to the respective links above.

The image dataset commands each return a quadruple of the format `x_train, y_train, x_test, y_test`, which are NumPy-arrays representing the training and test sets of features and labels, respectively.
For either of the arrays of `x_train` and `x_test`, one may call e.g. `plot_some_examples(x_train)` in order to display some of the images contained in that set.