# wavetrainer

<a href="https://pypi.org/project/wavetrainer/">
    <img alt="PyPi" src="https://img.shields.io/pypi/v/wavetrainer">
</a>

A library for automatically finding the optimal model within feature and hyperparameter space on time series models.

<p align="center">
    <img src="wavetrain.png" alt="wavetrain" width="200"/>
</p>

## Dependencies :globe_with_meridians:

Python 3.11.6:

- [pandas](https://pandas.pydata.org/)
- [optuna](https://optuna.readthedocs.io/en/stable/)
- [scikit-learn](https://scikit-learn.org/)
- [feature-engine](https://feature-engine.trainindata.com/en/latest/)
- [tqdm](https://github.com/tqdm/tqdm)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [catboost](https://catboost.ai/)
- [venn-abers](https://github.com/ip200/venn-abers)
- [pytz](https://pythonhosted.org/pytz/)
- [torch](https://pytorch.org/)
- [tabpfn](https://github.com/PriorLabs/TabPFN)
- [pytest-is-running](https://github.com/adamchainz/pytest-is-running)
- [xgboost](https://xgboost.readthedocs.io/en/release_3.0.0/)
- [jax](https://github.com/jax-ml/jax)
- [tabpfn_extensions](https://github.com/PriorLabs/tabpfn-extensions)
- [hyperopt](https://github.com/hyperopt/hyperopt)
- [pycaleva](https://github.com/MartinWeigl/pycaleva)
- [lightgbm](https://github.com/microsoft/LightGBM)
- [kaleido](https://github.com/plotly/Kaleido)
- [memory-profiler](https://github.com/pythonprofilers/memory_profiler)
- [matplotlib](https://matplotlib.org/)
- [pyarrow](https://arrow.apache.org/docs/python/index.html)

## Raison D'être :thought_balloon:

`wavetrainer` aims to split out the various aspects of creating a good model into different composable pieces and searches the space of these different pieces to find an optimal model. This came about after doing code like this multiple times on multiple projects. This is specifically geared towards time series models, validating itself through walk-forward analysis.

## Architecture :triangular_ruler:

`wavetrainer` is an object orientated library. The entities are organised like so:

* **Trainer**: A sklearn compatible object that can fit and predict data.
    * **Reducer**: An object that can reduce the feature space based on heuristics.
    * **Weights**: An object that adds weights to the features.
    * **Selector**: An object that can select which features to include from the training set.
    * **Calibrator**: An object that can calibrate the probabilities produced by the model.
    * **Model**: An object that represents the underlying model architecture being used.
    * **Windower**: An object that represents the lookback window of the data.

## Installation :inbox_tray:

This is a python package hosted on pypi, so to install simply run the following command:

`pip install wavetrainer`

or install using this local repository:

`python setup.py install --old-and-unmanageable`

## Usage example :eyes:

The use of `wavetrainer` is entirely through code due to it being a library. It attempts to hide most of its complexity from the user, so it only has a few functions of relevance in its outward API.

### Training

To train a model:

```python
import wavetrainer as wt
import pandas as pd
import numpy as np
import random

data_size = 10
df = pd.DataFrame(
    np.random.randint(0, 30, size=data_size),
    columns=["X"],
    index=pd.date_range("20180101", periods=data_size),
)
df["Y"] = [random.choice([True, False]) for _ in range(data_size)]

X = df["X"]
Y = df["Y"]

wavetrainer = wt.create("my_wavetrain")
wavetrainer = wavetrainer.fit(X, y=Y)
```

This will save it to the folder `my_wavetrain`.

### Load

To load a trainer (as well as its composite states):

```python
import wavetrainer as wt

wavetrainer = wt.load("my_wavetrain")
```

### Predict

To make a prediction from new data:

```python
import wavetrainer as wt
import pandas as pd
import numpy as np

wavetrainer = wt.load("my_wavetrain")
data_size = 1
df = pd.DataFrame(
    np.random.randint(0, 30, size=data_size),
    columns=["X"],
    index=pd.date_range("20180101", periods=data_size),
)
X = df["X"]

preds = wavetrainer.predict(X)
```

`preds` will now contain both the predictions and the probabilities associated with those predictions.

## License :memo:

The project is available under the [MIT License](LICENSE).
