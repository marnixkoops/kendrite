# 🧠 **kendrite**

A framework for attentive explainable deep learning on tabular data

### 💨 Quick start

```bash
kedro run
```

### 💡 Concept

Many QB projects boil down to a tabular learning problem where the goal is learning a function to predict a target given some input data. A significant amount of time, effort, and discussion are invested in understanding, mapping, and hand-crafting a large set of useful features to train a model. 

Neural networks can automate feature learning, greatly alleviating this effort and reducing time to impact. We have built a modular and ready-to-go pipeline that enables rapid testing with a neural approach for regression and classification problems without costly feature engineering.

### 🧱 Built upon

| Technology | Description                                                                                                               | Links                                                                                               |
|------------|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `kedro`    | Python framework for creating reproducible, maintainable and modular machine learning pipelines                           | [github](https://github.com/quantumblacklabs/kedro) [docs](https://kedro.readthedocs.io/en/stable/) |
| `tabnet`   | Interpretable `pytorch` deep learning architecture for modeling tabular data                                              | [arxiv](https://arxiv.org/abs/1908.07442) [github](https://github.com/dreamquark-ai/tabnet)         |
| `mlflow`   | Platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry | [github](https://github.com/mlflow/mlflow) [docs](https://mlflow.org/docs/latest/index.html)        |
| `ray[tune]`   |     Package for distributed hyper-parameter tuning. | [github](https://github.com/ray-project/ray) [docs](https://docs.ray.io/en/latest/tune/index.html)        |

### 🕸️ TabNet

Introduced by Sercan Arık, Tomas Pfister from Google Cloud AI in 2019.
Deep neural network architecture for regression and classification problems.

1. TabNet inputs raw tabular data without any preprocessing and is trained using gradient descent-based optimization.
2. TabNet uses sequential attention to choose which features to use. This enables both local and global explainability of the model within a single architecture.
3. TabNet outperforms or is on par with other tabular learning models such as XGBoost.
4. Finally, for the first time for tabular data, they show significant performance improvements by self-supervised learning.

### 🌊 MLflow

* Track pipeline runs, parameters, metrics and other results
* Enables, collaborative and reproducible experimentation in a team

### 🌀 Hyperparameter Tuning

* Built in `ray` pipeline to automate hyperparameter tuning with `optuna` (distributed)
* Trails are logged and can be investigated through `tensorboard`
