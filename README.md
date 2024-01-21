# Product Display Analysis and Modeling (In Progress)

## Description

Given a dataset of product attributes in different stores, our job is to perform a statistical analysis on the data to discover unerlying patterns, discover different preprocessing techniques, and finally, apply machine learning models to predict the display of a product in a store.

## Dataset

A single csv file contained in the `data\raw` folder.

## Tools

We are developing our models in a jupyter notebook environment, using Python and it's libraries:

- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- MLXtend
- XGBoost
- CatBoost
- Tensorflow
- Keras
- MLFlow
- Flask

## Setup

### For Development

Create a Pytohn virtual environment:

```{bash}
pip install notebook numpy statsmodels matplotlib seaborn pandas scikit-learn mlxtend xgboost catboost tensorflow mlflow
```

```{bash}
python -m venv .venv
source .venv/bin/activate
pip install requirements.txt
```

### For Production
