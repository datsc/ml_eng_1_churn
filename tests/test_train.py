import pytest 
import pandas as pd 
import numpy as np 
from sklearn.ensemble import GradientBoostingClassifier 
import pickle 
import os 
import sys 
print(sys.path)

#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src/")))

#Self test note for Ugur
print('again')
from train import load_data, preprocess#, train, save_model

# Define run parameters
current_path = '../src/'
filen = 'train.py'
full_filename = current_path+filen


#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def test_load_data_returns_dataframe():
    df = load_data("data/churn.csv")
    assert isinstance(df, pd.DataFrame)

def test_load_data_not_empty():
    df = load_data("data/churn.csv")
    assert len(df) > 0


def test_load_data_has_churn_column():
    df = load_data("data/churn.csv")
    assert "Churn" in df.columns

def test_preprocess_drops_customer_id():
    df = preprocess(load_data("data/churn.csv"))
    assert "customerID" not in df.columns

def test_preprocess_no_nulls():
    df = preprocess(load_data("data/churn.csv"))
    assert df.isnull().sum().sum() == 0

def test_preprocess_all_numeric():
    df = preprocess(load_data("data/churn.csv"))
    assert all(df.dtypes != object)

def test_preprocess_churn_is_binary():
    df = preprocess(load_data("data/churn.csv"))
    assert set(df["Churn"].unique()).issubset({0, 1})

#def test_train_returns_model():
 #   df = preprocess(load_data("data/churn.csv"))
  #  model, _, _ = train(df)
   # assert isinstance(model, GradientBoostingClassifier)

#def test_train_test_split_not_empty():
 #   df = preprocess(load_data("data/churn.csv"))
  #  _, X_test, y_test = train(df)
   # assert len(X_test) > 0
    #assert len(y_test) > 0

#def test_save_model_creates_file():
#    df = preprocess(load_data("data/churn.csv"))
 #   model, _, _ = train(df)
  #  path = "models/test_model.pkl"
  #  save_model(model, path=path)
  #  assert os.path.exists(path)
  #  os.remove(path)  # cleanup after test

#def test_saved_model_is_loadable():
 #   df = preprocess(load_data("data/churn.csv"))
  #  model, _, _ = train(df)
   # path = "models/test_model.pkl"
    #save_model(model, path=path)
   # with open(path, "rb") as f:
    #    loaded = pickle.load(f)
    #assert isinstance(loaded, GradientBoostingClassifier)
    #os.remove(path)  # cleanup after test
