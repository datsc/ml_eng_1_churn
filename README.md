# ml_eng_1_churn
TEST PROJECT WITH DVC

Author: Ugur Ural
This is a project done together with Claude as a speed and accuracy test.


Data comes from Kaggle: 
https://www.kaggle.com/datasets/blastchar/telco-customer-churn


# TODO LIST

 
TO RUN
######


conda activate tinyenv
python src/train.py
pytest tests/test_train.py -v
python src/evaluate.py
pytest tests/test_evaluate.py -v

TO REWRITE
##########

Git repo
Data
Train.py
Evaluate.py
DVC
mlflow
Docker on github codespaces
