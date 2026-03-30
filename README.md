# ml_eng_1_churn
TEST PROJECT WITH DVC

Author: Ugur Ural
This is a project done together with Claude as a speed and accuracy test.


Data comes from Kaggle: 
https://www.kaggle.com/datasets/blastchar/telco-customer-churn


# TODO LIST

- I only run small tests in test_train.py so far. Check and run the others.
 
TO RUN
######


conda activate tinyenv
python src/train.py
pytest tests/test_train.py -v
python src/evaluate.py
pytest tests/test_evaluate.py -v
