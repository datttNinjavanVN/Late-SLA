import pandas as pd
import pyarrow
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import re
import lightgbm as lgb
from datetime import timedelta
from typing import Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as Imbpipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

df = pd.read_parquet("processed.pq")

model_features = list(set(df.columns).difference({"not_late"}))
target = ["not_late"]

X = df[model_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=1, stratify=y_train)

pred_label = lgb.LGBMClassifier(random_state=0).fit(X_train, y_train).predict(X_test)

RF_model = CatBoostClassifier(verbose=False, eval_metric='F1', random_state=0).fit(X_train, y_train)

def tunning_threshold(y_valid, y_hat_proba, thresholds):
    """Find optimal threshold F1-Score.

    Args: 
        y_valid: true label.
        y_hat_proba: probability of label.
        thresholds: list of thresholds.

    Return:
        optimal threshold.
    """
    list_scores = []
    for t in thresholds:
        y_hat = (y_hat_proba >= t).astype('int')
        list_scores.append(f1_score(y_valid, y_hat))
    index = np.argmax(list_scores)
    print(f'Threshold = {thresholds[index]:.3f}, F1-Score = {list_scores[index]:.5f}')
    return thresholds[index]

y_valid_hat_RF_proba = RF_model.predict_proba(X_valid)

optimal_threshold_smote = tunning_threshold(
    y_valid, y_valid_hat_RF_proba[:,1], 
    thresholds = np.arange(0, 1, 0.001)
)

y_test_hat_RF_proba = RF_model.predict_proba(X_test)

y_test_hat_RF = (y_test_hat_RF_proba[:,1] >= optimal_threshold_smote).astype(int)
print('OLD/TEST DATA\n')
print(classification_report(y_test, y_test_hat_RF))

print(classification_report(y_test, pred_label))
print(roc_auc_score(y_test, pred_label))