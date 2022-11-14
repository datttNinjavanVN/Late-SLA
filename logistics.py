import pandas as pd
import pyarrow
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


df = pd.read_parquet("processed.pq")

model_features = list(set(df.columns).difference({"not_late"}))
target = ["not_late"]

X = df[model_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

pred_label = LogisticRegression(C = 100, solver = 'saga').fit(X_train, y_train).predict(X_test)

print(classification_report(y_test, pred_label))
print(roc_auc_score(y_test, pred_label))