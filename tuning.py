import pandas as pd
import pyarrow
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


df = pd.read_parquet("processed_prior.pq")

model_features = list(set(df.columns).difference({"not_late"}))
target = ["not_late"]

X = df[model_features]
y = df[target]

param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 5),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(LogisticRegression(), param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)

best_clf = clf.fit(X,y)

print(best_clf.best_estimator_)