import pandas as pd
import pyarrow
import numpy as np

df = pd.read_parquet("processed.pq")

df.info()

import numpy as np

model_features = list(set(df.columns).difference({"not_late"}))
target = ["not_late"]

X = df[model_features]
y = df[target]

id_pos = np.where(y.values.reshape(-1) == 1)[0]
id_neg = np.where(y.values.reshape(-1) == 0)[0]

np.random.shuffle(id_pos)
np.random.shuffle(id_neg)

# Tập train:
id_train_neg = id_neg[:500] 
id_train_pos = id_pos[:5000]
id_train = np.concatenate((id_train_neg, id_train_pos), axis = 0)

# Tập val:
id_val_neg = id_neg[500:800]
id_val_pos = id_pos[5000:8000]
id_val = np.concatenate((id_val_neg, id_val_pos), axis = 0)

# Tập test:
id_test_neg = id_neg[800:]
id_test_pos = id_pos[8000:]
id_test = np.concatenate((id_test_neg, id_test_pos), axis = 0)

# khởi tạo dataset
data_train = df.iloc[id_train]
data_val = df.iloc[id_val]
data_test = df.iloc[id_test] 

print('data train shape: ', data_train.shape)
print('data val shape: ', data_val.shape)
print('data test shape: ', data_test.shape)

from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

log_reg = LogisticRegression()

log_reg.fit(data_train[model_features], data_train['not_late'])
predictions = log_reg.predict_proba(data_test[model_features])
pred_label = log_reg.predict(data_test[model_features])

print(classification_report(np.array(data_test['not_late']), pred_label))