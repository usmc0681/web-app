from __future__ import print_function
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import pickle
import xgboost as xgb


xgb.__version__

# In[2]:


df_train = pd.read_csv("C:/Users/Tim/Desktop/Project/new1/Recidivism_Challenge_Train.csv")
df_test = pd.read_csv("C:/Users/Tim/Desktop/Project/new1/Recidivism_Challenge_Test.csv")

object_cols = [col for col in df_train.columns if df_train[col].dtype == 'object' or df_train[col].dtype == 'bool']

for col in object_cols:
    le = LabelEncoder()
    le.fit(df_train[col])
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

handle_train = df_train.drop(['Recidivism_Within_3years', 'ID'], axis=1)
handle_test = df_test.drop(['ID'], axis=1)

imputer = KNNImputer(n_neighbors=9)
imputer.fit(handle_train)
df_train[handle_train.columns] = imputer.transform(handle_train)
df_test[handle_test.columns] = imputer.transform(handle_test)

feature = df_train.drop(['ID', 'Recidivism_Within_3years'], axis=1)
label = df_train['Recidivism_Within_3years']
test = df_test.drop(['ID'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.30, random_state=42)

model = xgb.XGBClassifier(tree_method = 'gpu_hist', enable_categorical = True)
model.fit(X_train, y_train)

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical= True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical= True)

params_XGB_best = {'lambda': 3.06,
                   'alpha': 4.58,
                   'colsample_bytree': 0.9,
                   'subsample': 0.95,
                   'learning_rate': 0.01,
                   'n_estimators': 800,
                   'max_depth': 5,
                   'min_child_weight': 1,
                   'num_parallel_tree': 1}

pred_XGB_best = (xgb.XGBClassifier(**params_XGB_best).fit(X_train, y_train))

df_test['Recidivism_Within_3years'] = model.predict(test)
df_test['Recidivism_Within_3years'] = df_test['Recidivism_Within_3years']
df_test['Recidivism_Within_3years'].replace(1, True, inplace=True)
df_test['Recidivism_Within_3years'].replace(0, False, inplace=True)
df = df_test[['ID', 'Recidivism_Within_3years']]
df.set_index(['ID', 'Recidivism_Within_3years'], inplace=True)
df.to_csv('submission.csv')
df.head(10)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print('XGBoost model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

pickle.dump(model, open("recidivism.pkl", "wb"))
pred_XGB_best.save_model("recidivism.json")
