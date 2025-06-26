import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

train = pd.read_csv('https://raw.githubusercontent.com/nyb05/iitg_sa_1/refs/heads/main/Train_Data.csv')
test = pd.read_csv('https://raw.githubusercontent.com/nyb05/iitg_sa_1/refs/heads/main/Test_Data.csv')

train['age_group'] = train['age_group'].map({'Adult': 0, 'Senior': 1})
train = train.dropna(subset=['age_group'])

train = train.drop(columns=['SEQN'])
test_ids = test['SEQN']  
test = test.drop(columns=['SEQN'])


if 'age_group' in test.columns:
    test = test.drop(columns=['age_group'])

X = train.drop(columns=['age_group'])
y = train['age_group'].astype(int)


imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(test)

X_tr, X_val, y_tr, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_tr, y_tr)

y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_acc:.4f}')

model.fit(X_imputed, y)

test_pred = model.predict(X_test_imputed)

submission = pd.DataFrame({'age_group': test_pred})
submission.index.name = 'index'
submission.to_csv('age_group_submission.csv')

print('Submission file "age_group_submission.csv" created.')

from google.colab import files
files.download('age_group_submission.csv')