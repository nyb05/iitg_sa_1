import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

train_df = pd.read_csv("hacktrain.csv")
test_df  = pd.read_csv("hacktest.csv")
ts_cols = [c for c in train_df.columns if c.endswith("_N")]

X_raw = train_df[ts_cols].values
y_raw = train_df["class"].values
X_test_raw = test_df[ts_cols].values
test_ids    = test_df["ID"].values

le = LabelEncoder().fit(y_raw)
y = le.transform(y_raw)

mins = np.nanmin(X_raw, axis=0)
maxs = np.nanmax(X_raw, axis=0)
def minmax(X): return (X - mins) / (maxs - mins + 1e-8)
X = minmax(X_raw)
X_test = minmax(X_test_raw)

class NDVIPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, window=7, polyorder=2):
        self.window = window
        self.polyorder = polyorder

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        Xi = df.interpolate(axis=1, limit_direction="both")
        Xi = Xi.fillna(method="bfill", axis=1).fillna(method="ffill", axis=1).values
        w = min(self.window, Xi.shape[1] // 2 * 2 + 1)
        if w < 3: w = 3
        Xs = savgol_filter(Xi, window_length=w, polyorder=self.polyorder, axis=1)
        miss_cnt = np.isnan(X).sum(axis=1).reshape(-1,1)
        return np.hstack([Xs, miss_cnt])

class NDVIFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_pca=8):
        self.n_pca = n_pca

    def fit(self, X, y=None):
        self.pca_ = PCA(n_components=self.n_pca).fit(X[:,:-1])
        return self

    def transform(self, X):
        ts = X[:,:-1]
        miss = X[:,-1:]
        t_len = ts.shape[1]
        t = np.arange(t_len)
        stats = np.vstack([
            ts.mean(axis=1),
            ts.max(axis=1),
            ts.min(axis=1),
            ts.std(axis=1),
            skew(ts, axis=1),
            np.percentile(ts, 25, axis=1),
            np.percentile(ts, 50, axis=1),
            np.percentile(ts, 75, axis=1),
            (np.percentile(ts, 75, axis=1) - np.percentile(ts, 25, axis=1))
        ]).T
        slope = np.polyfit(t, ts.T, 1)[0].reshape(-1,1)
        fourier = []
        for f in (1,2):
            fourier.append(np.abs((ts*np.sin(2*np.pi*f*t/t_len)).mean(axis=1)))
            fourier.append(np.abs((ts*np.cos(2*np.pi*f*t/t_len)).mean(axis=1)))
        fourier = np.vstack(fourier).T
        pca_feats = self.pca_.transform(ts)
        return np.hstack([stats, slope, fourier, pca_feats, miss])

prep = NDVIPreprocessor(window=7, polyorder=2)
X_prep = prep.fit_transform(X)

sm = SMOTE(sampling_strategy={'water':2000,'farm':2000,'orchard':2000,'grass':2000,'impervious':2000})
rus = RandomUnderSampler(sampling_strategy={'forest': min((y_raw=='forest').sum(), 7000)})
X_bal, y_bal = sm.fit_resample(X_prep, y_raw)
X_bal, y_bal = rus.fit_resample(X_bal, y_bal)
y_bal_enc = le.transform(y_bal)

pipe = Pipeline([
    ("feat", NDVIFeatures(n_pca=8)),
    ("clf", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42))
])

param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth":    [3, 6],
    "clf__learning_rate":[0.01, 0.1]
}

gs = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_bal, y_bal_enc)
print("Best params:", gs.best_params_)
print("Best CV macro-F1:", gs.best_score_)

best_pipe = gs.best_estimator_
X_test_prep = prep.transform(X_test)
X_test_feat = best_pipe.named_steps["feat"].transform(X_test_prep)
y_pred = best_pipe.named_steps["clf"].predict(X_test_feat)
submission = pd.DataFrame({"ID": test_ids, "class": le.inverse_transform(y_pred)})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with", len(submission), "rows.")