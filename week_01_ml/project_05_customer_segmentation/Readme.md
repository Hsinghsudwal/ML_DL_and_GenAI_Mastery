# 🧠 Customer Segmentation System (End-to-End ML Pipeline)

A clean, modular pipeline for **unsupervised customer segmentation** using clustering, enhanced with **automated tuning, feature engineering, and downstream ML models**.

---
sssssss
# 📌 Overview

This project builds a **multi-stage segmentation system**:

1. Data preprocessing & feature engineering
2. Clustering (K-Means + Agglomerative)
3. Automated model selection (Optuna)
4. Cluster evaluation & visualization
5. Downstream supervised learning (XGBoost)
6. Advanced improvements (SMOTE, stability, alternative models)

---

# ⚙️ 1. Core Feature Setup

```python
features = ['Age', 'Annual_Income', 'Spending_Score']
X = df[features]
```

---

# 🔧 2. Preprocessing Pipeline

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

# 🤖 3. K-Means Clustering

```python
from sklearn.cluster import KMeans

k = 5

kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
df['Cluster_Income_Spending'] = kmeans.fit_predict(X_scaled)
```

---

# 🌳 4. Agglomerative Clustering

```python
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=5, linkage='ward')
df['Cluster_Hierarchical'] = agg.fit_predict(X_scaled)
```

---

# 📊 5. Evaluation Metrics

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

sil_km  = silhouette_score(X_scaled, df['Cluster_Income_Spending'])
sil_agg = silhouette_score(X_scaled, df['Cluster_Hierarchical'])

db_agg  = davies_bouldin_score(X_scaled, df['Cluster_Hierarchical'])

print(f"KMeans Silhouette: {sil_km:.4f}")
print(f"Agglomerative Silhouette: {sil_agg:.4f}")
print(f"Agglomerative DB Index: {db_agg:.4f}")
```

---

# 📈 6. Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=df,
    x='Annual_Income',
    y='Spending_Score',
    hue='Cluster_Income_Spending',
    palette='viridis',
    s=100
)

plt.title('Customer Segments — KMeans')
plt.show()
```

---

# 🚀 7. Automated K Selection (Optuna)

```python
import optuna
from sklearn.metrics import silhouette_score

def objective(trial):
    k = trial.suggest_int('k', 2, 10)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    return silhouette_score(X_scaled, labels)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

best_k = study.best_params['k']
print("Best k:", best_k)
```

---

# 🎯 8. Downstream Classification (XGBoost)

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X_cls = df[['Age', 'Annual_Income', 'Spending_Score']]
y_cls = df['Cluster_Income_Spending']

X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=200, eval_metric='mlogloss')
model.fit(X_tr, y_tr)

acc = (model.predict(X_te) == y_te).mean()
print("XGBoost Accuracy:", acc)
```

---

# ⚖️ 9. Handle Imbalanced Clusters (SMOTE)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tr, y_tr)
```

---

# 🧪 10. Hyperparameter Tuning (Optuna + XGBoost)

```python
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }

    model = XGBClassifier(**params, eval_metric='mlogloss')
    model.fit(X_tr, y_tr)

    return (model.predict(X_te) == y_te).mean()

study = optuna.create_study(direction='maximize')
study.optimize(xgb_objective, n_trials=50)

print("Best Params:", study.best_params)
```

---

# 🧬 11. Advanced Feature Engineering

```python
import numpy as np

df['Spending_Propensity'] = df['Spending_Score'] / (df['Annual_Income'] + 1e-6)
df['Income_per_Age'] = df['Annual_Income'] / (df['Age'] + 1e-6)
df['Log_Income'] = np.log1p(df['Annual_Income'])
df['Age_Spending'] = df['Age'] * df['Spending_Score']
```

---

# 🔍 12. Alternative Clustering Models

* **DBSCAN** → no need for k, detects outliers
* **Gaussian Mixture Models (GMM)** → soft clustering
* **HDBSCAN** → density-based hierarchical clustering

---

# 🔁 13. Cluster Stability (Advanced Validation)

```python
from sklearn.metrics import adjusted_rand_score

labels1 = KMeans(n_clusters=5, random_state=42).fit_predict(X_scaled)
labels2 = KMeans(n_clusters=5, random_state=99).fit_predict(X_scaled)

ari = adjusted_rand_score(labels1, labels2)
print("Cluster Stability (ARI):", ari)
```

---

# 🧠 Key Takeaways

* Combine **unsupervised + supervised learning** for production-ready systems
* Automate decisions (k, hyperparameters) → remove guesswork
* Engineer meaningful features → improves clustering quality
* Validate stability → ensures reliable segmentation

---

# 📦 Future Extensions

* Deploy as **API (FastAPI / Flask)**
* Build **real-time segmentation system**
* Add **multi-modal inputs (behavioral, transactional, time-series)**
* Upgrade to **agentic ML pipeline (auto decision-making system)**

---

