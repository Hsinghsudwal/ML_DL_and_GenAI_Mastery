import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import mlflow.sklearn

app = Flask(__name__)
RUN_ID    = os.getenv('MLFLOW_RUN_ID', '{run_id}')
THRESHOLD = float(os.getenv('CHURN_THRESHOLD', '{opt_cost_thresh:.2f}'))
model     = mlflow.sklearn.load_model(f'runs:/{{RUN_ID}}/model')

def _predict(records):
    df     = pd.DataFrame(records)
    probas = model.predict_proba(df)[:, 1]
    return [{{'customer_id': r.get('customerID', i),
              'churn_probability': round(float(p), 4),
              'churn_prediction':  int(p >= THRESHOLD),
              'risk_segment': 'high' if p >= 0.7 else ('medium' if p >= 0.4 else 'low')}}
            for i, (r, p) in enumerate(zip(records, probas))]

@app.route('/predict', methods=['POST'])
def predict_single():
    return jsonify(_predict([request.get_json(force=True)])[0])

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    data = request.get_json(force=True)
    return jsonify(_predict(data if isinstance(data, list) else data['records']))

@app.route('/health')
def health():
    return jsonify({{'status': 'ok', 'threshold': THRESHOLD}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))

# Testing: curl -X POST http://localhost:8080/health