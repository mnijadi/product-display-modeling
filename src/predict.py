""" data preprocessing utilities and model deployment script """

import pandas as pd
import mlflow
from flask import Flask, request, jsonify

import os

def preprocess_data(data):
    """preprocess data for modeling
    data is supposed to be a dictionary (json issued from a POST request), having the following keys:
    - 'cor_sales_in_vol': represents 'X1_sales_in_vol'
    - 'cor_sales_in_val': represents 'X2_sales_in_val'
    - 'CA_mag': represents 'X4_turnover'
    - 'value': reprensents 'X5_value'
    - 'ENSEIGNE': represents 'X6_sign' (to be one-hot encoded)
    - 'VenteConv': represents 'X3_sales'
    - 'Feature': represents 'X7_feature' (to be one-hot encoded)
    """
    features = {
        'X1_sales_in_vol': data['cor_sales_in_vol'],
        'X2_sales_in_val': data['cor_sales_in_val'],
        'X3_sales': data['VenteConv'],
        'X4_turnover': data['CA_mag'],
        'X5_value': data['value'],
        'X6_sign_AUCHAN': 1 if data['ENSEIGNE'] == 'AUCHAN' else 0,
        'X6_sign_CARREFOUR': 1 if data['ENSEIGNE'] == 'CARREFOUR' else 0,
        'X6_sign_CARREFOUR MARKET': 1 if data['ENSEIGNE'] == 'CARREFOUR MARKET' else 0,
        'X6_sign_CASINO': 1 if data['ENSEIGNE'] == 'CASINO' else 0,
        'X6_sign_CORA': 1 if data['ENSEIGNE'] == 'CORA' else 0,
        'X6_sign_GEANT': 1 if data['ENSEIGNE'] == 'GEANT' else 0,
        'X6_sign_INTERMARCHE': 1 if data['ENSEIGNE'] == 'INTERMARCHE' else 0,
        'X6_sign_LECLERC': 1 if data['ENSEIGNE'] == 'LECLERC' else 0,
        'X6_sign_MONOPRIX': 1 if data['ENSEIGNE'] == 'MONOPRIX' else 0,
        'X6_sign_OTHERS': 1 if data['ENSEIGNE'] in ['MATCH', 'MARCHE U', 'PRISUNIC', 'HYPER U', 'ECOMARCHE', 'OTHERS', 'FRANPRIX', 'SHOPI'] else 0,
        'X6_sign_SIMPLY MARKET': 1 if data['ENSEIGNE'] == 'SIMPLY MARKET' else 0,
        'X6_sign_SUPER U': 1 if data['ENSEIGNE'] == 'SUPER U' else 0,
        'X7_feature_Feat': 1 if data['Feature'] == 'Feat' else 0,
        'X7_feature_No_Feat': 1 if data['Feature'] == 'No_Feat' else 0,
    }
    for k in features.keys():
        features[k] = float(features[k])
    return features


# load model from mlflow
mlflow.set_tracking_uri('http://127.0.0.1:8080')
ensembles_df = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name('Ensembles').experiment_id)
run_id = ensembles_df.loc[ensembles_df['tags.Version']=='v5', 'run_id'].values[0]
logged_model = 'runs:/4d7051ac23a94172847f29433e1d3040/cb'
model = mlflow.pyfunc.load_model(logged_model)

app = Flask('product-display-modeling')

@app.route('/predict', methods=['POST'])
def predict():
    """predict endpoint"""
    data = request.get_json()
    features = preprocess_data(data)
    prediction = model.predict(features)
    return jsonify({
        'Display': 'Displ' if prediction == 1 else 0,
        'RUN_ID': run_id
        })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)