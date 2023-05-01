import numpy as np
from flask import Flask, request, jsonify, render_template
import dask.dataframe as dd
from dask_ml.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
# from dask_ml.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
from flask import request
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__) #Initialize the flask App

model = joblib.load('models/apartment_classification/latest.joblib') 

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['GET', 'POST'])
def predict():

    def parse_input():
        return [[
            float(request.args.get('square_meters')),
            float(request.args.get('rooms')),
            float(request.args.get('age')),
            float(request.args.get('price'))
        ]]
    prediction_result = model.predict_proba(parse_input()).flatten()[1]
    return jsonify(
        value_input=request.args,
        sale_probability=prediction_result
    )
    
    
@app.route('/retrain_model',methods=['GET'])
def retrain_model():
    
    #####
    # from reset_db import reset_db
    # reset_db()
    #####
    
    
    df = dd.read_parquet('./data/train_set.parquet/*')
    df_test = dd.read_parquet('./data/test_set.parquet/*')
    
    
    X = df[['sqare_meters','rooms','age','price']]
    y = df.sold.astype(int)

    X_test = df_test[['sqare_meters','rooms','age','price']]
    y_test = df_test.sold.astype(int)
    
    pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
    parameters = {'lr__C': [0.1, 1, 10]}
    clf = GridSearchCV(pipe, parameters)
    clf.fit(X.values, y) 
    
    y_test_pred = clf.predict_proba(X_test.values.compute_chunk_sizes())
    
    acc = accuracy_score(y_true=y_test, y_pred=y_test_pred[:,1] > 0.5)
    auc = roc_auc_score(y_true=y_test, y_score=y_test_pred[:,1])    
    
    new_model = joblib.dump(clf, 'models/apartment_classification/latest.joblib') 
    
    
    return jsonify(
        new_model = new_model,
        acc=acc,
        auc=auc
    )
    
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')