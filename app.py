from flask import Flask, jsonify, render_template
import dask.dataframe as dd
from dask_ml.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
# from dask_ml.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_absolute_error, mean_squared_error
import joblib
from flask import request
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime
from uuid import uuid4
from numpy import sqrt
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model = joblib.load('models/apartment_classification/latest.joblib')

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


@app.route('/retrain_model', methods=['GET'])
def retrain_model():
    #####
    # from reset_db import reset_db
    # reset_db()
    #####

    df = dd.read_parquet('./data/train_set.parquet/*')

    X = df[['sqare_meters', 'rooms', 'age', 'price']]
    y = df.sold.astype(int)

    pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
    parameters = {'lr__C': [0.1, 1, 10]}
    clf = GridSearchCV(pipe, parameters)
    clf.fit(X.values, y)

    new_model = joblib.dump(clf, 'models/apartment_classification/latest.joblib')

    return jsonify(
        new_model=new_model
    )


@app.route('/test_model', methods=['GET'])
def test_model():
    model = joblib.load('models/apartment_classification/latest.joblib')
    df_test = dd.read_parquet('./data/test_set.parquet/*')
    X_test = df_test[['sqare_meters', 'rooms', 'age', 'price']]
    y_test = df_test.sold.astype(int)

    y_test_pred = model.predict_proba(X_test.values.compute_chunk_sizes())

    acc = accuracy_score(y_true=y_test, y_pred=y_test_pred[:, 1] > 0.5)
    auc = roc_auc_score(y_true=y_test, y_score=y_test_pred[:, 1])
    loss = log_loss(y_test, y_test_pred)    mae = mean_absolute_error(y_true=y_test, y_pred=y_test_pred[:, 1])
    rmse = sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred[:, 1]))

    return jsonify(
        acc=acc,
        auc=auc,
        loss=loss,
        mae=mae,
        rmse=rmse
    )

@app.route('/houses', methods=['GET'])
def get_houses():
    df_test = pd.read_parquet('./data/active_set.parquet')
    result = []
    expired_data = []
    current_timestamp = datetime.now()
    for _, data in df_test.iterrows():
        if data.expiration_timestamp < current_timestamp:
            append_data(data, expired_data)
        else:
            append_data(data, result)

    if expired_data:
        move_data_to_train_set(expired_data)

    return jsonify(data=result)

@app.route('/houses', methods=['PUT'])
def put_house():
    id = request.json["id"]
    data = pd.read_parquet('./data/active_set.parquet')

    result = data.loc[data['id'] == id].iloc[0]
    move_data_to_train_set(result, sold=True)

    df = data.loc[data['id'] != id]
    df.to_parquet('./data/active_set.parquet')

    return jsonify(message="success ;-))")

@app.route('/houses', methods=['POST'])
def post_house():
    request_data = request.json
    new_data = {
        'sqare_meters': request_data['square_meters'],
        'rooms': request_data['rooms'],
        'age': request_data['age'],
        'price': request_data['price'],
        'expiration_timestamp': datetime.strptime(request.json["expiration_timestamp"], '%a, %d %b %Y %H:%M:%S %Z'),
        'id': str(uuid4())
    }
    new_data = pd.DataFrame.from_dict(new_data, orient='index').T
    print(new_data)
    print(type(new_data))

    df = pd.read_parquet('./data/active_set.parquet')
    df = df.append(new_data)
    print(df)
    df.to_parquet('./data/active_set.parquet')

    return jsonify(message="success :-DDD")


def append_data(data, result):
    result.append({
            "id": data.id,
            "expiration_timestamp": data.expiration_timestamp,
            "price": data.price,
            "age": data.age,
            "rooms": data.rooms,
            "sqare_meters": data.sqare_meters,
        })


def move_data_to_train_set(expired_data, sold=False):
    df = pd.read_parquet('./data/train_set.parquet')
    expired_data = expired_data[['sqare_meters', 'rooms', 'age', 'price']]
    expired_data['sold'] = str(sold)

    df = df.append(expired_data)

    df.to_parquet('./data/train_set.parquet/part.0.parquet')

