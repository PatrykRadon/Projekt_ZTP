import numpy as np
import pandas as pd
from dask_ml.model_selection import train_test_split
import dask.dataframe as dd
def reset_db():
    N = 3000
    m2 = np.random.gamma(shape=2, scale=1, size=N)
    m2 = np.round(19.34 + 30*m2/m2.mean(), 2)
    rooms = 1 + np.clip(np.floor(np.random.poisson(lam=1.3, size=N)), a_min=0, a_max=8)
    age = np.random.gamma(shape=4, scale=1, size=N)
    age = np.round(5*age/age.mean(), 0)
    price = np.round(m2*np.random.normal(7500, 1500, size=N) * (1+(rooms*np.random.normal(0.2, 0.05, size=N))**0.3) * (0.92**(age**0.5)), -3)
    ideal_price = m2*7500 * (1+(rooms*0.2)**0.3) * (0.92**(age**0.5)) 
    house_df = pd.DataFrame(np.stack([m2, rooms, age, price]).T, columns = ['sqare_meters', 'rooms', 'age', 'price'])
    house_df['sold'] = house_df['price'] < ideal_price*np.random.normal(0.8, 0.3, size=N)
    house_df['sold'].mean()

    
    df_train, df_test = train_test_split(house_df, random_state=0)
    df_test.to_csv('./data/test_set.csv', index=False)
    df_train.to_csv('./data/train_set.csv', index=False)

    df_train = dd.read_csv('./data/train_set.csv')
    df_test = dd.read_csv('./data/test_set.csv')

    df_train.to_parquet('./data/train_set.parquet')
    df_test.to_parquet('./data/test_set.parquet')