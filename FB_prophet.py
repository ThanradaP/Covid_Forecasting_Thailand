from importlib_metadata import version
import pandas as pd
from fbprophet import Prophet
import json
from fbprophet.serialize import model_to_json,model_from_json

df = pd.read_csv('covid_data_cleaned.csv')
prophet_df = (df[['date','new_cases']].rename(columns={"date": "ds", "new_cases": "y"}))
prophet_df.head()
model = Prophet()
model.fit(prophet_df)


