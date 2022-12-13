
import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model('rf_tuned_loaded_api')

# Define predict function
@app.post('/predict')
def predict(uf, tot_orders_12m, tot_items_12m, tot_items_dist_12m, receita_12m, recencia):
    data = pd.DataFrame([[uf, tot_orders_12m, tot_items_12m, tot_items_dist_12m, receita_12m, recencia]])
    data.columns = ['uf', 'tot_orders_12m', 'tot_items_12m', 'tot_items_dist_12m', 'receita_12m', 'recencia']
    predictions = predict_model(model, data=data) 
    return {'prediction': list(predictions['Label'])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)