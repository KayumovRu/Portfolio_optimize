import pandas as pd
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# инициализируем приложение
app = FastAPI(title='Invest Portfolio Optimization', version='0.3',
              description='Monte Carlo for Invest Portfolio Optimization')

# загружаем модель
model = pd.read_pickle('../models/model_monte_carlo.pkl')

class Data(BaseModel):
    deposit: int

@app.get('/')
@app.get('/home')
def read_home():
    """
     Home
     """
    return {'message': 'System is OK'}

@app.post("/optimize")
def optimize(data: Data):

    portfolio = model.iloc[model['Sharpe'].idxmax()]

    depo = data.deposit

    result = {
                'PROFIT': "{:.4f}%".format(portfolio.RETURN),
                'EUR': round(portfolio.EUR * depo, 2),
                'GOLD': round(portfolio.GOLD * depo, 2),
                'Bitcoin': round(portfolio.Bitcoin * depo, 2),
                'Apple': round(portfolio.Apple * depo, 2),
                'Exxon': round(portfolio.Exxon * depo, 2),
                'VISA': round(portfolio.VISA * depo, 2),
                'Oil': round(portfolio.Oil * depo, 2)
            }

    return result

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)