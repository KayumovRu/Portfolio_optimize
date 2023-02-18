import pandas as pd
import numpy as np

df = pd.read_csv('mvp_data.csv', parse_dates=True, index_col=0)
tickers = ['EUR', 'GOLD', 'Bitcoin', 'Apple', 'Exxon', 'VISA', 'Oil']
df.columns = tickers
df_w = df.fillna(method='ffill').resample('W').ffill()
df_w_pct = df_w.pct_change().iloc[1:]

train_df = df_w_pct[:'2021-12-31'] # 2020-2021
test_df = df_w_pct['2022-01-01':] # 2022

num_assets = len(tickers)

mean_returns = train_df.mean()
cov_matrix = train_df.cov()

#массив из нулей
num_iter = 1000
simulations = np.zeros((4 + len(tickers)- 1, num_iter))

for i in range(num_iter):
        # случайные веса + нормализация, чтобы сумма 1
        weights = np.array(np.random.random(num_assets))
        weights /= np.sum(weights)
        
        # доходность и стандартное отклонение
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights)))
        simulations[0,i] = portfolio_return
        simulations[1,i] = portfolio_std_dev
        
        # кэф Шарпа
        simulations[2,i] = simulations[0,i] / simulations[1,i]
        
        # dtcf
        for j in range(len(weights)):
                simulations[j+3,i] = weights[j]

# формируем результат
df_sim = pd.DataFrame(simulations.T, 
                         columns=['RETURN','stdev','Sharpe',
                                   tickers[0], tickers[1], tickers[2], tickers[3],
                                   tickers[4], tickers[5], tickers[6]])

# сохраняем модель
df_sim.to_pickle('model_monte_carlo.pkl')