# PORTFOLIO, ENVIRONMENT ===========================================

import numpy as np
import pandas as pd

import gym

from core.utils import date_to_index, index_to_date, sharpe,  max_drawdown

EPS = 1e-8

class DataProcessor(object):

    def __init__(self, product_list, market_feature, feature_num, steps,
                window_length, mode, start_index=0, start_date=None, train_ratio=0.8):

        self.train_ratio = train_ratio
        self.steps = steps + 1
        self.window_length = window_length
        self.window_size = 1
        self.start_index = start_index
        self.start_date = start_date
        self.feature_num = feature_num
        self.market_feature = market_feature
        self.mode = mode
        self._data = []
         
        self.product_list = product_list
        self.load_observations()

    def load_observations(self):
        ts_d = pd.read_csv('../datasets/invert/' + 'i_w1_aapl.csv') # загружаем любой файл данных для формирования шкалы
        ts_d_len = len(ts_d)
        csv_data = np.zeros((ts_d_len-self.window_size+1, self.feature_num,
                             len(self.product_list), self.window_size), dtype=float)
        
        for k in range(len(self.product_list)):
            product = self.product_list[k]
            ts_d = pd.read_csv('../datasets/invert/' + 'i_w1_' + product + '.csv')
            ts_d = ts_d.dropna(axis=0,how='any')
            for j in range(len(self.market_feature)):
                ts_d_temp = ts_d[self.market_feature[j]].values
                for i in range(len(ts_d)-self.window_size+1):
                    temp = np.zeros((self.window_size))
                    for t in range(i, i+self.window_size):

                        temp[t-i] = ts_d_temp[t]
                    csv_data[i][j][k] = temp
        csv_data = csv_data[::-1].copy()
        observations = csv_data
        if self.mode == "Train":
            self._data = observations[0:int(self.train_ratio * observations.shape[0])]
        elif self.mode == "Test":
            self._data = observations[int(self.train_ratio * observations.shape[0]):]
        self._data = np.squeeze(self._data)
        self._data = self._data.transpose(2, 0, 1)

    def _step(self):

        self.step += 1

        obs = self.data[:, self.step:self.step + self.window_length, :].copy()

        next_obs = self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()

        done = self.step >= self.steps

        return obs, done, next_obs

    def reset(self):
        # np.random.seed(33)
        self.step = 0
        
        if self.start_date is None:
            self.idx = np.random.randint(
                low=self.window_length, high=self._data.shape[1] - self.steps)
        else:
            self.idx = date_to_index(self.start_date) - self.start_index

        data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 1, :8]
        self.data = data
        return self.data[:, self.step:self.step + self.window_length, :].copy(), \
               self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()
    
# Расчет портфеля
class Portfolio(object):

    def __init__(self, steps, trading_cost, product_num):

        self.steps = steps
        self.cost = trading_cost
        self.product_num = product_num

    def _step(self, w1, y1, reset): # кажется, от reset можно избавиться, если убрать в вызовах
        """
        Step.
        w1 - новые веса портфеля, например [0.2, 0.8, 0.0] - в сумме дают единицу
        y1 - относительный вектор цен, например [1.0, 0.9, 1.1]
        Уровнения пронумерованы по статье https://arxiv.org/abs/1706.10059
        """

        w0 = self.w0
        p0 = self.p0
        returns = self.returns # возвраты при расчете по шарпу

        # Reward 3.3 - Шарп  + log_return
        p1 = p0 * np.dot(y1, w1) 
        p1 = np.clip(p1, 0, np.inf) 
        rho1 = p1 / p0 - 1  # относительные возвраты (доходность)
        returns = np.append(returns, rho1)
        r1 = np.log((p1 + EPS) / (p0 + EPS))  # логдоходность
        rew_sharpe = sharpe(returns) / self.steps  # текущий шарп 
        rew_return = r1 / self.steps * 1000. # основной компонент награды (22) average logarithmic accumulated return
        reward = rew_sharpe + rew_return

        # запоминаем на след шаг
        self.w0 = w1
        self.p0 = p1
        self.returns = returns

        # условие завершения по исчтерпанию средств
        done = bool(p1 == 0)

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": 0,
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        
        # сделал начало с равнодолевых, но еще проверить
        base_weight = int(1/self.product_num * 10000)/10000
        self.w0 = np.array([1-sum([base_weight] * self.product_num)] + [base_weight]*self.product_num)

        self.p0 = 1.0
        self.returns = np.array([0])

# Среда
class envs(gym.Env):
    """
    Основано на https://arxiv.org/abs/1706.10059
    """
    def __init__(self,
                 product_list,
                 market_feature,
                 feature_num,
                 steps,
                 window_length,
                 mode, # Test или Train передается для правильной обработки DataProcessor
                 start_index = 0,
                 start_date = None,
                 trading_cost = 0, # комиссия за сделку
                 train_ratio = 0.8,
                 ):

        self.window_length = window_length
        self.start_index = start_index
        self.dataprocessor = DataProcessor(
            product_list=product_list,
            market_feature=market_feature,
            feature_num=feature_num,
            steps=steps,
            mode=mode,
            window_length=window_length,
            start_index=start_index,
            start_date=start_date,
            train_ratio=train_ratio) ###
        self.portfolio = Portfolio(steps=steps, trading_cost=trading_cost, product_num=len(product_list))
        
    def step(self, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """

        # Нормализация действий
        action = np.clip(action, 0, 1)
        weights = action
        weights /= (weights.sum() + EPS)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)  # если веса нулевые, то нормализовать до [1,0...]

        observation, done1, next_obs, = self.dataprocessor._step()

        c_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((c_observation, observation), axis=0)
        c_next_obs = np.ones((1, 1, next_obs.shape[2]))
        next_obs = np.concatenate((c_next_obs, next_obs), axis=0)

        # получаем ценовой вектор y1 = вектор относительной цены последнего периода наблюдения (Close/Open)
        close_price_vector = observation[:, -1, 3]
        open_price_vector = observation[:, -1, 0]

        reset = 0        
        y1 = close_price_vector / open_price_vector

        reward, info, done2 = self.portfolio._step(weights, y1, reset)
        info['date'] = index_to_date(self.start_index + self.dataprocessor.idx + self.dataprocessor.step)
        self.infos.append(info)

        return observation, reward, done1 or done2, info

    def reset(self):
        self.infos = []
        self.portfolio.reset()
        observation, next_obs = self.dataprocessor.reset()
        c_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((c_observation, observation), axis=0)
        c_next_obs = np.ones((1, 1, next_obs.shape[2]))
        next_obs = np.concatenate((c_next_obs, next_obs), axis=0)
        info = {}
        return observation, info

    # возврат результатов и метрик
    def render(self):
        df_info = pd.DataFrame(self.infos)
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        mdd = max_drawdown(df_info.portfolio_value)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        return sharpe_ratio, mdd, df_info["portfolio_value"][-1]

def load_observations(window_size, market_feature, feature_num, product_list):
    product_list = product_list
    ts_d = pd.read_csv('../datasets/invert/' + 'i_w1_aapl.csv') # ДНЕВНАЯ ШКАЛА ОДНОГО ИЗ ИНСТРУМЕНТОВ - ПОМЕНЯТЬ?
    ts_d_len = len(ts_d)
    data = np.zeros((ts_d_len - window_size + 1, feature_num, len(product_list), window_size), dtype=float)
    
    for k in range(len(product_list)):
        product = product_list[k]
        ts_d = pd.read_csv('../datasets/invert/' + 'i_w1_' + product + '.csv')
        ts_d = ts_d.dropna(axis=0, how='any')
        for j in range(len(market_feature)):
            ts_d_temp = ts_d[market_feature[j]].values
            for i in range(len(ts_d)-window_size+1):
                temp = np.zeros((window_size))
                for t in range(i, i+window_size):
                    temp[t-i] = ts_d_temp[t]
                data[i][j][k] = temp
    data = data[::-1].copy()
    observations = data
    return observations, ts_d_len