# MAIN.PY =====================================

import pandas as pd
import math
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import numpy as np

import sys
sys.path.insert(0, '../')

from core.ddpg import DDPG
from core.utils import date_to_index, obs_normalizer
from core.model import OrnsteinUhlenbeckActionNoise, loadModel
from core.environment import envs, load_observations

import torch


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
C_CUDA = torch.cuda.is_available()
# /=========================

# (только для отладки, условная воспроизводимость)
# !!! также убрать вручную расставленные seed в классах
# np.random.seed(33)
# random.seed(33)
# torch.manual_seed(33)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

app = FastAPI(title='Invest Portfolio Optimization', version='0.5',
              description='Invest Portfolio Optimization')

# ====================================================================
PATH_DATA = '.../datasets/'
models_path = '../models/'
date_format = '%Y-%m-%d'
EPS = 1e-8

# тестирование DRL модели, поддерживается: DDPG
def testModel(env, model, start_depo=10000):
    observation, info = env.reset()
    observation = obs_normalizer(observation)
    observation = observation.transpose(2, 0, 1)
    done = False
    ep_reward = 0
    deposit = start_depo
    deposits = [start_depo]
    actions = []
    while not done:
        observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
        action = model(observation).squeeze(0).cpu().detach().numpy()        
        observation, reward, done, info = env.step(action)
        ep_reward += reward
        r = info['log_return']
        deposit = deposit * math.exp(r)
        deposits.append(deposit) # депозит
        actions.append(action.tolist()) # распределение активов
        observation = obs_normalizer(observation)
        observation = observation.transpose(2, 0, 1)
    sharpe_ratio, mdd, portfolio_value = env.render()
    return deposits, actions, sharpe_ratio, mdd, portfolio_value

# тестирование базовых моделей; поддерживаются: EW - равнодолевая, Markowitz - модель Марковица
def testBaseline(env, product_num, baseline='EW', start_depo=10000):
    observation, info = env.reset()
    observation = obs_normalizer(observation)
    observation = observation.transpose(2, 0, 1)
    done = False
    ep_reward = 0
    deposit = start_depo
    deposits = [start_depo]
    actions = []
    if baseline == 'EW':
        base_weight = int(1/product_num * 10000)/10000
        base_action = [base_weight] * product_num
        base_action.insert(0, 1-sum(base_action))
        action = base_action
    elif baseline == 'Markowitz': #
        action = []
    while not done:
        observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
        observation, reward, done, info = env.step(action)
        ep_reward += reward
        r = info['log_return']
        deposit = deposit * math.exp(r)
        deposits.append(deposit) # депозит
        actions.append(action) # распред активов
        observation = obs_normalizer(observation)
        observation = observation.transpose(2, 0, 1)
    sharpe_ratio, mdd, portfolio_value = env.render()
    return deposits, actions, sharpe_ratio, mdd, portfolio_value

# API =========================================================================

# структура данных для INFERENCE
class DataInference(BaseModel):
    deposit: int
    train_ratio: float
    basemodels: list
    model_ddpg: str
    comission: float

# структура данных для TRAIN
class DataTrain(BaseModel):
    username: str
    deposit: int
    train_ratio: float
    episodes_num: int

# HOME проверочная функция
@app.get('/')
@app.get('/home')
def read_home():
    """
     Home
     """
    return {'message': 'System is OK'}

# INFERENCE применение модели 
@app.post("/inference")
def inference(data: DataInference):
    
    market_feature = ['open', 'high', 'low', 'close']
    feature_num = len(market_feature)
    window_size = 1
    window_length = 3
    train_ratio = data.train_ratio

    start_depo = data.deposit

    df_assets = pd.read_csv('../datasets/' + "assets.csv")
    product_list = df_assets["code"]
    product_num = len(product_list)

    observations, ts_d_len = load_observations(window_size, market_feature, feature_num, product_list)

    train_size = int(train_ratio*ts_d_len)
    test_observations = observations[int(train_ratio * observations.shape[0]):]
    test_observations = np.squeeze(test_observations)
    test_observations = test_observations.transpose(2, 0, 1)
    mode = "Test"

    ts = pd.read_csv("../datasets/w1_aapl.csv")

    start_test = ts.iloc[train_size]['date']
    steps = ts_d_len - train_size - window_length - 2 
    shift_train_size = date_to_index(start_test) - window_length - train_size

    env = envs(product_list, market_feature, feature_num, steps, window_length, mode,
            start_index = train_size + shift_train_size, 
            start_date = start_test,
            train_ratio = train_ratio)

    inference_result = {}

    # по DRL-моделям
    if data.model_ddpg != "none":
        model = loadModel("../models/" + data.model_ddpg, product_num, window_length)
        depo_results, action_results, sharpe_ratio, mdd, portfolio_value = testModel(env, model, start_depo)

        inference_result["DDPG"] = {
            'depo_results':     depo_results, # изменение депозита в формате [t1, t2, ...]
            'action_results':   action_results, # распределение активов, структура: [[a11, a21], [a12, a22], ...], где aNx - доля N актива в период x
            'sharpe_ratio':     sharpe_ratio,
            'mdd':              mdd,
            'portfolio_value':  portfolio_value,
            'assets_list':      df_assets["ticker"],
        }
    
    # по базовым моделям
    # available_basemodels = ['EW', 'Markowitz'] # поддерживаемые бэкендом модели
    available_basemodels = ['EW']
    basemodels = list(set(available_basemodels) & set(data.basemodels)) # оставляем только те модели с фронта, которые поддерживаются
    for basemodel in basemodels:
        depo_results, action_results, sharpe_ratio, mdd, portfolio_value = testBaseline(env, product_num, basemodel, start_depo)

        inference_result[basemodel] = {
            'depo_results':     depo_results, 
            'action_results':   action_results, 
            'sharpe_ratio':     sharpe_ratio,
            'mdd':              mdd,
            'portfolio_value':  portfolio_value,
            'assets_list':      df_assets["ticker"],
        }

    return inference_result

# TRAIN
@app.post("/train")
async def train(data: DataTrain):
    market_feature = ['open', 'high', 'low', 'close']
    feature_num = len(market_feature)

    df_assets = pd.read_csv('../datasets/' + "assets.csv")
    product_list = df_assets["code"]
    product_num = len(product_list)

    action_dim = [product_num+1]

    mode = "Train"
    window_length = 3
    train_ratio = data.train_ratio
    window_size = 1

    observations, ts_d_len = load_observations(window_size, market_feature, feature_num, product_list)
    train_size = int(train_ratio*ts_d_len)
    steps = train_size - window_length - 2

    env = envs(product_list, market_feature, feature_num, steps, window_length, mode,
            train_ratio = train_ratio)

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    config_json = {
        "episode": data.episodes_num,
        "max step": steps,
        "buffer size": 100000,
        "batch size": 64,
        "tau": 0.001,
        "gamma": 0.99,
        "actor learning rate": 0.0001,
        "critic learning rate": 0.001,
        "policy learning rate": 0.0001
    }

    ddpg_model = DDPG(env, product_num, window_length, actor_noise, data.username, config_json)
    ep_reward_history = ddpg_model.train()

    train_result= {
        'steps': steps,
        'ep_reward_history': ep_reward_history,
    }

    return train_result


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)