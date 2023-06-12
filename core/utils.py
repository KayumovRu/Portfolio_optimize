# UTILS  ====================================================================

import datetime
import numpy as np

date_format = '%Y-%m-%d'
start_date = '2009-01-01' # глобальный старт
start_datetime = datetime.datetime.strptime(start_date, date_format)

EPS = 1e-8

def date_to_index(date_string):
    return (datetime.datetime.strptime(date_string, date_format) - start_datetime).days

def index_to_date(index):
    return (start_datetime + datetime.timedelta(index)).strftime(date_format)

# Шарп в пересчете на год
def sharpe(returns, freq=52, rfr=0):
    try:
        return (np.sqrt(freq) * np.mean(returns - rfr + EPS)) / np.std(returns - rfr + EPS)
    except:
        return 0.

# MDD
def max_drawdown(return_list):
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 
    if i == 0:
        return 0
    j = np.argmax(return_list[:i]) 
    return (return_list[j] - return_list[i]) / (return_list[j])

def normalize(x):
    return (x - 1) * 100

def obs_normalizer(observation):
    # Normalize the observation into close/open ratio
    if isinstance(observation, tuple):
        observation = observation[0]
    observation = observation[:, :, 3:4] / observation[:, :, 0:1]
    observation = normalize(observation)
    return observation

def hidden_init(layer):
    # Initialize the parameter of hidden layer
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)