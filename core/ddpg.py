# DDPG ====================================

import numpy as np
import json

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from core.model import Actor, Critic, ReplayBuffer
from core.utils import obs_normalizer

C_CUDA = torch.cuda.is_available()
models_path = '../models/'

class DDPG(object):
    def __init__(self, env, product_num, win_size, actor_noise, username, config):

        self.modelname = 'DDPG_' + username # для уникального названия модели от каждого пользователя
        self.config = config    
        self.env = env
        self.actor_noise = actor_noise
        
        if C_CUDA:
            self.actor = Actor(product_num,win_size).cuda()
            self.actor_target = Actor(product_num,win_size).cuda()
            self.critic = Critic(product_num,win_size).cuda()
            self.critic_target = Critic(product_num,win_size).cuda()
        else:
            self.actor = Actor(product_num,win_size)
            self.actor_target = Actor(product_num,win_size)
            self.critic = Critic(product_num,win_size)
            self.critic_target = Critic(product_num,win_size)
        
        self.actor.reset_parameters()
        self.actor_target.reset_parameters()
        self.critic_target.reset_parameters()
        self.actor.reset_parameters()
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.config['actor learning rate'])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.config['critic learning rate'])
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
    
    def act(self, state):
        if C_CUDA:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).cuda()
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).squeeze(0).cpu().detach().numpy()+ self.actor_noise()
        return action
    
    def critic_learn(self, state, action, predicted_q_value):
        actual_q = self.critic(state, action)
        if C_CUDA:
            target_Q = torch.tensor(predicted_q_value, dtype=torch.float).cuda()
        else:
            target_Q = torch.tensor(predicted_q_value, dtype=torch.float)
        target_Q=Variable(target_Q,requires_grad=True)
        td_error  = F.mse_loss(actual_q, target_Q)
        self.critic_optim.zero_grad()
        td_error.backward()
        self.critic_optim.step()
        return predicted_q_value,td_error
    
    def actor_learn(self, state):

        loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        return loss
    
    def soft_update(self, net_target, net, tau):
        for target_param, param  in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def train(self):
        num_episode = self.config['episode']
        batch_size = self.config['batch size']
        gamma = self.config['gamma']
        tau = self.config['tau']
        self.buffer = ReplayBuffer(self.config['buffer size'])
        total_step = 0
        ep_reward_history = [] # история по итоговым вознаграждениям эпизодов
        # writer = SummaryWriter(logdir=self.summary_path)
        # Основной цикл обучения
        for i in range(num_episode):
            previous_observation = self.env.reset()
            previous_observation = obs_normalizer(previous_observation) # нормализуем
            previous_observation = previous_observation.transpose(2, 0, 1)
            ep_reward = 0
            ep_ave_max_q = 0
            
            # Эпизод
            for j in range (self.config['max step']):
                # 1. Действие Actor на основе состояния (st)
                action = self.act(previous_observation)
        		# 2. Получение награды и достижение нового состояния (st+1)
                observation, reward, done, _ = self.env.step(action)
                observation = obs_normalizer(observation)
                observation = observation.transpose(2, 0, 1) # изм размера
        		# 3. Сохранение в Replay Buffer (st, at, rt, st+1)
                self.buffer.add(previous_observation, action, reward, done, observation)
                if self.buffer.size() >= batch_size:
        			# 4. Образец (si, ai, ri, si+1) из буффера
                    s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(batch_size)
                    # Конвертация в torch tensor
                    if C_CUDA:
                        s_batch = torch.tensor(s_batch, dtype=torch.float).cuda()
                        a_batch = torch.tensor(a_batch, dtype=torch.float).cuda()
                        r_batch = torch.tensor(r_batch, dtype=torch.float).cuda()
                        t_batch = torch.tensor(t_batch, dtype=torch.float).cuda()
                        s2_batch = torch.tensor(s2_batch, dtype=torch.float).cuda()
                        target_q = self.critic_target(s2_batch,self.actor_target(s2_batch)).cpu().detach()
                    else:
                        s_batch = torch.tensor(s_batch, dtype=torch.float)
                        a_batch = torch.tensor(a_batch, dtype=torch.float)
                        r_batch = torch.tensor(r_batch, dtype=torch.float)
                        t_batch = torch.tensor(t_batch, dtype=torch.float)
                        s2_batch = torch.tensor(s2_batch, dtype=torch.float)
                        target_q = self.critic_target(s2_batch,self.actor_target(s2_batch)).detach()
                    y_i = []
                    for k in range(batch_size):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k].cpu().numpy() + gamma * target_q[k].numpy()) # y_i = r_batch + gamma * target_q
        			# 6. CRITIC - Обновление параметров Q для приближения Q(si,ai) к y
                    predicted_q_value, td_error = self.critic_learn(s_batch, a_batch,np.reshape(y_i, (batch_size, 1)))
                    # writer.add_scalar('TD error', td_error, global_step=total_step)
                    ep_ave_max_q += np.amax(predicted_q_value)
        			# 7. ACTOR - Обновление параметры Actor для максимизации Q(si, actor(si))
                    actor_loss = self.actor_learn(s_batch)
                    # writer.add_scalar('Actor loss', actor_loss, global_step=total_step)
        			# 8. Сброс на каждом шаге Q^ = Q, actor^ = actor
                    self.soft_update(self.critic_target, self.critic, tau)
                    self.soft_update(self.actor_target, self.actor, tau)
                ep_reward += reward
                previous_observation = observation
                total_step = total_step + 1
                if done or j == self.config['max step'] - 1:
                    ep_reward_history.append(ep_reward)
                    # формируем и записываем прогресс-файл
                    progress_json = {
                        "in_process":       True,
                        "episode":          i+1,
                        "max_step":         self.config['max step'],
                        "total_steps":      total_step,
                        "mean_steps_in_ep": int(total_step / (i+1)),
                        "reward_history":   ep_reward_history,
                    }
                    with open('../models/progress/' + self.modelname + '.json', 'w') as f:
                        json.dump(progress_json, f)
        
        # записываем прогресс-файл c in_process = False
        progress_json = {
            "in_process":       False,
            "episode":          i+1,
            "max_step":         self.config['max step'],
            "total_steps":      total_step,
            "mean_steps_in_ep": int(total_step / (i+1)),
            "reward_history":   ep_reward_history,
        }
        with open('../models/progress/' + self.modelname + '.json', 'w') as f:
            json.dump(progress_json, f)

        torch.save(self.actor.state_dict(), models_path + self.modelname)

        return ep_reward_history
