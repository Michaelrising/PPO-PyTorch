#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:46:21 2021

@author: michael
"""

# Environment
from abc import ABC
import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import bernoulli
from collections import deque
from gym.utils import seeding
import torch
import argparse


class CancerControl(gym.Env, ABC):
    def __init__(self, patient, t=0.):
        # patient is a dictionary: the patient specific parameters:
        # A, alpha, K, pars, initial states and the terminal states of original drug administration scheme
        # time step: one day
        self.t = t
        self.A, self.K, self.pars, self.init_states, self.terminate_states, self.weight, \
        self.base, self.m1, self.m2, drug_decay, drug_length = patient
        self.terminate_states[0] = 5e+8
        # the terminal state of sensitive cancer cell is replaced by the capacity of sensitive cancer cell
        # note that the terminal state of the AI cancer cell is the mainly focus in our problem
        self.gamma = 0.99  # RL discount factor
        # observation space is a continuous space
        low = np.array([0., 0., 0., -1, -1, -1, 0], dtype=np.float32)
        high = np.array([1,1,1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high)
        self.treatOnOff = 1  # default On and changes in every step function
        self.cpa = np.array([0, 50, 100, 150, 200])
        self.leu = np.array([0, 7.5])
        self._action_set = np.stack((np.tile(self.cpa, 2), np.sort(self.leu.repeat(5))), axis=1)
        self.action_space = spaces.Discrete(10)
        self.steps = 0
        self.penalty_index = np.zeros(2, dtype = np.float)
        self.reward_index = 0
        # the first two denotes the drug dosage, the last two denotes the duration time of each drug's treatment,
        # the longest duration for the first line drug is 300 weeks, 
        # for second line drug, the longest duration is 12 months
        # Note that for LEU, the total dosage should be 7.5*8 ml for one treatment duration
        pp = (self.A, self.K, self.pars)
        self.cancerode = CancerODEGlv(*pp)
        self.dose = np.zeros(2)
        self.normalized_coef = np.append(self.K, self.K[0] / (1.1 * 5) * self.cancerode.cell_size * 22.1).reshape(-1)
        # self._dose = np.zeros(2)
        self.dosage_arr = []
        self.leu_on = False
        self._action = None
        self.drug_penalty_decay = drug_decay
        self.drug_penalty_length = drug_length
        self.rest_decay = 0.95
        self.max_episodes_steps = 120
        self.metastasis_ai_deque = deque(maxlen=121)
        self.metastasis_ad_deque = deque(maxlen=121)

    def CancerEvo(self, dose):
        # get the cell counts and the PSA level of each day
        # 28 days for one action
        ###################
        ts = np.linspace(start=self.t, stop=self.t + 28 - 1, num=28, dtype=np.int)
        dose_leu = dose[1]
        temp = np.zeros(ts.shape[0], dtype=np.float)
        if dose_leu != 0:
            if not self.leu_on:
                temp[0:7 * 1] = - 3.75 / 6 * np.linspace(0, 7, 7, endpoint= False)
                temp[(7 * 1):(7 * 3)] = (7.5 + 3.75) / (20 - 6) * np.linspace(7, 21, 14, endpoint= False) + (
                        7.5 - (7.5 + 3.75) / (20 - 6) * 20)
                temp[(7 * 3):] = 7.5
            else:
                temp[:] = 7.5
        else:  # current dosage is 0
            temp[:] = 0

        drug = np.repeat(dose.reshape(1, -1), ts.shape[0], axis=0)
        drug[:, 1] = temp

        self.cancerode.ts = ts
        # normalization the drug concentration
        self.cancerode.drug = drug * np.array([1 / 200, 1 / 7.5])
        y0 = self.states
        # dose = torch.from_numpy(dose_)
        t_interval = (int(self.t), int(self.t) + 28 - 1)
        out = solve_ivp(self.cancerode.forward, t_span=t_interval, y0=y0, t_eval=ts, method="DOP853")
        # out = Solve_ivp.solver(self.cancerode.forward, ts = ts, y0 = y0, params = (), atol=1e-08, rtol = 1e-05)
        dy = self.cancerode.forward(t = int(self.t) + 28 - 1, y = out.y[:,-1].reshape(-1))
        return out.y, dy

    def step(self, action):
        if self.steps == 0:
            self.states = self.init_states.copy()
            self.penalty_index = np.zeros(2, dtype=np.float)
            self.reward_index = 0
            x0, _ = self.init_states[0:2].copy(), self.init_states[2].copy()
            self.leu_on = False

        # By taking action into next state, and obtain new state and reward
        # the action is the next month's dosage
        # update states
        phi0, _ = self._utilize(self.t, action)
        dose_ = self._action_set[action]
        self.dose = dose_
        _dose = self.dosage_arr[-1] if self.steps > 0 else np.zeros(2, dtype=np.float)
        _dose_leu = _dose[1]
        self.leu_on = bool(_dose_leu)
        # penalty index  for the continuous drug administration, and reward index for no-drug administration
        if (dose_ == 0).all() and (self.penalty_index >= 1).all():
            self.reward_index += 1
            self.penalty_index -= np.ones(2, dtype=np.float)
        if dose_[0] != 0 and dose_[1] == 0:
            self.penalty_index[0] += 1.
            if self.penalty_index[1] >= 1:
                self.penalty_index[1] -= 1.
            self.reward_index = 0
        if dose_[1] != 0 and dose_[0] == 0:
            self.penalty_index[1] += 1.
            if self.penalty_index[0] >= 1:
                self.penalty_index[0] -= 1
            self.reward_index = 0
        if (dose_ != 0).all():
            self.reward_index = 0
            self.penalty_index += np.ones(2, dtype=np.float)

        evolution, df = self.CancerEvo(dose_)
        self.states = evolution[:, -1]
        x, psa = self.states[0:2], self.states[-1]
        t_current = self.t + 28
        self.t = t_current
        self._action = action
        phi1, c2 = self._utilize(self.t, action)
        # reward
        # threshold1 = bool(sum(x)/sum(self.init_states[:2]) < 0.25) * (1 - sum(x)/sum(self.init_states[:2]))
        # threshold2 = bool(sum(x)/sum(self.init_states[:2]) > 0.5) * (sum(x)/sum(self.init_states[:2]))
        # reward = self.weight1[0] * threshold1 - self.weight1[1] * threshold2
        reward = 0
        metastasis_ai = bernoulli.rvs((x[1]/self.K[1])**1.5, size=1).item() if 1 > x[1]/self.K[1] > self.m1 else 0
        self.metastasis_ai_deque.append(metastasis_ai)
        metastasis_ad = bernoulli.rvs((x[0] / self.K[0])**1.5, size=1).item() if 1 > x[0]/self.K[0] > self.m1 else 0
        self.metastasis_ad_deque.append(metastasis_ad)
        done = bool(
            x[0] >= self.K[0]
            or x[1] >= self.K[1]
            # or x[1] / x[0] > 20 # from the patients original data
            # or bool(self.LEU_On and dose_[1] != 0)
            # or bool(sum(x) > sum(self.init_states[:2]))
            or self.steps > self.max_episodes_steps
            or bool(self.metastasis_ad_deque.count(1) >= self.m2)
            or bool(self.metastasis_ai_deque.count(1) >= self.m2)
        ) if 1 > x[0]/self.K[0] else True

        self.dosage_arr.append(dose_)
        r_shape = self.gamma * phi1 - phi0
        #if not done:
        # dosage penalty with effect until the drug are eliminated from body
        dosages = np.array(self.dosage_arr)
        d_decay = np.flipud(self.drug_penalty_decay ** np.arange(min(self.drug_penalty_length, len(self.dosage_arr))))
        # d_decay = d_decay[int(len(self.dosage_arr) - self.drug_penalty_length):len(self.dosage_arr)]
        # c_decay = np.flipud(self.drug_penalty_decay ** np.arange(self.penalty_index[0]))
        # l_decay = np.flipud(self.drug_penalty_decay ** np.arange(self.penalty_index[1]))
        c_dose = (dosages[int(len(self.dosage_arr) - self.drug_penalty_length):, 0]*d_decay).sum() / 200
        l_dose = (dosages[int(len(self.dosage_arr) - self.drug_penalty_length):, 1]*d_decay).sum() / 7.5
        d = np.array([c_dose, l_dose])
        drug_penalty = sum((self.base**self.penalty_index -1) *d * np.array([.7, .3])) # (self.base**self.penalty_index -1) *
        reward += 5*(r_shape + c2) - drug_penalty + self.steps + 1
        # reward += self.reward_index * 0.5
        if done:
            drug_penalty = 0
            dosage = np.array([0.7, 0.3]) * np.array(self.dosage_arr).sum(axis = 0)/np.array([200, 7.5])
            # reward -= sum(dosage)
            reward -= self.max_episodes_steps - self.steps # /self.max_episodes_steps
        # reward *= self.steps/self.max_episodes_steps
        self.steps += 1
        normalized_states = np.log10(self.states) / np.log10(self.normalized_coef)
        normalized_df = np.zeros(3)
        for i, dx in enumerate(df):
            if dx > 1:
                normalized_df[i] = (np.log10(dx) + 1)/np.log10(self.normalized_coef[i])
            elif dx < -1:
                normalized_df[i] = (-np.log10(-dx) - 1) / np.log10(self.normalized_coef[i])
            else:
                normalized_df[i] = dx/np.log10(self.normalized_coef[i])
        fea = np.concatenate((normalized_states, normalized_df, np.array([self.steps/self.max_episodes_steps]))).reshape(-1)
        return fea, self.states, reward, done, {"evolution": evolution, "dose": dose_, "potential": phi0, "r_shape": r_shape,
                                   "c2": c2, "dose_p": drug_penalty,
                                   "ad": self.metastasis_ad_deque.count(1), "ai": self.metastasis_ai_deque.count(1)}

    def _utilize(self, t, action):
        # potential function is only related to states
        # the potential decreases after using drug but increase when without drug administration
        # what we want to describe in the potential function is:
        # 1) the decrease of the cancer cell is rewarding of the therapy
        # 2) the competition of the cancer cell: stronger, better

        pars = self.pars
        phi = pars[-4]
        A = self.A.copy()
        A[0,1] = 1 / (1 + np.exp(-phi * np.array([t / 28 / 12])))
        K = self.K
        states = self.states.copy()
        x, psa = states[0:2], states[2]
        # potential for cancer volume of androgen dependent cancer cell
        # normalized_vt = -sum(np.log10(x))/sum(np.log10(K)) * 100
        vt = - sum(x)/sum(K)
        # normalized_vai = -np.log10(x[1])/np.log10(K[1]) * 100 if x[1] > 1 else 0
        vai = - x[1]/K[1]
        # normalized_c2 = np.log10(x[0])/np.log10(K[0]) * A[0,1] * 100
        c2 = (x[0] * A[0,1] / K[1])**phi # the competition index from AD to AI cells
        potential = self.weight @ np.array([vt, vai]) # (np.log(vai) + np.log(vad))
        return potential, c2

    def seed(self, seed=None):
        _, _ = seeding.np_random(seed)
        return

    def reset(self):
        self.steps = 0
        self.states = self.init_states.copy()
        self.cancerode.drug = np.zeros((1,2))
        self.cancerode.ts = np.array([0, 1])
        normalized_states = np.log10(self.states) / np.log10(self.normalized_coef)
        normalized_df = np.zeros(3)
        df = self.cancerode.forward(t = 0, y = self.states.reshape(-1))
        for i, dx in enumerate(df):
            if dx > 1:
                normalized_df[i] = (np.log10(dx) + 1) / np.log10(self.normalized_coef[i])
            elif dx < -1:
                normalized_df[i] = (-np.log10(-dx) - 1) / np.log10(self.normalized_coef[i])
            else:
                normalized_df[i] = dx / np.log10(self.normalized_coef[i])
        fea = np.concatenate(
            (normalized_states, normalized_df, np.array([self.steps / self.max_episodes_steps]))).reshape(-1)
        # normalized_df = self.cancerode.forward(t = 0, y = self.states.reshape(-1)) # /self.normalized_coef
        # normalized_states = self.states # /self.normalized_coef
        # fea = np.concatenate((normalized_states, normalized_df, np.array([self.steps/self.max_episodes_steps]))).reshape(-1)
        self.t = 0.
        self.dosage_arr = []
        self.penalty_index = np.zeros(2, dtype=np.float)
        self.reward_index = np.zeros(2, dtype=np.float)
        self.metastasis_ad_deque.clear()
        self.metastasis_ai_deque.clear()
        self.leu_on = False
        return fea, self.states


class CancerODEGlv:
    def __init__(self, A, K, pars):
        self.A = A.copy()
        self.K = K
        self.pars = pars
        self.ts = np.array([0, 1])
        self.drug = np.zeros((1,2))
        self.cell_size = 5.236e-10  # 4. / 3. * 3.1415926 * (5e-4cm) ** 3   # cm^3

    def forward(self, t, y):
        r = self.pars[0:2]
        beta = self.pars[2:4]
        Beta = np.zeros((2, 2), dtype=np.float)
        Beta[:, 0] = beta
        phi = self.pars[-4]
        betac = self.pars[(-2):]
        self.A[0, 1] = 1 / (1 + np.exp(-self.pars[-3] * np.array([t / 28 / 12])))
        gamma = 0.25

        x = y[0:2]  # cell count
        p = y[-1]  # psa level
        index = np.where(np.int(t) == self.ts)[0][0] if (np.int(t) <= self.ts).any() else -1
        drug = self.drug[index].reshape(-1)
        dxdt = np.multiply(r * x, (1 - (x @ self.A / self.K) ** phi -drug @ Beta))
        dpdt = betac @ x * self.cell_size - gamma * p
        # mutation: we assume the mutation rate is 4%, with 5 or more mutations wil become new AI cells
        df = np.append(dxdt, dpdt)
        return df
