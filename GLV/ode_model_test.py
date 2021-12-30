#!/usr/bin/env pyth
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:45:32 2021

@author: michael
"""
from datetime import datetime
import torch
import xitorch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from torch import nn, optim
from LoadData import LoadData
import shutil
from _utils import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Union, Mapping, Any, Sequence, Optional
from xitorch._utils.assertfuncs import assert_fcn_params, assert_runtime
from xitorch._core.pure_function import get_pure_function, make_sibling
from xitorch._impls.integrate.ivp.explicit_rk import rk4_ivp, rk38_ivp
from xitorch._impls.integrate.ivp.adaptive_rk import rk23_adaptive, rk45_adaptive
from xitorch._utils.misc import set_default_option, TensorNonTensorSeparator, TensorPacker
from xitorch._utils.tensor import convert_none_grads_to_zeros
from xitorch._docstr.api_docstr import get_methods_docstr
from xitorch.debug.modes import is_debug_enabled

__all__ = ["solve_ivp"]


def solve_ivp(fcn: Union[Callable[..., torch.Tensor], Callable[..., Sequence[torch.Tensor]]],
              ts: torch.Tensor,
              y0: torch.Tensor,
              params: Sequence[Any] = [],
              bck_options: Mapping[str, Any] = {},
              method: Optional[str] = None,
              **fwd_options) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    r"""
    Solve the initial value problem (IVP) or also commonly known as ordinary
    differential equations (ODE), where given the initial value :math:`\mathbf{y_0}`,
    it then solves

    .. math::

        \mathbf{y}(t) = \mathbf{y_0} + \int_{t_0}^{t} \mathbf{f}(t', \mathbf{y}, \theta)\ \mathrm{d}t'

    Arguments
    ---------
    fcn: callable
        The function that represents dy/dt. The function takes an input of a
        single time ``t`` and tensor ``y`` with shape ``(*ny)`` and
        produce :math:`\mathrm{d}\mathbf{y}/\mathrm{d}t` with shape ``(*ny)``.pip3 insta;;
        The output of the function must be a tensor with shape ``(*ny)`` or
        a list of tensors.
    ts: torch.tensor
        The time points where the value of `y` will be returned.
        It must be monotonically increasing or decreasing.
        It is a tensor with shape ``(nt,)``.
    y0: torch.tensor
        The initial value of ``y``, i.e. ``y(t[0]) == y0``.
        It is a tensor with shape ``(*ny)`` or a list of tensors.
    params: list
        Sequence of other parameters required in the function.
    bck_options: dict
        Options for the backward solve_ivp method. If not specified, it will
        take the same options as fwd_options.
    method: str or None
        Initial value problem solver. If None, it will choose ``"rk45"``.
    **fwd_options
        Method-specific option (see method section below).

    Returns
    -------
    torch.tensor or a list of tensors
        The values of ``y`` for each time step in ``ts``.
        It is a tensor with shape ``(nt,*ny)`` or a list of tensors
    """
    if is_debug_enabled():
        assert_fcn_params(fcn, (ts[0], y0, *params))
    assert_runtime(len(ts.shape) == 1, "Argument ts must be a 1D tensor")

    if method is None:  # set the default method
        method = "rk45"
    fwd_options["method"] = method

    # run once to see if the outputs is a tuple or a single tensor
    is_y0_list = isinstance(y0, list) or isinstance(y0, tuple)
    dydt = fcn(ts[0], y0, *params)
    is_dydt_list = isinstance(dydt, list) or isinstance(dydt, tuple)
    if is_y0_list != is_dydt_list:
        raise RuntimeError("The y0 and output of fcn must both be tuple or a tensor")

    pfcn = get_pure_function(fcn)
    if is_y0_list:
        nt = len(ts)
        roller = TensorPacker(y0)

        @make_sibling(pfcn)
        def pfcn2(t, ytensor, *params):
            ylist = roller.pack(ytensor)
            res_list = pfcn(t, ylist, *params)
            res = roller.flatten(res_list)
            return res

        y0 = roller.flatten(y0)
        res = _SolveIVP.apply(pfcn2, ts, fwd_options, bck_options, len(params), y0, *params, *pfcn.objparams())
        return roller.pack(res)
    else:
        return _SolveIVP.apply(pfcn, ts, fwd_options, bck_options, len(params), y0, *params, *pfcn.objparams())


class _SolveIVP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pfcn, ts, fwd_options, bck_options, nparams, y0, *allparams):
        config = fwd_options
        ctx.bck_config = set_default_option(config, bck_options)

        params = allparams[:nparams]
        objparams = allparams[nparams:]

        orig_method = config.pop("method")
        method = orig_method.lower()
        try:
            solver = {
                "rk4": rk4_ivp,
                "rk38": rk38_ivp,
                "rk23": rk23_adaptive,
                "rk45": rk45_adaptive,
            }[method]
        except KeyError:
            raise RuntimeError("Unknown solve_ivp method: %s" % config["method"])
        yt = solver(pfcn, ts, y0, params, **config)
        # print(yt)

        #########################
        #####multiprocessing#####
        #########################

        # save the parameters for backward
        ctx.param_sep = TensorNonTensorSeparator(allparams, varonly=True)
        tensor_params = ctx.param_sep.get_tensor_params()
        ctx.save_for_backward(ts, y0, *tensor_params)
        ctx.pfcn = pfcn
        ctx.nparams = nparams
        ctx.yt = yt
        ctx.ts_requires_grad = ts.requires_grad

        return yt

    @staticmethod
    def backward(ctx, grad_yt):
        # grad_yt: (nt, *ny)
        nparams = ctx.nparams
        pfcn = ctx.pfcn
        param_sep = ctx.param_sep
        yt = ctx.yt
        ts_requires_grad = ctx.ts_requires_grad

        # restore the parameters
        saved_tensors = ctx.saved_tensors
        ts = saved_tensors[0]
        y0 = saved_tensors[1]
        tensor_params = list(saved_tensors[2:])
        allparams = param_sep.reconstruct_params(tensor_params)  # maybe change the device
        ntensor_params = len(tensor_params)
        params = allparams[:nparams]
        objparams = allparams[nparams:]

        grad_enabled = torch.is_grad_enabled()

        # custom function to evaluate the input `pfcn` based on whether we want
        # to connect the graph or not
        def pfunc2(t, y, tensor_params):
            if not grad_enabled:
                # if graph is not constructed, then use the default tensor_params
                ycopy = y.detach().requires_grad_()  # [yi.detach().requires_grad_() for yi in y]
                tcopy = t.detach().requires_grad_(False)
                f = pfcn(tcopy, ycopy, *params)
                return f, ycopy, tensor_params
            else:
                # if graph is constructed, then use the clone of the tensor params
                # so that infinite loop of backward can be avoided
                tensor_params_copy = [p.clone().requires_grad_() for p in tensor_params]
                ycopy = y.clone().requires_grad_()
                tcopy = t.clone().requires_grad_(False)
                allparams_copy = param_sep.reconstruct_params(tensor_params_copy)
                params_copy = allparams_copy[:nparams]
                objparams_copy = allparams_copy[nparams:]
                with pfcn.useobjparams(objparams_copy):
                    f = pfcn(tcopy, ycopy, *params_copy)
                return f, ycopy, tensor_params_copy

        # slices and indices definitions on the augmented states
        y_index = 0
        dLdy_index = 1
        # dLdt_index = 2
        # dLdt_slice = slice(dLdt_index, dLdt_index+1, None) # [2:3]
        dLdp_slice = slice(-ntensor_params, None, None) if ntensor_params > 0 else slice(0, 0,
                                                                                         None)  # [-ntensor_params:]
        state_size = 2 + ntensor_params  # 3 + ntensor_params
        states = [None for _ in range(state_size)]  # .to(ts.device())

        def new_pfunc(t, states, *tensor_params):
            # t: single-element
            y = states[y_index]
            dLdy = -states[dLdy_index]
            with torch.enable_grad():
                f, y2, tensor_params2 = pfunc2(t, y, tensor_params)
            allgradinputs = ([y2] + list(tensor_params2))
            # allgradinputs = list(tensor_params2)
            allgrads = torch.autograd.grad(f,
                                           inputs=allgradinputs,
                                           grad_outputs=dLdy,
                                           retain_graph=True,
                                           allow_unused=True,
                                           create_graph=torch.is_grad_enabled())  # list of (*ny)
            allgrads = convert_none_grads_to_zeros(allgrads, allgradinputs)
            outs = (
                f,  # dydt
                *allgrads,
            )
            return outs

        ts_flip = ts.flip(0)
        t_flip_idx = -1
        states[y_index] = yt[t_flip_idx]
        states[dLdy_index] = grad_yt[t_flip_idx]
        # states[dLdt_index] = torch.zeros_like(ts[0])
        states[dLdp_slice] = [torch.zeros_like(tp) for tp in tensor_params]
        grad_ts = [None for _ in range(len(ts))] if ts_requires_grad else None

        for i in range(len(ts_flip) - 1):
            # sprint(i)
            # print(states[-1])
            t_flip_idx -= 1
            # ctx.bck_config["methods"] = 'rk4'
            outs = solve_ivp(new_pfunc, ts_flip[i:i + 2], states, tensor_params,
                             fwd_options=ctx.bck_config, bck_options=ctx.bck_config, method="rk4")
            # only take the output for the earliest time
            states = [out[-1] for out in outs]
            states[y_index] = yt[t_flip_idx]
            # gyt is the contribution from the input grad_y
            # gy0 is the propagated gradients from the later time step
            states[dLdy_index] = grad_yt[t_flip_idx] + states[dLdy_index]
        for ii in range(len(states)):
            states[ii] = states[ii].to(ts.device)

        # if ts_requires_grad:
        #     grad_ts[0] = states[dLdt_index].reshape(-1)

        grad_y0 = states[dLdy_index]  # dL/dy0, (*ny)
        if ts_requires_grad:
            grad_ts = torch.cat(grad_ts).reshape(*ts.shape)
        grad_tensor_params = states[dLdp_slice]
        grad_ntensor_params = [None for _ in range(len(allparams) - ntensor_params)]
        grad_params = param_sep.reconstruct_params(grad_tensor_params, grad_ntensor_params)
        return (None, grad_ts, None, None, None, grad_y0, *grad_params)


# docstring completion
ivp_methods = {
    "rk45": rk45_adaptive,
    "rk23": rk23_adaptive,
    "rk4": rk4_ivp,
}
solve_ivp.__doc__ = get_methods_docstr(solve_ivp, ivp_methods)


###################################################
################### GLV model #####################
###################################################

class CancerODEGlv_CPU(xitorch.EditableModule):

    def __init__(self, patientNo, **params):
        # self.r = params["r"]
        self.A = params["A"]
        self.K = params["K"]
        self.pars = params["pars"]
        self.data = LoadData()._Patient_data(patientNo)
        if patientNo == "patient002":
            self.data = self.data[:84]
        if patientNo == "patient046":
            self.data[43:46, 1] -= 10
        if patientNo == "patient056":
            self.data[46, 1] = (self.data[44, 1] + self.data[48, 1]) / 2
        if patientNo == "patient086":
            self.data[1, 1] = (self.data[1, 1] + self.data[8, 1]) / 2
        if patientNo == "patient104":
            self.data = self.data[:(-3)]
        # normalization of drug
        self.CPA = torch.from_numpy(self.data[:, 2]).float()
        self.LEU = torch.from_numpy(self.data[:, 3]).float()
        self.Days = torch.from_numpy(self.data[:, 6] - self.data[0, 6]).float()
        self.OnOff = self.data[:, 5];
        # self.pre_leu()
        self.Response = self.drug_response(
            torch.linspace(start=self.Days[0], end=self.Days[-1], steps=int(self.Days[-1] - self.Days[0]) + 1))
        self.cell_size = 5.236e-10  # 4. / 3. * 3.1415926 * (5e-4cm) ** 3   # cm^3

    def drug_response(self, t):
        drug = torch.zeros((t.shape[0], 2), dtype=torch.float)
        slice0 = torch.bucketize(t, self.Days, right=True) - 1
        slice_75 = torch.where(self.LEU == 7.5)[0]
        dose75_date = self.Days[slice_75]
        slice_225 = torch.where(self.LEU == 22.5)[0]
        dose225_date = self.Days[slice_225]
        slice_30 = torch.where(self.LEU == 30)[0]
        dose30_date = self.Days[slice_30]
        slice_45 = torch.where(self.LEU == 45)[0]
        dose45_date = self.Days[slice_45]

        I0_CPA_dose = self.CPA[slice0]
        I0_LEU_dose = self.LEU[slice0] # LEu dose has 7.5/22.5/30/45, 4 different dosages, for 4/12/16/24 weeks, and we can see that no matter
        dose75 = dose225 = dose30 = dose45 = 1
        _date = -100
        for date in dose75_date.int():
            if abs(date - _date) < 7 or abs(date - _date) == 7:
                dose75 += 1
            else:
                dose75 = 1
            if dose75 == 1:
                temp = torch.zeros(28, dtype = torch.float)  # last 12 weeks
                temp[0:7 * 1] = - 3.75 / 6 * torch.arange(0, 7, 1)
                temp[(7 * 1):(7 * 3)] = (7.5 + 3.75) / (20 - 6) * torch.arange(7, 21, 1) + (
                            7.5 - (7.5 + 3.75) / (20 - 6) * 20)
                temp[(7 * 3):] = 7.5
                I0_LEU_dose[date: (date + 7 * 4)] = temp[0:I0_LEU_dose[date: (date + 7 * 4)].size(0)]
            else:
                I0_LEU_dose[_date: (date + 7 * 4)] = 7.5
            _date = date + 7 * 4
        _date = -100
        for date in dose225_date.int():
            if abs(date - _date) <  7 or abs(date - _date) == 7:
                dose225 += 1
            else:
                dose225 = 1
            if dose225 == 1:
                temp = torch.zeros(7*12, dtype = torch.float) # last 12 weeks
                temp[0:7*1] = - 3.75/6 * torch.arange(0,7,1)
                temp[(7*1):(7*3)] = (7.5 + 3.75)/(20-6) *torch.arange(7, 21, 1) +(7.5 -  (7.5 + 3.75)/(20-6)*20)
                temp[(7*3):] = 7.5
                I0_LEU_dose[date: (date + 7 * 12)] = temp[0:I0_LEU_dose[date: (date + 7 * 12)].size(0)]
            else:
                I0_LEU_dose[_date: (date + 7 * 12)] = 7.5
            _date = date + 7 * 12

        _date = -100
        for date in dose30_date.int():
            if abs(date - _date) <  7 or abs(date - _date) == 7:
                dose30 += 1
            else:
                dose30 = 1
            if dose30 == 1:
                temp = torch.zeros(7*16, dtype = torch.float)  # last 12 weeks
                temp[0:7 * 1] = - 3.75 / 6 * torch.arange(0, 7, 1)
                temp[(7 * 1):(7 * 3)] = (7.5 + 3.75) / (20 - 6) * torch.arange(7, 21, 1) + (
                            7.5 - (7.5 + 3.75) / (20 - 6) * 20)
                temp[(7 * 3):] = 7.5
                I0_LEU_dose[date: (date + 7 * 16)] = temp[0:I0_LEU_dose[date: (date + 7 * 16)].size(0)]
            else:
                I0_LEU_dose[_date: (date + 7 * 16)] = 7.5
            _date = date + 7 * 16
        _date = -100
        for date in dose45_date.int():
            if abs(date - _date) <  7 or abs(date - _date) == 7:
                dose45 += 1
            else:
                dose45 = 1
            if dose45 == 1:
                temp = torch.zeros(7 * 24, dtype = torch.float)  # last 12 weeks
                temp[0:7 * 1] = - 3.75 / 6 * torch.arange(0, 7, 1)
                temp[(7 * 1):(7 * 3)] = (7.5 + 3.75) / (20 - 6) * torch.arange(7, 21, 1) + (
                            7.5 - (7.5 + 3.75) / (20 - 6) * 20)
                temp[(7 * 3):] = 7.5
                I0_LEU_dose[date: (date + 7 * 24)] = temp[0:I0_LEU_dose[date: (date + 7 * 24)].size(0)]
            else:
                I0_LEU_dose[_date: (date + 7 * 24)] = 7.5
            _date = date + 7 * 24

        I0_LEU_dose[torch.cat((torch.where(I0_LEU_dose == 22.5)[0],torch.where(I0_LEU_dose == 30)[0],torch.where(I0_LEU_dose == 45)[0]))] = 7.5
        I0_LEU_dose = I0_LEU_dose / 7.5
        I0_CPA_dose = I0_CPA_dose / 200
        for ii in range(1, I0_LEU_dose.shape[0] - 1):
            if  I0_LEU_dose[ii+1] < 0 and I0_LEU_dose[ii-1] == 1.:
                I0_LEU_dose[ii : (ii + 7 * 3)] = 1.
        drug[:, 0] = I0_CPA_dose
        drug[:, 1] = I0_LEU_dose
        return drug

    def forward(self, t, y):
        r = self.pars[0:2]
        beta = self.pars[2:4]
        Beta = torch.zeros((2, 2), dtype=torch.float)
        Beta[:, 0] = beta
        phi = self.pars[-4]
        betac = self.pars[(-2):]
        A = torch.tensor([1., .5, .5, 1.]).view(2, 2)
        A[0,1] = 1/(1 + torch.exp(-self.pars[-3] * torch.tensor([t /28/12]))) #+= self.pars[-3] * t /28/12 #
        # A[0, 1] = a1; A[1, 0] = a2
        gamma = 0.25  # the half life time for psa is 2.5 days, so each day psa decrease by  25%
        # drug = torch.zeros((t.shape[0], 2), dtype = torch.float)
        # slice0 = torch.bucketize(t, self.Days, right = True) - 1
        # I_CPA_dose = self.CPA[slice0];I_LEU_dose = self.LEU[slice0]
        index = int(t) if len(t.shape) == 0 else t.int().cpu().numpy()
        x = y[0:2]  # cell count
        p = y[-1]  # psa level
        # comp = torch.tensor([x[0]*2 + x[1]*a, x[0]*a + x[1]*1])
        dxdt = torch.multiply(
            r * x, (1 - (x @ A / self.K) ** phi - self.Response[index] @ Beta))  # -
        # self.Response[index, 0] * self.Response[index, 1] * Mu))
        dpdt = betac @ x * self.cell_size - gamma * p  # /(abs(sum(dxdt)) + gamma) /(abs(sum(dxdt)*1000/self.K) + gamma)
        df = torch.zeros(3, dtype=torch.float)
        df[0:2], df[-1] = dxdt, dpdt
        return df

    def getparamnames(self, methodname, prefix=""):
        if methodname == "forward":
            return [prefix + "A",  prefix + "K", prefix + "pars"]  # , prefix+"beta", prefix+"gamma", prefix+"alpha"]
        else:
            raise KeyError()

def MSEloss_weight(inputs, targets, weight):
    return torch.sum(weight * (inputs - targets) ** 2)


def clip_grad(grad, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grad) == 0:
        return torch.tensor(0.)
    device = grad.device
    total_norm = torch.norm(grad.detach(), norm_type).to(device)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grad.detach().mul_(clip_coef.to(grad.device))
    return total_norm

# As for the determination of the initial value, we define if psa =10, the cell count for AD is 1e8

from collections import deque

def train_glv(args, alldata):

    i = args.number
    fail_deque =  deque(maxlen = 10)
    alpha = 0.25 #Alpha[i]
    # under_i = [4, 13, 19, 54, 60, 83, 87, 95, 96, 100, 101, 105]
    # if i in under_i:
    #     alpha = 0.5
    cell_size = 5.236e-10
    fail_flag = False
    if len(str(i)) == 1:
        patientNo = "patient00" + str(i)
    elif len(str(i)) == 2:
        patientNo = "patient0" + str(i)
    else:
        patientNo = "patient" + str(i)
    print(patientNo)

    # PARS = np.array(pd.read_csv("PARS.csv"), dtype=np.float).reshape(-1)
    # if patientNo in ["patient012","patient015" ,"patient006"]:
    #     continue
    data = alldata[patientNo]
    if patientNo == "patient002":
        data = data[:84]
    if patientNo == "patient046":
        data[43:46, 1] -= 10
    if patientNo == "patient056":
        data[46, 1] = (data[44,1] + data[48,1])/2
    if patientNo == "patient086":
        data[1,1] = (data[1,1]+data[8,1])/2
    if patientNo == "patient104":
        data = data[:(-3)]
    Days = data[:, 6] - data[0, 6]
    OnOff = data[:, 5]
    Cycle = data[:,-3]
    PSA = data[:, 1]
    index = np.where(np.isnan(PSA))[0]
    PSA = torch.from_numpy(np.delete(PSA, index)).float()
    DAYS = np.delete(Days, index)
    treatInt = [0.]
    validate_set = deque(maxlen=int(Cycle[-1]))
    for ii in range(1, OnOff.shape[0] - 1):
        if OnOff[ii - 1] == 1 and OnOff[ii] == 0:
            treatInt.append(Days[ii])
        if OnOff[ii - 1] == 0 and OnOff[ii] == 1:
            treatInt.append(Days[ii])
    treatInt.append(Days[-1])
    slicing = np.digitize(treatInt, DAYS, right=True)
    for kk in np.arange(slicing.shape[0]-1, step = 2):
        if kk + 2 < slicing.shape[0]:
            loo = np.random.choice(DAYS[slicing[kk] + 1:slicing[kk + 2]])
        elif kk + 2 >= slicing.shape[0] and DAYS[slicing[kk]+1:].size != 0:
            loo = np.random.choice(DAYS[slicing[kk]+1:])
        else:
            break
        validate_set.append(loo)
    validate_days = np.array(validate_set, dtype=np.int32)
    validate_psa = PSA[np.isin(DAYS, validate_set)].detach().numpy()
    train_days = DAYS[~np.isin(DAYS, validate_set)]
    train_slice = np.digitize(treatInt, train_days, right=True)
    train_psa = PSA[~np.isin(DAYS, validate_set)]
    # 5e+8 5cm^3 = 5000mm^3
    mean_v = 5
    mean_psa = 22.1
    K1 = 1.1 * mean_v * (max(PSA)/mean_psa)/cell_size  # 2e+11
    K2 = alpha * K1
    K = torch.tensor([K1, K2]) #1.1 * mean_v * (max(PSA)/mean_psa)/cell_size #
    # r = torch.tensor([1.2, 1.5], dtype = torch.float) # AD is 2 times more responsive compared to AI cell
    # e = torch.tensor([[0.8], [0.9]], dtype = torch.float) # AD is 5 times more competitive effects exert than AI cells
    A = torch.tensor([1., .5, 0.5, 1.], dtype=torch.float).view(2, 2)
    inputs = torch.linspace(start=Days[0], end=Days[-1], steps=int(Days[-1] - Days[0]) + 1, dtype=torch.float)
    # criterion = nn.MSELoss()
    # learning_rate = torch.tensor([.00001, .00001, .001, .001, .0001, .00001, .001, .001])
    # momentum = 0.98
    # velocity = 0.
    # pars = torch.tensor([0.05, 0.05, 2, 1, 1., -0.01, 1., 1.]).float().requires_grad_()
    parsdir = "./retrain-sigmoid/model_pars"
    parslist = os.listdir(parsdir)
    PARS = torch.zeros(8, dtype = torch.float)
    # reading the ode parameters and the initial/terminal states
    for file in parslist:
        pars_df = pd.read_csv(parsdir + '/' + file)
        pars = torch.from_numpy(np.array(pars_df.loc[4, ~np.isnan(pars_df.loc[4, :])])).float()
        PARS += pars

    pars = PARS/len(parslist)
    pars[-3] = 0
    # if patientNo in ["patient063", "patient075", "patient083", "patient050", "patient036", "patient088", "patient105", "patient087", "patient095", "patient101",
    #                  "patient001", "patient003", "patient004", "patient011", "patient013", "patient046", "patient063", "patient036", "patient062", "patient046",
    #                  "patient013", "patient016", "patient017", "patient056", "patient071", "patient078", "patient079", "patient091", "patient104", "patient105",
    #                  "patient032", "patient083", "patient092"]:
    #     pars_df = pd.read_csv("./retrain-sigmoid/model_pars" + '/Args_' + patientNo + ".csv")
    #     pars = torch.from_numpy(np.array(pars_df.loc[4, ~np.isnan(pars_df.loc[4, :])])).float()
    pars = pars.requires_grad_()
    inits_pars = pars.detach().numpy()
    loss_deque = deque(maxlen=10)
    loss = torch.tensor([10000], dtype=torch.float)
    Epoch = 10000
    best_loss = 10000
    best_pars = pars.detach().numpy()
    cancerode = CancerODEGlv_CPU(patientNo, A=A, K=K, pars=pars)
    optimizer = torch.optim.Adam([cancerode.pars], lr = .001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size =1000, gamma=0.8)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=15)
    if not os.path.exists("./analysis-sigmoid"):
        os.mkdir("./analysis-sigmoid")
    if not os.path.exists("./analysis-sigmoid/model_infos/" + patientNo):
        os.makedirs("./analysis-sigmoid/model_infos/" + patientNo)
    if not os.path.exists("./analysis-sigmoid/model_plots/" + patientNo):
        # shutil.rmtree("./retrain-sigmoid/model_plots/" + patientNo)
        os.makedirs("./analysis-sigmoid/model_plots/" + patientNo)
    if not os.path.exists("./analysis-sigmoid/model_pars/" + patientNo):
        os.makedirs("./analysis-sigmoid/model_pars/" + patientNo)
    if not os.path.exists("./analysis-sigmoid/model_validate"):
        os.makedirs("./analysis-sigmoid/model_validate")
    log = []
    t = datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = "./analysis-sigmoid/model_infos" + '/' + str(patientNo) + '/' + str(t) + "/" + str(args.t)
    writer = SummaryWriter(log_dir=summary_dir)
    for epoch in range(Epoch):
        Init = torch.tensor([mean_v/mean_psa * PSA[0]/cell_size, 1e-4 * K2, PSA[0]], dtype=torch.float)
        _loss = loss.detach().numpy()
        _pars = pars.detach().numpy()
        loss = torch.zeros(1, dtype=torch.float)

        res = Init.detach().numpy().reshape(1, -1)
        optimizer.zero_grad()
        for ii in range(len(treatInt) - 1):
            if ii == len(treatInt) - 2:
                INPUTS = inputs[int(treatInt[ii]):]
            else:
                INPUTS = inputs[int(treatInt[ii]):int(treatInt[ii + 1])]
            ts = INPUTS.requires_grad_(False)
            OUT = solve_ivp(cancerode.forward, ts=ts, y0=Init, params=(), atol=1e-08, rtol=1e-05)
            res = np.append(res, OUT.detach().numpy(), axis=0)
            d = train_days[train_slice[ii]:train_slice[ii + 1]]
            slicing1 = np.digitize(d, INPUTS, right=True)
            EST_PSA = OUT[slicing1, -1]
            EST_PSA[torch.isnan(EST_PSA)] = 1000
            psa = train_psa[train_slice[ii]: train_slice[ii + 1]]
            if ii == 0:
                psa[0] = Init[-1]
            Init = OUT[-1]
            weights = 1.
            n_psa = psa.shape[0]
            if ii % 2 == 0:
                weights = torch.linspace(start=1, end=n_psa, steps=n_psa)
            if ii % 2 == 1:
                weights = torch.linspace(start=n_psa, end=1, steps=n_psa)
            weights = weights/sum(weights)
            loss = MSEloss_weight(EST_PSA, psa, weights) #+ loss
            loss.backward(retain_graph=True if ii != len(treatInt) - 2 else False)

        # pars_grad = torch.autograd.grad(loss, (cancerode.pars,))[0]
        # clip_grad(pars_grad, max_norm=10);
        # velocity = momentum * velocity - learning_rate * pars_grad # pars_grad
        # pars = pars.add(velocity)  # .detach().requires_grad_()
        # optimizer.zero_grad()
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(pars, max_norm = 100 * (1.001 - epoch/Epoch))
        optimizer.step()
        with torch.no_grad():
            pars[:2].clamp_( min = 5e-3, max = 1e-1)
            pars[2:4].clamp_( min=0)
            pars[-2:].clamp_( min=1e-3)
        scheduler.step()
        # cancerode.pars = pars
        pars_grad = pars.grad
        flag_pars = pars_grad.detach().numpy()[-3]
        # avoiding over-fitted
        loss_deque.append(_loss)
        print(pars.detach().numpy())
        loss_array = np.array(loss_deque, dtype=np.float).reshape(-1)
        if epoch > 10:
            loss_decay_mask = sum((loss_array[1:] - loss_array[:-1]) / loss_array[:-1] > 0)

        if loss.detach().numpy().item() < best_loss:
            best_loss = loss.detach().numpy().item()
            best_pars = pars.detach().numpy()

        with torch.no_grad():
            res = res[1:]
            ad = res[:, 0]
            ai = res[:, 1]
            p = res[:, 2]
            val_psa = p[validate_days]
            validate_loss = np.mean((val_psa - validate_psa) ** 2)
            # log.append([epoch, loss.detach().numpy().item(), validate_loss])
            writer.add_scalar('Loss', loss.detach().numpy().item(), epoch)
            writer.add_scalar('V-Loss', validate_loss, epoch)
            # if epoch % 50 == 0:
            #     file_writing_obj = open('./analysis-sigmoid/model_infos/infos-' + patientNo + "-" + str(args.t) + '.txt', 'w')
            #     file_writing_obj.write(str(log))
            #     file_writing_obj.close()
                #print('Epoch: {} \t Loss: {:.2f} \t Val_Loss: {:.2f}'.format(epoch, loss.detach().numpy().item(), validate_loss))
            if epoch % 100 == 0:
                # x = inputs.detach().numpy()
                # plt.scatter(DAYS, PSA, color="black", marker="*", alpha=0.6)
                # plt.plot(x, p, color="black", linestyle="-", linewidth=1)
                # plt.xlabel("Time (Days)" )
                # plt.ylabel("PSA level (ug/ml)")
                # plt.savefig("./analysis-sigmoid/model_plots/"+patientNo+"/PSA_" + str(args.t) + "-" + patientNo + "_" + str(epoch) + ".png", dpi=100)
                # # plt.show()
                # plt.close()
                # plt.plot(x, ad, color="black", linestyle="--", linewidth=1, label="AD")
                # plt.plot(x, ai, color="black", linestyle="-.", linewidth=1, label="AI")
                # plt.xlabel("Time (Days)")
                # plt.ylabel("Cell counts")
                # plt.legend(loc='upper right')
                # plt.savefig("./analysis-sigmoid/model_plots/"+patientNo+"/Cell_All_" + str(args.t) + "-" + patientNo + "_" + str(epoch) + ".png", dpi=100)
                # # plt.show()
                # plt.close()
                # plt.plot(x, ai, color="black", linestyle="-.", linewidth=1, label="AI")
                # plt.xlabel("Time (Days)")
                # plt.ylabel("Cell counts")
                # plt.savefig("./analysis-sigmoid/model_plots/"+patientNo+"/Cell_AI_" + str(args.t) + "-" + patientNo + "_" + str(epoch) + ".png", dpi=100)
                # # plt.show()
                # plt.close()
                A1 = A.detach().numpy().reshape(-1)
                K1 = K.detach().numpy()
                terminate = res[-1]
                init = Init.detach().numpy()
                states1 = np.append(init, terminate)
                pars_detach = pars.detach().numpy()
                plist = [A1, K1, states1, pars_detach, best_pars]
                plist_df = pd.DataFrame(plist)
                plist_df.to_csv("./analysis-sigmoid/model_pars/"+patientNo+"/Args_" + str(args.t) + "-" + patientNo + ".csv",
                                index=False)
        if epoch > 2000:
            early_stopping(validate_loss, pars)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    Init = torch.tensor([mean_v/mean_psa * PSA[0]/cell_size, 1e-4 * K2, PSA[0]], dtype=torch.float)
    cancerode = CancerODEGlv_CPU(patientNo, A=A, K=K, pars=pars)
    out = solve_ivp(cancerode.forward, ts=inputs, y0=Init, params=(), atol=1e-08, rtol=1e-05)
    ad = out[:, 0].detach().numpy()
    ai = out[:, 1].detach().numpy()
    psa = out[:, -1].detach().numpy()
    pred_validate_psa = psa[validate_days]
    validate_loss = np.array([sum((pred_validate_psa - validate_psa)**2)], dtype = np.float)
    validate_list = [validate_psa, pred_validate_psa, validate_loss]
    validate_df = pd.DataFrame(validate_list, index = ["true", 'predict', 'loss'])
    validate_df.to_csv("./analysis-sigmoid/model_validate/" + patientNo + "/validate_" + str(args.t) + "-" + patientNo + ".csv", index=True)
    print(validate_loss)
    x = inputs.numpy()
    plt.scatter(DAYS, PSA, color="black", marker="*", alpha=0.6)

    plt.plot(x, psa, color="black", linestyle="-", linewidth=1)
    plt.xlabel("Time (Days)")
    plt.ylabel("PSA level (ug/ml)")
    plt.savefig("./analysis-sigmoid/model_plots/"+patientNo+"/Final_PSA_" + str(args.t) + "-" + patientNo + ".png", dpi=300)
    plt.close()

    plt.plot(x, ad, color="black", linestyle="--", linewidth=1, label="AD")
    plt.plot(x, ai, color="black", linestyle="-.", linewidth=1, label="AI")
    plt.xlabel("Time (Days)")
    plt.ylabel("Cell counts")
    plt.legend(loc='upper right')
    plt.savefig("./analysis-sigmoid/model_plots/"+patientNo+"/Final_Cell_" + str(args.t) + "-" + patientNo + ".png", dpi=300)
    plt.close()

    # competition strength
    c = (out[:, :2] @ A[:, 1] / K[1]) ** pars[-3]
    cc = c.detach().numpy()
    plt.plot(x, cc, color="b", linestyle="-", linewidth=1, label="Competition for AI")
    plt.savefig("./analysis-sigmoid/model_plots/"+patientNo+"/Final_Competition_" + str(args.t) + "-" + patientNo + ".png", dpi=300)
    plt.close()

    A = A.detach().numpy().reshape(-1)
    K = K.detach().numpy()
    terminate = out[-1].detach().numpy()
    init = Init.detach().numpy()
    states = np.append(init, terminate)
    pars_detach = pars.detach().numpy()
    plist = [A, K, states, pars_detach, best_pars]
    plist_df = pd.DataFrame(plist)
    plist_df.to_csv("./analysis-sigmoid/model_pars/"+patientNo+"/Args_" + str(args.t) + "-" + patientNo + ".csv", index=False)


parser = argparse.ArgumentParser(description='Patient arguments')
parser.add_argument('--number', '-n', help='Patient No., int type, requested', default=13, type = int)
parser.add_argument('--t',  default=0, type = int)
args = parser.parse_args()

if __name__ == "__main__":
    alldata =  LoadData().Double_Drug()
    train_glv(args, alldata)

