#!/usr/bin/env pyth
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:45:32 2021

@author: michael
"""

import torch
import xitorch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from torch import nn, optim
from .LoadData import LoadData
import shutil

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
        self.cpa = torch.tensor([0, 50, 100, 150, 200], dtype=torch.float)
        self.leu = torch.tensor([0, 7.5], dtype=torch.float)
        self._action_set = torch.stack((self.cpa.repeat(2), torch.sort(self.leu.repeat(5))[0]), dim=1)
        self.CPA = torch.from_numpy(self.data[:, 2]).float()
        self.LEU = torch.from_numpy(self.data[:, 3]).float()
        self.__dose = np.array([self.CPA[-2], self.LEU[-2]])
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

    def new_dose(self, t, action):
        dose = self._action_set[action]
        slicing = int(t)
        drug = torch.zeros(2)
        if self.__dose[1] != 0:
            if dose[1] != 0:
                drug[1] = 1
        elif self.__dose[1] == 0:
            if dose[1] != 0:
                temp = torch.zeros(7 * 4, dtype=torch.float)  # last 4 weeks
                temp[0:7 * 1] = - 3.75 / 6 * torch.arange(0, 7, 1)
                temp[(7 * 1):(7 * 3)] = (7.5 + 3.75) / (20 - 6) * torch.arange(7, 21, 1) + (
                        7.5 - (7.5 + 3.75) / (20 - 6) * 20)
                temp[(7 * 3):] = 7.5
                drug[1] = temp[slicing]/7.5
        drug[0] = dose[0]/200
        self.__dose = dose
        return drug

    def forward(self, t, y, action = None, ending = None):
        r = self.pars[0:2]
        beta = self.pars[2:4]
        Beta = torch.zeros((2, 2), dtype=torch.float)
        Beta[:, 0] = beta
        phi = self.pars[-4]
        betac = self.pars[(-2):]
        A = torch.tensor([1., .5, .5, 1.]).view(2, 2)
        A[0,1] = 1/(1 + torch.exp(-self.pars[-3] * torch.tensor([t /28/12])))
        gamma = 0.25  # the half life time for psa is 2.5 days, so each day psa decrease by  25%
        index = int(t) if len(t.shape) == 0 else t.int().cpu().numpy()
        drug = self.Response[index] if action is None else self.new_dose(t - ending, action)
        x = y[0:2]  # cell count
        p = y[-1]  # psa level
        dxdt = torch.multiply(
            r * x, (1 - (x @ A / self.K) ** phi - drug @ Beta))  # -
        dpdt = betac @ x * self.cell_size - gamma * p
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

# As for the determination of the initial value, we define if psa =10, the cell count for AD is 1