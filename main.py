# -*- coding=utf-8 -*-
# written by kai zhao
# 
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
import scipy.io as io
import random
import numba
from numba import jit
import time

time_start = time.time()


@jit(nopython=True)
def relu(V):
    theta = -40.0  
    k = 1
    return k * np.log(1 + np.exp((V - theta) / k))


@jit(nopython=True)
def prefer_neurons(perdefinedAmp, prefer, q, sigma):
    dt = 0.001
    I = perdefinedAmp
    sig = sigma
    cortexResponse = np.zeros(3000)
    delta = I * (1. / (1. + np.exp(prefer * (q - 10.) / sig)))

    for i in range(1000):
        cortexResponse[i + 1] = cortexResponse[i] + (delta - cortexResponse[i]) * dt / 0.01
    cortexResponse[1000:3000] = 0.
    return cortexResponse


def optogenetic(I_stim):
    return I_stim


@jit(nopython=True)
def brain_model(f_cortex_L, f_cortex_R, I_D1L_Inhi, I_D2L_Inhi, I_PVL_Inhi):
    T = 3.0
    dt = 0.001
    n = int(T / dt)
    t = np.linspace(0., T, n)
    tau = 0.05
    E = -55.
    E1 = -65.
    E2 = 0.

    V_D1L_contra = np.ones(n) * E
    V_D1L_ipsi = np.ones(n) * E
    V_D2L_contra = np.ones(n) * E
    V_D2L_ipsi = np.ones(n) * E
    V_PVL = np.ones(n) * E

    V_D1R_contra = np.ones(n) * E
    V_D1R_ipsi = np.ones(n) * E
    V_D2R_contra = np.ones(n) * E
    V_D2R_ipsi = np.ones(n) * E
    V_PVR = np.ones(n) * E

    f_D1L_contra = np.ones(n) * relu(-55.)
    f_D1L_ipsi = np.ones(n) * relu(-55.)
    f_D2L_contra = np.ones(n) * relu(-55.)
    f_D2L_ipsi = np.ones(n) * relu(-55.)

    f_D1R_contra = np.ones(n) * relu(-55.)
    f_D1R_ipsi = np.ones(n) * relu(-55.)
    f_D2R_contra = np.ones(n) * relu(-55.)
    f_D2R_ipsi = np.ones(n) * relu(-55.)

    I_D1Lcontra_extern = np.zeros(n)
    I_D1Lcontra_intern = np.zeros(n)
    I_D1Lipsi_extern = np.zeros(n)
    I_D1Lipsi_intern = np.zeros(n)
    I_D2Lcontra_extern = np.zeros(n)
    I_D2Lcontra_intern = np.zeros(n)
    I_D2Lipsi_extern = np.zeros(n)
    I_D2Lipsi_intern = np.zeros(n)

    I_D1Rcontra_extern = np.zeros(n)
    I_D1Rcontra_intern = np.zeros(n)
    I_D1Ripsi_extern = np.zeros(n)
    I_D1Ripsi_intern = np.zeros(n)
    I_D2Rcontra_extern = np.zeros(n)
    I_D2Rcontra_intern = np.zeros(n)
    I_D2Ripsi_extern = np.zeros(n)
    I_D2Ripsi_intern = np.zeros(n)

    I_PVL_self_inhi = np.zeros(n)
    I_D1Lcontra_self_inhi = np.zeros(n)
    I_D1Lipsi_self_inhi = np.zeros(n)
    I_D2Lcontra_self_inhi = np.zeros(n)
    I_D2Lipsi_self_inhi = np.zeros(n)

    I_PVR_self_inhi = np.zeros(n)
    I_D1Rcontra_self_inhi = np.zeros(n)
    I_D1Ripsi_self_inhi = np.zeros(n)
    I_D2Rcontra_self_inhi = np.zeros(n)
    I_D2Ripsi_self_inhi = np.zeros(n)

    w_self_inhi = 0.6
    w_PV_D1D2_Wweights = 29.4
    w_D1Lipsi_D1Lcontra = 0.01
    w_D2Lcontra_D1Lcontra = 0.01
    w_D2Lipsi_D1Lcontra = 0.01
    w_PVL_D1Lcontra = w_PV_D1D2_Wweights

    w_D1Lcontra_D1Lipsi = 0.01
    w_D2Lcontra_D1Lipsi = 0.01
    w_D2Lipsi_D1Lipsi = 0.01
    w_PVL_D1Lipsi = w_PV_D1D2_Wweights

    w_D1Lcontra_D2Lcontra = 0.01
    w_D1Lipsi_D2Lcontra = 0.01
    w_D2Lipsi_D2Lcontra = 0.01
    w_PVL_D2Lcontra = w_PV_D1D2_Wweights

    w_D1Lcontra_D2Lipsi = 0.01
    w_D1Lipsi_D2Lipsi = 0.01
    w_D2Lcontra_D2Lipsi = 0.01
    w_PVL_D2Lipsi = w_PV_D1D2_Wweights

    prep_L = np.zeros(n)
    f_PVL = np.ones(n) * relu(-55.)
    I_PVL_extern = np.zeros(n)
    f_PVR = np.ones(n) * relu(-55.)
    I_PVR_extern = np.zeros(n)
    w_cortexL_PVL = 0.015  # 2.352
    w_cortexR_PVL = 0.015

    w_cortexL_D1Lcontra = 0.80
    w_cortexL_D2Lcontra = 0.70
    w_cortexL_D1Lipsi = 0.12
    w_cortexL_D2Lipsi = 0.06

    w_cortexR_D1Lcontra = 0.051
    w_cortexR_D2Lcontra = 0.04
    w_cortexR_D1Lipsi = 0.60
    w_cortexR_D2Lipsi = 0.456

    w_cortexL_D1Rcontra = 0.051
    w_cortexL_D2Rcontra = 0.04
    w_cortexL_D1Ripsi = 0.60
    w_cortexL_D2Ripsi = 0.456

    w_cortexR_D1Rcontra = 0.80
    w_cortexR_D2Rcontra = 0.70
    w_cortexR_D1Ripsi = 0.12
    w_cortexR_D2Ripsi = 0.06

    sigma = 0.05
    noise_tau = 0.2
    sigma_bis = sigma * np.sqrt(2. / noise_tau)
    sqrtdt = np.sqrt(dt)

    for i in range(n - 1):
        I_PVL_extern[i] = (E2 - V_PVL[i]) * (f_cortex_L[i] * w_cortexL_PVL + f_cortex_R[i] * w_cortexR_PVL)
        I_alpha_ext = I_PVL_extern
        I_PVL_self_inhi[i] = (E1 - V_PVL[i]) * f_PVL[i] * w_self_inhi
        V_PVL[i + 1] = V_PVL[i] + (E - V_PVL[i] + I_alpha_ext[i] + I_PVL_self_inhi[
            i] - I_PVL_Inhi) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_PVL[i + 1] < -65:
            V_PVL[i + 1] = -65
        f_PVL[i + 1] = relu(V_PVL[i + 1])

        I_D1Lcontra_extern[i] = (E2 - V_D1L_contra[i]) * (
                f_cortex_L[i] * w_cortexL_D1Lcontra + f_cortex_R[i] * w_cortexR_D1Lcontra)
        I_alpha_ext = I_D1Lcontra_extern
        I_D1Lcontra_intern[i] = (E1 - V_D1L_contra[i]) * (
                f_D1L_ipsi[i] * w_D1Lipsi_D1Lcontra + f_D2L_contra[i] * w_D2Lcontra_D1Lcontra + f_D2L_ipsi[
            i] * w_D2Lipsi_D1Lcontra + f_PVL[i] * w_PVL_D1Lcontra + f_D1R_contra[i] * 0.0)
        I_alpha_int = I_D1Lcontra_intern
        I_D1Lcontra_self_inhi[i] = (E1 - V_D1L_contra[i]) * f_D1L_contra[i] * w_self_inhi
        V_D1L_contra[i + 1] = V_D1L_contra[i] + (
                E - V_D1L_contra[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D1Lcontra_self_inhi[
            i] - I_D1L_Inhi) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D1L_contra[i + 1] < -65:
            V_D1L_contra[i + 1] = -65
        f_D1L_contra[i + 1] = relu(V_D1L_contra[i + 1])

        I_D1Lipsi_extern[i] = (E2 - V_D1L_ipsi[i]) * (
                f_cortex_L[i] * w_cortexL_D1Lipsi + f_cortex_R[i] * w_cortexR_D1Lipsi)
        I_alpha_ext = I_D1Lipsi_extern
        I_D1Lipsi_intern[i] = (E1 - V_D1L_ipsi[i]) * (
                f_D1L_contra[i] * w_D1Lcontra_D1Lipsi + f_D2L_contra[i] * w_D2Lcontra_D1Lipsi + f_D2L_ipsi[
            i] * w_D2Lipsi_D1Lipsi + f_PVL[i] * w_PVL_D1Lipsi)
        I_alpha_int = I_D1Lipsi_intern
        I_D1Lipsi_self_inhi[i] = (E1 - V_D1L_ipsi[i]) * f_D1L_ipsi[i] * w_self_inhi

        V_D1L_ipsi[i + 1] = V_D1L_ipsi[i] + (E - V_D1L_ipsi[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D1Lipsi_self_inhi[
            i] - I_D1L_Inhi) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D1L_ipsi[i + 1] < -65:
            V_D1L_ipsi[i + 1] = -65

        f_D1L_ipsi[i + 1] = relu(V_D1L_ipsi[i + 1])

        I_D2Lcontra_extern[i] = (E2 - V_D2L_contra[i]) * (
                f_cortex_L[i] * w_cortexL_D2Lcontra + f_cortex_R[i] * w_cortexR_D2Lcontra)
        I_alpha_ext = I_D2Lcontra_extern
        I_D2Lcontra_intern[i] = (E1 - V_D2L_contra[i]) * (
                f_D1L_contra[i] * w_D1Lcontra_D2Lcontra + f_D1L_ipsi[i] * w_D1Lipsi_D2Lcontra + f_D2L_ipsi[
            i] * w_D2Lipsi_D2Lcontra + f_PVL[i] * w_PVL_D2Lcontra + f_D2R_contra[i] * 0.0)
        I_alpha_int = I_D2Lcontra_intern
        I_D2Lcontra_self_inhi[i] = (E1 - V_D2L_contra[i]) * f_D2L_contra[i] * w_self_inhi

        V_D2L_contra[i + 1] = V_D2L_contra[i] + (
                E - V_D2L_contra[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D2Lcontra_self_inhi[
            i] - I_D2L_Inhi) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D2L_contra[i + 1] < -65:
            V_D2L_contra[i + 1] = -65

        f_D2L_contra[i + 1] = relu(V_D2L_contra[i + 1])

        I_D2Lipsi_extern[i] = (E2 - V_D2L_ipsi[i]) * (
                f_cortex_L[i] * w_cortexL_D2Lipsi + f_cortex_R[i] * w_cortexR_D2Lipsi)
        I_alpha_ext = I_D2Lipsi_extern
        I_D2Lipsi_intern[i] = (E1 - V_D2L_ipsi[i]) * (
                f_D1L_contra[i] * w_D1Lcontra_D2Lipsi + f_D1L_ipsi[i] * w_D1Lipsi_D2Lipsi + f_D2L_contra[
            i] * w_D2Lcontra_D2Lipsi + f_PVL[i] * w_PVL_D2Lipsi)
        I_alpha_int = I_D2Lipsi_intern
        I_D2Lipsi_self_inhi[i] = (E1 - V_D2L_ipsi[i]) * f_D2L_ipsi[i] * w_self_inhi

        V_D2L_ipsi[i + 1] = V_D2L_ipsi[i] + (E - V_D2L_ipsi[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D2Lipsi_self_inhi[
            i] - I_D2L_Inhi) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D2L_ipsi[i + 1] < -65:
            V_D2L_ipsi[i + 1] = -65

        f_D2L_ipsi[i + 1] = relu(V_D2L_ipsi[i + 1])

        I_PVR_extern[i] = (E2 - V_PVR[i]) * (f_cortex_L[i] * w_cortexL_PVL + f_cortex_R[i] * w_cortexR_PVL)
        I_alpha_ext = I_PVR_extern
        I_PVR_self_inhi[i] = (E1 - V_PVR[i]) * f_PVR[i] * w_self_inhi
        V_PVR[i + 1] = V_PVR[i] + (E - V_PVR[i] + I_alpha_ext[i] + I_PVR_self_inhi[
            i]) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_PVR[i + 1] < -65:
            V_PVR[i + 1] = -65
        f_PVR[i + 1] = relu(V_PVR[i + 1])

        I_D1Rcontra_extern[i] = (E2 - V_D1R_contra[i]) * (
                f_cortex_L[i] * w_cortexL_D1Rcontra + f_cortex_R[i] * w_cortexR_D1Rcontra)
        I_alpha_ext = I_D1Rcontra_extern
        I_D1Rcontra_intern[i] = (E1 - V_D1R_contra[i]) * (
                f_D1R_ipsi[i] * w_D1Lipsi_D1Lcontra + f_D2R_contra[i] * w_D2Lcontra_D1Lcontra + f_D2R_ipsi[
            i] * w_D2Lipsi_D1Lcontra + f_PVR[i] * w_PVL_D1Lcontra + f_D1L_contra[i] * 0.0)
        I_alpha_int = I_D1Rcontra_intern
        I_D1Rcontra_self_inhi[i] = (E1 - V_D1R_contra[i]) * f_D1R_contra[i] * w_self_inhi

        V_D1R_contra[i + 1] = V_D1R_contra[i] + (
                E - V_D1R_contra[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D1Rcontra_self_inhi[
            i]) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D1R_contra[i + 1] < -65:
            V_D1R_contra[i + 1] = -65
        f_D1R_contra[i + 1] = relu(V_D1R_contra[i + 1])

        I_D1Ripsi_extern[i] = (E2 - V_D1R_ipsi[i]) * (
                f_cortex_L[i] * w_cortexL_D1Ripsi + f_cortex_R[i] * w_cortexR_D1Ripsi)
        I_alpha_ext = I_D1Ripsi_extern
        I_D1Ripsi_intern[i] = (E1 - V_D1R_ipsi[i]) * (
                f_D1R_contra[i] * w_D1Lcontra_D1Lipsi + f_D2R_contra[i] * w_D2Lcontra_D1Lipsi + f_D2R_ipsi[
            i] * w_D2Lipsi_D1Lipsi + f_PVR[i] * w_PVL_D1Lipsi)
        I_alpha_int = I_D1Ripsi_intern
        I_D1Ripsi_self_inhi[i] = (E1 - V_D1R_ipsi[i]) * f_D1R_ipsi[i] * w_self_inhi
        V_D1R_ipsi[i + 1] = V_D1R_ipsi[i] + (E - V_D1R_ipsi[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D1Ripsi_self_inhi[
            i]) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D1R_ipsi[i + 1] < -65:
            V_D1R_ipsi[i + 1] = -65
        f_D1R_ipsi[i + 1] = relu(V_D1R_ipsi[i + 1])

        I_D2Rcontra_extern[i] = (E2 - V_D2R_contra[i]) * (
                f_cortex_L[i] * w_cortexL_D2Rcontra + f_cortex_R[i] * w_cortexR_D2Rcontra)
        I_alpha_ext = I_D2Rcontra_extern
        I_D2Rcontra_intern[i] = (E1 - V_D2R_contra[i]) * (
                f_D1R_contra[i] * w_D1Lcontra_D2Lcontra + f_D1R_ipsi[i] * w_D1Lipsi_D2Lcontra + f_D2R_ipsi[
            i] * w_D2Lipsi_D2Lcontra + f_PVR[i] * w_PVL_D2Lcontra + f_D2L_contra[i] * 0.0)
        I_alpha_int = I_D2Rcontra_intern
        I_D2Rcontra_self_inhi[i] = (E1 - V_D2R_contra[i]) * f_D2R_contra[i] * w_self_inhi
        V_D2R_contra[i + 1] = V_D2R_contra[i] + (
                E - V_D2R_contra[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D2Rcontra_self_inhi[
            i]) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D2R_contra[i + 1] < -65:
            V_D2R_contra[i + 1] = -65
        f_D2R_contra[i + 1] = relu(V_D2R_contra[i + 1])

        I_D2Ripsi_extern[i] = (E2 - V_D2R_ipsi[i]) * (
                f_cortex_L[i] * w_cortexL_D2Ripsi + f_cortex_R[i] * w_cortexR_D2Ripsi)
        I_alpha_ext = I_D2Ripsi_extern
        I_D2Ripsi_intern[i] = (E1 - V_D2R_ipsi[i]) * (
                f_D1R_contra[i] * w_D1Lcontra_D2Lipsi + f_D1R_ipsi[i] * w_D1Lipsi_D2Lipsi + f_D2R_contra[
            i] * w_D2Lcontra_D2Lipsi + f_PVR[i] * w_PVL_D2Lipsi)
        I_alpha_int = I_D2Ripsi_intern
        I_D2Ripsi_self_inhi[i] = (E1 - V_D2R_ipsi[i]) * f_D2R_ipsi[i] * w_self_inhi
        V_D2R_ipsi[i + 1] = V_D2R_ipsi[i] + (E - V_D2R_ipsi[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D2Ripsi_self_inhi[
            i]) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D2R_ipsi[i + 1] < -65:
            V_D2R_ipsi[i + 1] = -65
        f_D2R_ipsi[i + 1] = relu(V_D2R_ipsi[i + 1])
    return f_D1L_contra, f_D1L_ipsi, f_D2L_contra, f_D2L_ipsi, f_D1R_contra, f_D1R_ipsi, f_D2R_contra, f_D2R_ipsi


@jit(nopython=True)
def SNr_model(f_D1L_contra, f_D1L_ipsi, f_D2L_contra, f_D2L_ipsi, f_D1R_contra, f_D1R_ipsi, f_D2R_contra, f_D2R_ipsi):
    T = 3.0
    dt = 0.001
    n = int(T / dt)
    t = np.linspace(0., T, n)
    tau = 0.1
    E = -55.
    E1 = -65.
    E2 = 0.
    V_middle_init = -50.

    w_D2_middle = 0.1
    w_middle_SNr = 0.01
    w_d1_snr = 0.055

    w_snr_self_inhi = 0.01
    w_snr_eachother_inhi = 0.01

    I_SNrL_inhi = np.zeros(n)
    I_SNrL_exci = np.zeros(n)
    I_SNrR_inhi = np.zeros(n)
    I_SNrR_exci = np.zeros(n)
    V_SNrL = np.ones(n) * E
    V_SNrR = np.ones(n) * E

    I_from_SNrL_inhi = np.zeros(n)
    I_SNrL_self_inhi = np.zeros(n)
    I_SNrR_self_inhi = np.zeros(n)
    I_from_SNrR_inhi = np.zeros(n)

    f_SNrL = np.ones(n) * relu(-55.)
    f_SNrR = np.ones(n) * relu(-55.)

    sigma = 0.035
    noise_tau = 0.05
    sigma_bis = sigma * np.sqrt(2. / noise_tau)
    sqrtdt = np.sqrt(dt)
    for i in range(n - 1):
        I_SNrL_inhi[i] = (E1 - V_SNrL[i]) * w_d1_snr * (f_D1L_contra[i] + f_D1L_ipsi[i])
        I_from_SNrR_inhi[i] = (E1 - V_SNrL[i]) * f_SNrR[i] * w_snr_eachother_inhi
        I_SNrL_self_inhi[i] = (E1 - V_SNrL[i]) * w_snr_self_inhi * f_SNrL[i]
        I_SNrL_exci[i] = (E2 - V_SNrL[i]) * w_middle_SNr * (f_D2L_contra[i] + f_D2L_ipsi[i])

        V_SNrL[i + 1] = V_SNrL[i] + (
                E - V_SNrL[i] + I_SNrL_inhi[i] + I_SNrL_exci[i] + I_from_SNrR_inhi[i] + I_SNrL_self_inhi[
            i]) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_SNrL[i + 1] < -65:
            V_SNrL[i + 1] = -65
        f_SNrL[i + 1] = relu(V_SNrL[i + 1])

        I_SNrR_inhi[i] = (E1 - V_SNrR[i]) * w_d1_snr * (f_D1R_contra[i] + f_D1R_ipsi[i])
        I_from_SNrL_inhi[i] = (E1 - V_SNrR[i]) * f_SNrL[i] * w_snr_eachother_inhi
        I_SNrR_self_inhi[i] = (E1 - V_SNrR[i]) * w_snr_self_inhi * f_SNrR[i]
        I_SNrR_exci[i] = (E2 - V_SNrR[i]) * w_middle_SNr * (f_D2R_contra[i] + f_D2R_ipsi[i])

        V_SNrR[i + 1] = V_SNrR[i] + (
                E - V_SNrR[i] + I_SNrR_inhi[i] + I_SNrR_exci[i] + I_from_SNrL_inhi[i] + I_SNrR_self_inhi[
            i]) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_SNrR[i + 1] < -65:
            V_SNrR[i + 1] = -65
        f_SNrR[i + 1] = relu(V_SNrR[i + 1])

    return f_SNrL, f_SNrR


@jit(nopython=True)
def confidence_model(freq):
    dt = 0.001
    sigma = 0.05
    tau = 0.05
    sigma_bis = sigma * np.sqrt(2. / tau)
    sqrtdt = np.sqrt(dt)

    x = random.uniform(0, 1)
    freq = freq - 10
    prob = 0.5 * np.exp(-(np.power(freq, 2) / 5.)) + sigma_bis * sqrtdt * np.random.randn()
    if x < prob:
        select_prefer = 1.
    else:
        select_prefer = -1.
    return select_prefer


def main_loop(stim_value, total_Sessions, Trial_repeat, I_D1L_Inhi, I_D2L_Inhi, I_PVL_Inhi):
    t_start = 300
    t_end = 3000

    freq = [0.0, 2.86, 5.71, 8.57, 11.43, 14.29, 17.14, 20.00]

    arr_rightward = np.array([])
    total_SNrL = np.array([])
    total_SNrR = np.array([])
    for session in range(total_Sessions):
        rightward = np.array([])
        SNrL_var = np.array([])
        SNrR_var = np.array([])
        for i in range(8):
            left_choice = 0
            right_choice = 0
            for trials in range(Trial_repeat):
                prefer_prob = confidence_model(freq[i])

                f_cortex_L = prefer_neurons(20., prefer_prob, freq[i], 0.000001)
                f_cortex_R = prefer_neurons(20., -1. * prefer_prob, freq[i], 0.000001)
                f_cortex_L[0] = 0.
                f_cortex_R[0] = 0.

                f_D1L_contra, f_D1L_ipsi, f_D2L_contra, f_D2L_ipsi, f_D1R_contra, f_D1R_ipsi, f_D2R_contra, f_D2R_ipsi = brain_model(
                    f_cortex_L, f_cortex_R, I_D1L_Inhi, I_D2L_Inhi, I_PVL_Inhi)

                SNrL, SNrR = SNr_model(f_D1L_contra, f_D1L_ipsi, f_D2L_contra, f_D2L_ipsi, f_D1R_contra, f_D1R_ipsi,
                                       f_D2R_contra, f_D2R_ipsi)

                var = SNrL[t_start:t_end] - SNrR[t_start:t_end]
                single_var = np.abs(SNrL[t_start:t_end] - SNrR[t_start:t_end])
                index = np.where(single_var > 0.00)
                if var[index[0][0]] >= 0:
                    left_choice = left_choice + 1
                else:
                    right_choice = right_choice + 1
            rightward = np.append(rightward, right_choice)

        arr_rightward = np.append(arr_rightward, rightward, axis=0)

    arr_rightward = arr_rightward.reshape((total_Sessions, 8))
    locals()['arr_rightward_' + str(stim_value)] = arr_rightward
    io.savemat('arr_rightward_' + str(stim_value) + '.mat',
               {'arr_rightward_' + str(stim_value): eval('arr_rightward_' + str(stim_value))})


num_split = 1
total_Sessions = 22
Trial_repeat = 1000
range_of_stim_I_current = np.linspace(4.5, 4.5, num_split)
print(range_of_stim_I_current)

for i in range(num_split):
    D1L_Inhi = 0
    D2L_Inhi = 0
    PVL_Inhi = 0
    Semi_Inhi = 0
    D1L_Exci = 0
    D2L_Exci = 0

    if D1L_Inhi:
        I_D1L_Inhi = range_of_stim_I_current[i]
    else:
        I_D1L_Inhi = 0.

    if D2L_Inhi:
        I_D2L_Inhi = range_of_stim_I_current[i]
    else:
        I_D2L_Inhi = 0.

    if PVL_Inhi:
        I_PVL_Inhi = range_of_stim_I_current[i]
    else:
        I_PVL_Inhi = 0.

    if Semi_Inhi:
        I_D1L_Inhi = range_of_stim_I_current[i]
        I_D2L_Inhi = range_of_stim_I_current[i]
        I_PVL_Inhi = range_of_stim_I_current[i]
    elif D1L_Inhi or D2L_Inhi or PVL_Inhi:
        I_D1L_Inhi
        I_D2L_Inhi
        I_PVL_Inhi
    else:
        I_D1L_Inhi = 0.
        I_D2L_Inhi = 0.
        I_PVL_Inhi = 0.

    if D1L_Exci:
        I_D1L_Inhi = -range_of_stim_I_current[i]
    elif D1L_Inhi or Semi_Inhi:
        I_D1L_Inhi
    else:
        I_D1L_Inhi = 0.

    if D2L_Exci:
        I_D2L_Inhi = -range_of_stim_I_current[i]
    elif D2L_Inhi or Semi_Inhi:
        I_D2L_Inhi
    else:
        I_D2L_Inhi = 0.

    stim_value = i + 1
    main_loop(stim_value, total_Sessions, Trial_repeat, I_D1L_Inhi, I_D2L_Inhi, I_PVL_Inhi)
plt.show()
time_end = time.time()
print('Time total cost', time_end - time_start, 's')
