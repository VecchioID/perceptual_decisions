# -*- coding:utf-8 -*-
"""
This is souce code for Cui, L., Tang, S., Zhao, K., Pan, J., Zhang, Z., Si, B., & Xu, N. L. (2021).
Asymmetrical choice-related ensemble activity in direct and indirect-pathway striatal neurons drives perceptual decisions. bioRxiv.
https://www.biorxiv.org/content/10.1101/2021.11.16.468594v2.full.pdf
"""
import numpy as np
from numpy.core.fromnumeric import mean
import scipy.io as io
import random
import numba
from numba import jit
import time


@jit(nopython=True)
def transfer_function_g(V, k=1):  # k = 1 for spiny projection neurons, k = 5 represents interneurons in the striatum.
    theta = -40.0  # discharge threshold
    return k * np.log(1 + np.exp((V - theta)))


@jit(nopython=True)
def rate_ctx(peak_rate, perceptual_decision, q):
    # Equation 3
    dt = 0.001
    I = peak_rate
    sig = 0.000001  # response sensitivity
    cortexResponse = np.zeros(3000)
    delta = I * (1. / (1. + np.exp(perceptual_decision * (q - 10.) / sig)))

    # simulate 1 second cortex activity
    s_stop = 1000
    s_end  = 3000
    for i in range(s_stop):
        cortexResponse[i + 1] = cortexResponse[i] + (delta - cortexResponse[i]) * dt / 0.01
    cortexResponse[s_stop:s_end] = 0.
    return cortexResponse


@jit(nopython=True)
def simu_params_global():
    T = 3.0  # Total time for single trail is 3 seconds
    dt = 0.001  # Time interval for simulation
    n = int(T / dt)
    t = np.linspace(0., T, n)
    tau = 0.05
    E = -55.
    E1 = -65.
    E2 = 0.
    return T, dt, n, t, tau, E, E1, E2


@jit(nopython=True)
def neuron_params_init(n, E):
    # direct - and indirect - pathway spiny projection neurons params At the initial time, the parameters of neurons
    # may not be set to the following ideal values, but for the convenience of calculation and without affecting the
    # final results, the initial values are set as follows in the code.
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

    f_D1L_contra = np.ones(n) * transfer_function_g(E)
    f_D1L_ipsi = np.ones(n) * transfer_function_g(E)
    f_D2L_contra = np.ones(n) * transfer_function_g(E)
    f_D2L_ipsi = np.ones(n) * transfer_function_g(E)

    f_D1R_contra = np.ones(n) * transfer_function_g(E)
    f_D1R_ipsi = np.ones(n) * transfer_function_g(E)
    f_D2R_contra = np.ones(n) * transfer_function_g(E)
    f_D2R_ipsi = np.ones(n) * transfer_function_g(E)

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

    return V_D1L_contra, V_D1L_ipsi, V_D2L_contra, V_D2L_ipsi, V_PVL, \
           V_D1R_contra, V_D1R_ipsi, V_D2R_contra, V_D2R_ipsi, V_PVR, \
           f_D1L_contra, f_D1L_ipsi, f_D2L_contra, f_D2L_ipsi, \
           f_D1R_contra, f_D1R_ipsi, f_D2R_contra, f_D2R_ipsi, \
           I_D1Lcontra_extern, I_D1Lcontra_intern, I_D1Lipsi_extern, I_D1Lipsi_intern, \
           I_D2Lcontra_extern, I_D2Lcontra_intern, I_D2Lipsi_extern, I_D2Lipsi_intern, \
           I_D1Rcontra_extern, I_D1Rcontra_intern, I_D1Ripsi_extern, I_D1Ripsi_intern, \
           I_D2Rcontra_extern, I_D2Rcontra_intern, I_D2Ripsi_extern, I_D2Ripsi_intern, \
           I_PVL_self_inhi, \
           I_D1Lcontra_self_inhi, I_D1Lipsi_self_inhi, \
           I_D2Lcontra_self_inhi, I_D2Lipsi_self_inhi, \
           I_PVR_self_inhi, \
           I_D1Rcontra_self_inhi, I_D1Ripsi_self_inhi, \
           I_D2Rcontra_self_inhi, I_D2Ripsi_self_inhi


@jit(nopython=True)
def network(f_cortex_L, f_cortex_R, I_D1L_Inhi, I_D2L_Inhi, I_PVL_Inhi):
    T, dt, n, t, tau, E, E1, E2 = simu_params_global()

    V_D1L_contra, V_D1L_ipsi, V_D2L_contra, V_D2L_ipsi, V_PVL, \
    V_D1R_contra, V_D1R_ipsi, V_D2R_contra, V_D2R_ipsi, V_PVR, \
    f_D1L_contra, f_D1L_ipsi, f_D2L_contra, f_D2L_ipsi, \
    f_D1R_contra, f_D1R_ipsi, f_D2R_contra, f_D2R_ipsi, \
    I_D1Lcontra_extern, I_D1Lcontra_intern, I_D1Lipsi_extern, I_D1Lipsi_intern, \
    I_D2Lcontra_extern, I_D2Lcontra_intern, I_D2Lipsi_extern, I_D2Lipsi_intern, \
    I_D1Rcontra_extern, I_D1Rcontra_intern, I_D1Ripsi_extern, I_D1Ripsi_intern, \
    I_D2Rcontra_extern, I_D2Rcontra_intern, I_D2Ripsi_extern, I_D2Ripsi_intern, \
    I_PVL_self_inhi, \
    I_D1Lcontra_self_inhi, I_D1Lipsi_self_inhi, \
    I_D2Lcontra_self_inhi, I_D2Lipsi_self_inhi, \
    I_PVR_self_inhi, \
    I_D1Rcontra_self_inhi, I_D1Ripsi_self_inhi, \
    I_D2Rcontra_self_inhi, I_D2Ripsi_self_inhi = neuron_params_init(n, E)

    # set weights
    w_self_inhi           = 0.6
    w_PV_D1D2             = 5.88
    w_D1Lipsi_D1Lcontra   = 0.01
    w_D2Lcontra_D1Lcontra = 0.01
    w_D2Lipsi_D1Lcontra   = 0.01
    w_PVL_D1Lcontra       = w_PV_D1D2

    w_D1Lcontra_D1Lipsi   = 0.01
    w_D2Lcontra_D1Lipsi   = 0.01
    w_D2Lipsi_D1Lipsi     = 0.01
    w_PVL_D1Lipsi         = w_PV_D1D2

    w_D1Lcontra_D2Lcontra = 0.01
    w_D1Lipsi_D2Lcontra   = 0.01
    w_D2Lipsi_D2Lcontra   = 0.01
    w_PVL_D2Lcontra       = w_PV_D1D2

    w_D1Lcontra_D2Lipsi   = 0.01
    w_D1Lipsi_D2Lipsi     = 0.01
    w_D2Lcontra_D2Lipsi   = 0.01
    w_PVL_D2Lipsi         = w_PV_D1D2

    f_PVL = np.ones(n) * transfer_function_g(E, k=5)
    f_PVR = np.ones(n) * transfer_function_g(E, k=5)
    I_PVL_extern  = np.zeros(n)
    I_PVR_extern  = np.zeros(n)
    w_cortexL_PVL = 0.015
    w_cortexR_PVL = 0.015

    w_cortexL_D1Lcontra = 0.80
    w_cortexL_D2Lcontra = 0.70
    w_cortexL_D1Lipsi   = 0.12
    w_cortexL_D2Lipsi   = 0.06

    w_cortexR_D1Lcontra = 0.051
    w_cortexR_D2Lcontra = 0.04
    w_cortexR_D1Lipsi   = 0.60
    w_cortexR_D2Lipsi   = 0.456

    w_cortexL_D1Rcontra = 0.051
    w_cortexL_D2Rcontra = 0.04
    w_cortexL_D1Ripsi   = 0.60
    w_cortexL_D2Ripsi   = 0.456

    w_cortexR_D1Rcontra = 0.80
    w_cortexR_D2Rcontra = 0.70
    w_cortexR_D1Ripsi   = 0.12
    w_cortexR_D2Ripsi   = 0.06

    sigma = 0.05
    noise_tau = 0.2
    sigma_bis = sigma * np.sqrt(2. / noise_tau)
    sqrtdt = np.sqrt(dt)

    for i in range(n - 1):
        # ===========================
        # ===== left hemisphere  ====
        # ===========================
        # f_PVL
        I_PVL_extern[i]    = (E2 - V_PVL[i]) * (f_cortex_L[i] * w_cortexL_PVL + f_cortex_R[i] * w_cortexR_PVL)
        I_alpha_ext        = I_PVL_extern
        I_PVL_self_inhi[i] = (E1 - V_PVL[i]) * f_PVL[i] * w_self_inhi
        V_PVL[i + 1]       = V_PVL[i] + (E - V_PVL[i] + I_alpha_ext[i] + I_PVL_self_inhi[i] - I_PVL_Inhi) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_PVL[i + 1]    < -65:
            V_PVL[i + 1]   = -65
        f_PVL[i + 1]       = transfer_function_g(V_PVL[i + 1], k=5)

        # f_D1L_contra
        I_D1Lcontra_extern[i]    = (E2 - V_D1L_contra[i]) * (f_cortex_L[i] * w_cortexL_D1Lcontra + f_cortex_R[i] * w_cortexR_D1Lcontra)
        I_alpha_ext              = I_D1Lcontra_extern
        I_D1Lcontra_intern[i]    = (E1 - V_D1L_contra[i]) * (f_D1L_ipsi[i] * w_D1Lipsi_D1Lcontra + f_D2L_contra[i] * w_D2Lcontra_D1Lcontra + f_D2L_ipsi[i] * w_D2Lipsi_D1Lcontra + f_PVL[i] * w_PVL_D1Lcontra)
        I_alpha_int              = I_D1Lcontra_intern
        I_D1Lcontra_self_inhi[i] = (E1 - V_D1L_contra[i]) * f_D1L_contra[i] * w_self_inhi
        V_D1L_contra[i + 1]      = V_D1L_contra[i] + (E - V_D1L_contra[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D1Lcontra_self_inhi[i] - I_D1L_Inhi) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D1L_contra[i + 1]   < -65:
            V_D1L_contra[i + 1]  = -65
        f_D1L_contra[i + 1]      = transfer_function_g(V_D1L_contra[i + 1])

        # f_D1L_ipsi
        I_D1Lipsi_extern[i]    = (E2 - V_D1L_ipsi[i]) * (f_cortex_L[i] * w_cortexL_D1Lipsi + f_cortex_R[i] * w_cortexR_D1Lipsi)
        I_alpha_ext            = I_D1Lipsi_extern
        I_D1Lipsi_intern[i]    = (E1 - V_D1L_ipsi[i]) * (f_D1L_contra[i] * w_D1Lcontra_D1Lipsi + f_D2L_contra[i] * w_D2Lcontra_D1Lipsi + f_D2L_ipsi[i] * w_D2Lipsi_D1Lipsi + f_PVL[i] * w_PVL_D1Lipsi)
        I_alpha_int            = I_D1Lipsi_intern
        I_D1Lipsi_self_inhi[i] = (E1 - V_D1L_ipsi[i]) * f_D1L_ipsi[i] * w_self_inhi
        V_D1L_ipsi[i + 1]      = V_D1L_ipsi[i] + (E - V_D1L_ipsi[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D1Lipsi_self_inhi[i] - I_D1L_Inhi) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D1L_ipsi[i + 1]   < -65:
            V_D1L_ipsi[i + 1]  = -65
        f_D1L_ipsi[i + 1]      = transfer_function_g(V_D1L_ipsi[i + 1])

        # f_D2L_contra
        I_D2Lcontra_extern[i]    = (E2 - V_D2L_contra[i]) * (f_cortex_L[i] * w_cortexL_D2Lcontra + f_cortex_R[i] * w_cortexR_D2Lcontra)
        I_alpha_ext              = I_D2Lcontra_extern
        I_D2Lcontra_intern[i]    = (E1 - V_D2L_contra[i]) * (f_D1L_contra[i] * w_D1Lcontra_D2Lcontra + f_D1L_ipsi[i] * w_D1Lipsi_D2Lcontra + f_D2L_ipsi[i] * w_D2Lipsi_D2Lcontra + f_PVL[i] * w_PVL_D2Lcontra)
        I_alpha_int              = I_D2Lcontra_intern
        I_D2Lcontra_self_inhi[i] = (E1 - V_D2L_contra[i]) * f_D2L_contra[i] * w_self_inhi
        V_D2L_contra[i + 1]      = V_D2L_contra[i] + (E - V_D2L_contra[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D2Lcontra_self_inhi[i] - I_D2L_Inhi) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D2L_contra[i + 1]   < -65:
            V_D2L_contra[i + 1]  = -65
        f_D2L_contra[i + 1]      = transfer_function_g(V_D2L_contra[i + 1])

        # f_D2L_ipsi
        I_D2Lipsi_extern[i]    = (E2 - V_D2L_ipsi[i]) * (f_cortex_L[i] * w_cortexL_D2Lipsi + f_cortex_R[i] * w_cortexR_D2Lipsi)
        I_alpha_ext            = I_D2Lipsi_extern
        I_D2Lipsi_intern[i]    = (E1 - V_D2L_ipsi[i]) * (f_D1L_contra[i] * w_D1Lcontra_D2Lipsi + f_D1L_ipsi[i] * w_D1Lipsi_D2Lipsi + f_D2L_contra[i] * w_D2Lcontra_D2Lipsi + f_PVL[i] * w_PVL_D2Lipsi)
        I_alpha_int            = I_D2Lipsi_intern
        I_D2Lipsi_self_inhi[i] = (E1 - V_D2L_ipsi[i]) * f_D2L_ipsi[i] * w_self_inhi
        V_D2L_ipsi[i + 1]      = V_D2L_ipsi[i] + (E - V_D2L_ipsi[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D2Lipsi_self_inhi[i] - I_D2L_Inhi) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D2L_ipsi[i + 1]   < -65:
            V_D2L_ipsi[i + 1]  = -65
        f_D2L_ipsi[i + 1]      = transfer_function_g(V_D2L_ipsi[i + 1])

        # =============================
        # ===== right hemisphere  =====
        # =============================
        # f_PVR
        I_PVR_extern[i]    = (E2 - V_PVR[i]) * (f_cortex_L[i] * w_cortexL_PVL + f_cortex_R[i] * w_cortexR_PVL)
        I_alpha_ext        = I_PVR_extern
        I_PVR_self_inhi[i] = (E1 - V_PVR[i]) * f_PVR[i] * w_self_inhi
        V_PVR[i + 1]       = V_PVR[i] + (E - V_PVR[i] + I_alpha_ext[i] + I_PVR_self_inhi[i]) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_PVR[i + 1]    < -65:
            V_PVR[i + 1]   = -65
        f_PVR[i + 1]       = transfer_function_g(V_PVR[i + 1], k=5)

        # f_D1R_contra
        I_D1Rcontra_extern[i]    = (E2 - V_D1R_contra[i]) * (f_cortex_L[i] * w_cortexL_D1Rcontra + f_cortex_R[i] * w_cortexR_D1Rcontra)
        I_alpha_ext              = I_D1Rcontra_extern
        I_D1Rcontra_intern[i]    = (E1 - V_D1R_contra[i]) * (f_D1R_ipsi[i] * w_D1Lipsi_D1Lcontra + f_D2R_contra[i] * w_D2Lcontra_D1Lcontra + f_D2R_ipsi[i] * w_D2Lipsi_D1Lcontra + f_PVR[i] * w_PVL_D1Lcontra)
        I_alpha_int              = I_D1Rcontra_intern
        I_D1Rcontra_self_inhi[i] = (E1 - V_D1R_contra[i]) * f_D1R_contra[i] * w_self_inhi
        V_D1R_contra[i + 1]      = V_D1R_contra[i] + (E - V_D1R_contra[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D1Rcontra_self_inhi[i]) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D1R_contra[i + 1]   < -65:
            V_D1R_contra[i + 1]  = -65
        f_D1R_contra[i + 1]      = transfer_function_g(V_D1R_contra[i + 1])

        # f_D1R_ipsi
        I_D1Ripsi_extern[i]    = (E2 - V_D1R_ipsi[i]) * (f_cortex_L[i] * w_cortexL_D1Ripsi + f_cortex_R[i] * w_cortexR_D1Ripsi)
        I_alpha_ext            = I_D1Ripsi_extern
        I_D1Ripsi_intern[i]    = (E1 - V_D1R_ipsi[i]) * (f_D1R_contra[i] * w_D1Lcontra_D1Lipsi + f_D2R_contra[i] * w_D2Lcontra_D1Lipsi + f_D2R_ipsi[i] * w_D2Lipsi_D1Lipsi + f_PVR[i] * w_PVL_D1Lipsi)
        I_alpha_int            = I_D1Ripsi_intern
        I_D1Ripsi_self_inhi[i] = (E1 - V_D1R_ipsi[i]) * f_D1R_ipsi[i] * w_self_inhi
        V_D1R_ipsi[i + 1]      = V_D1R_ipsi[i] + (E - V_D1R_ipsi[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D1Ripsi_self_inhi[i]) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D1R_ipsi[i + 1]   < -65:
            V_D1R_ipsi[i + 1]  = -65
        f_D1R_ipsi[i + 1]      = transfer_function_g(V_D1R_ipsi[i + 1])

        # f_D2R_contra
        I_D2Rcontra_extern[i]    = (E2 - V_D2R_contra[i]) * (f_cortex_L[i] * w_cortexL_D2Rcontra + f_cortex_R[i] * w_cortexR_D2Rcontra)
        I_alpha_ext              = I_D2Rcontra_extern
        I_D2Rcontra_intern[i]    = (E1 - V_D2R_contra[i]) * (f_D1R_contra[i] * w_D1Lcontra_D2Lcontra + f_D1R_ipsi[i] * w_D1Lipsi_D2Lcontra + f_D2R_ipsi[i] * w_D2Lipsi_D2Lcontra + f_PVR[i] * w_PVL_D2Lcontra)
        I_alpha_int              = I_D2Rcontra_intern
        I_D2Rcontra_self_inhi[i] = (E1 - V_D2R_contra[i]) * f_D2R_contra[i] * w_self_inhi
        V_D2R_contra[i + 1]      = V_D2R_contra[i] + (E - V_D2R_contra[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D2Rcontra_self_inhi[i]) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D2R_contra[i + 1]   < -65:
            V_D2R_contra[i + 1]  = -65
        f_D2R_contra[i + 1]      = transfer_function_g(V_D2R_contra[i + 1])

        # f_D2R_ipsi
        I_D2Ripsi_extern[i]    = (E2 - V_D2R_ipsi[i]) * (f_cortex_L[i] * w_cortexL_D2Ripsi + f_cortex_R[i] * w_cortexR_D2Ripsi)
        I_alpha_ext            = I_D2Ripsi_extern
        I_D2Ripsi_intern[i]    = (E1 - V_D2R_ipsi[i]) * (f_D1R_contra[i] * w_D1Lcontra_D2Lipsi + f_D1R_ipsi[i] * w_D1Lipsi_D2Lipsi + f_D2R_contra[i] * w_D2Lcontra_D2Lipsi + f_PVR[i] * w_PVL_D2Lipsi)
        I_alpha_int            = I_D2Ripsi_intern
        I_D2Ripsi_self_inhi[i] = (E1 - V_D2R_ipsi[i]) * f_D2R_ipsi[i] * w_self_inhi
        V_D2R_ipsi[i + 1]      = V_D2R_ipsi[i] + (E - V_D2R_ipsi[i] + I_alpha_ext[i] + I_alpha_int[i] + I_D2Ripsi_self_inhi[i]) * dt / tau + sigma_bis * sqrtdt * np.random.randn()
        if V_D2R_ipsi[i + 1]   < -65:
            V_D2R_ipsi[i + 1]  = -65
        f_D2R_ipsi[i + 1]      = transfer_function_g(V_D2R_ipsi[i + 1])

    return f_D1L_contra, f_D1L_ipsi, f_D2L_contra, f_D2L_ipsi, f_D1R_contra, f_D1R_ipsi, f_D2R_contra, f_D2R_ipsi


@jit(nopython=True)
def SNr_unit(f_D1L_contra, f_D1L_ipsi, f_D2L_contra, f_D2L_ipsi, f_D1R_contra, f_D1R_ipsi, f_D2R_contra, f_D2R_ipsi):
    # params for simulation
    T   = 3.0
    dt  = 0.001
    n   = int(T / dt)
    tau = 0.1
    E   = -55.
    E1  = -65.
    E2  = 0.
    # weights in SNr
    w_d1_snr             = 0.055
    w_d2_snr             = 0.01
    w_snr_self_inhi      = 0.01
    w_snr_eachother_inhi = 0.01
    # Initialization current, voltage, firing rate as in def neuron_params_init()
    I_SNrL_inhi = np.zeros(n)
    I_SNrL_exci = np.zeros(n)
    I_SNrR_inhi = np.zeros(n)
    I_SNrR_exci = np.zeros(n)
    V_SNrL      = np.ones(n) * E
    V_SNrR      = np.ones(n) * E
    I_from_SNrL_inhi = np.zeros(n)
    I_SNrL_self_inhi = np.zeros(n)
    I_SNrR_self_inhi = np.zeros(n)
    I_from_SNrR_inhi = np.zeros(n)
    f_SNrL = np.ones(n) * transfer_function_g(E)
    f_SNrR = np.ones(n) * transfer_function_g(E)

    # set noise for snr
    sigma     = 0.035
    noise_tau = 0.05
    sigma_bis = sigma * np.sqrt(2. / noise_tau)
    sqrtdt    = np.sqrt(dt)
    noise_snr = sigma_bis * sqrtdt

    for i in range(n - 1):
        # f_SNrL
        I_SNrL_inhi[i]      = (E1 - V_SNrL[i]) * w_d1_snr * (f_D1L_contra[i] + f_D1L_ipsi[i])
        I_from_SNrR_inhi[i] = (E1 - V_SNrL[i]) * f_SNrR[i] * w_snr_eachother_inhi
        I_SNrL_self_inhi[i] = (E1 - V_SNrL[i]) * w_snr_self_inhi * f_SNrL[i]
        I_SNrL_exci[i]      = (E2 - V_SNrL[i]) * w_d2_snr * (f_D2L_contra[i] + f_D2L_ipsi[i])
        V_SNrL[i + 1]       = V_SNrL[i] + (E - V_SNrL[i] + I_SNrL_inhi[i] + I_SNrL_exci[i] + I_from_SNrR_inhi[i] + I_SNrL_self_inhi[i]) * dt / tau + noise_snr * np.random.randn()
        if V_SNrL[i + 1]    < -65:
            V_SNrL[i + 1]   = -65
        f_SNrL[i + 1]       = transfer_function_g(V_SNrL[i + 1])
        # f_SNrR
        I_SNrR_inhi[i]      = (E1 - V_SNrR[i]) * w_d1_snr * (f_D1R_contra[i] + f_D1R_ipsi[i])
        I_from_SNrL_inhi[i] = (E1 - V_SNrR[i]) * f_SNrL[i] * w_snr_eachother_inhi
        I_SNrR_self_inhi[i] = (E1 - V_SNrR[i]) * w_snr_self_inhi * f_SNrR[i]
        I_SNrR_exci[i]      = (E2 - V_SNrR[i]) * w_d2_snr * (f_D2R_contra[i] + f_D2R_ipsi[i])
        V_SNrR[i + 1]       = V_SNrR[i] + (E - V_SNrR[i] + I_SNrR_inhi[i] + I_SNrR_exci[i] + I_from_SNrL_inhi[i] + I_SNrR_self_inhi[i]) * dt / tau + noise_snr * np.random.randn()
        if V_SNrR[i + 1]    < -65:
            V_SNrR[i + 1]   = -65
        f_SNrR[i + 1]       = transfer_function_g(V_SNrR[i + 1])

    return f_SNrL, f_SNrR


@jit(nopython=True)
def decision_prob(freq):
    # ----- Equation 1 & 2 ------
    # params for simulation
    dt         = 0.001
    sigma      = 0.05
    tau        = 0.05
    sigma_bis  = sigma * np.sqrt(2. / tau)
    sqrtdt     = np.sqrt(dt)
    noise_decision = sigma_bis * sqrtdt

    # params for equation
    x          = random.uniform(0, 1)
    freq       = freq - 10
    sigma_prob = 2.24
    prob       = 0.5 * np.exp(-(np.power(freq/sigma_prob, 2))) + noise_decision * np.random.randn()
    if x < prob:
        perceptual_decision = 1.
    else:
        perceptual_decision = -1.
    return perceptual_decision


def loop(stim_value, total_Sessions, Trial_repeat, I_D1L_Inhi, I_D2L_Inhi, I_PVL_Inhi):
    # todo 是否改一点参数让 t_start=100？？？
    t_start = 100
    t_end   = 3000

    freq = [0.0, 2.86, 5.71, 8.57, 11.43, 14.29, 17.14, 20.00]
    contra_choice_tot = np.array([])
    for session in range(total_Sessions):
        contra_choice = np.array([])
        for i in range(8):
            left_choice_sum  = 0
            right_choice_sum = 0
            for trials in range(Trial_repeat):
                perceptual_decision = decision_prob(freq[i])
                f_cortex_L    = rate_ctx(20., perceptual_decision, freq[i])
                f_cortex_R    = rate_ctx(20., -1. * perceptual_decision, freq[i])
                f_cortex_L[0] = 0.
                f_cortex_R[0] = 0.
                f_D1L_contra, f_D1L_ipsi, f_D2L_contra, f_D2L_ipsi, f_D1R_contra, f_D1R_ipsi, f_D2R_contra, f_D2R_ipsi = network(f_cortex_L, f_cortex_R, I_D1L_Inhi, I_D2L_Inhi, I_PVL_Inhi)
                SNrL, SNrR = SNr_unit(f_D1L_contra, f_D1L_ipsi, f_D2L_contra, f_D2L_ipsi, f_D1R_contra, f_D1R_ipsi, f_D2R_contra, f_D2R_ipsi)

                diff       = SNrL[t_start:t_end] - SNrR[t_start:t_end]
                diff_abs   = np.abs(SNrL[t_start:t_end] - SNrR[t_start:t_end])
                index      = np.where(diff_abs > 0.00)
                if diff[index[0][0]] >= 0:
                    left_choice_sum  = left_choice_sum + 1
                else:
                    right_choice_sum = right_choice_sum + 1
            contra_choice = np.append(contra_choice, right_choice_sum)
        contra_choice_tot = np.append(contra_choice_tot, contra_choice, axis=0)

    contra_choice_tot = contra_choice_tot.reshape((total_Sessions, 8))
    locals()['contra_choice_tot_' + str(stim_value)] = contra_choice_tot
    io.savemat('contra_choice_tot_' + str(stim_value) + '.mat', {'contra_choice_tot_' + str(stim_value): eval('contra_choice_tot_' + str(stim_value))})


def main():
    num_split      = 1  # set interval of stimulus
    total_Sessions = 22
    Trial_repeat   = 100
    # change this to simulate different optogenetic perturbation current
    Optogenetic_perturbation_current = np.linspace(-1.5, -3.5, num_split)  # i.e. D1 inactivation -> -2.5±1.0, other params are provided in paper.
    for i in range(num_split):
        # change 0 to 1  make optogenetic perturbation, to reproduce the control results, all set to 0(false)
        D1L_Inhi  = 0
        D2L_Inhi  = 0
        PVL_Inhi  = 0
        Semi_Inhi = 0
        D1L_Exci  = 0
        D2L_Exci  = 0

        if D1L_Inhi:
            I_D1L_Inhi = Optogenetic_perturbation_current[i]
        else:
            I_D1L_Inhi = 0.

        if D2L_Inhi:
            I_D2L_Inhi = Optogenetic_perturbation_current[i]
        else:
            I_D2L_Inhi = 0.

        if PVL_Inhi:
            I_PVL_Inhi = Optogenetic_perturbation_current[i]
        else:
            I_PVL_Inhi = 0.

        if Semi_Inhi:
            I_D1L_Inhi = Optogenetic_perturbation_current[i]
            I_D2L_Inhi = Optogenetic_perturbation_current[i]
            I_PVL_Inhi = Optogenetic_perturbation_current[i]
        elif D1L_Inhi or D2L_Inhi or PVL_Inhi:
            I_D1L_Inhi
            I_D2L_Inhi
            I_PVL_Inhi
        else:
            I_D1L_Inhi = 0.
            I_D2L_Inhi = 0.
            I_PVL_Inhi = 0.

        if D1L_Exci:
            I_D1L_Inhi = -Optogenetic_perturbation_current[i]
        elif D1L_Inhi or Semi_Inhi:
            I_D1L_Inhi
        else:
            I_D1L_Inhi = 0.

        if D2L_Exci:
            I_D2L_Inhi = -Optogenetic_perturbation_current[i]
        elif D2L_Inhi or Semi_Inhi:
            I_D2L_Inhi
        else:
            I_D2L_Inhi = 0.

        stim_value = i + 1
        loop(stim_value, total_Sessions, Trial_repeat, I_D1L_Inhi, I_D2L_Inhi, I_PVL_Inhi)


if __name__ == '__main__':
    # start simulation
    time_start = time.time()
    main()
    time_end = time.time()
    print('Total Simulation Cost {:.2f} seconds.'.format(time_end - time_start))
