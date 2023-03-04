#!/usr/bin/env python
# coding: utf-8

'''
Simulate label collection approaches to measure disparities 
between two groups. Experiments with collecting samples
based on group statistics, equal representation, and uniform.
Datasets include ACS and BRFSS.
'''

import numpy as np
import argparse
from functools import partial
from joblib import Parallel, delayed
from datetime import datetime
import os
import sys
from pathlib import Path
import torch

from data import gendata_single
from data import sigma_uniform_sample
from data import sigma_group_sample
from data import Simulate_data
from data import Dataset

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_context('talk')
sns.set_palette('colorblind')

SMALL_NUMBER = 1e-6
STORAGE_PATH = 'der_run_storage/'

def evaluate_rmse(y_est, y_true):
    return np.sqrt(np.mean((y_est - y_true)**2))

def metrics_uniform(X, P, S, Y):
    '''
    Computes true values of metrics from large enough population.
    '''
    X, P, S, Y = np.array(X), np.array(P), np.array(S), np.array(Y)
    A = X[:,1]  # TODO handle group information at arbitrary index
    W = X[:,2]
    n = S.shape[0]
    P_s, Y_s, A_s, W_s = P[S==1], Y[S==1], A[S==1], W[S==1]
    P_A1_s, P_A0_s = P_s[A_s==1], P_s[A_s!=1]
    Y_A1_s, Y_A0_s = Y_s[A_s==1], Y_s[A_s!=1]
    W_A1_s, W_A0_s = W_s[A_s==1], W_s[A_s!=1]
    # w =  1 / P_s
    # w_A1 = 1 / P_A1_s
    # w_A0 = 1 / P_A0_s
    # Y_wtd = w * Y_s
    # Y_A1_wtd = w_A1 * Y_A1_s
    # Y_A0_wtd = w_A0 * Y_A0_s
    # #     Y_wtd = [s*y/p for (p,s,y) in zip(P,S,Y)]
    # if stable:
    #     est_full = np.sum(Y_wtd) / np.sum(w)
    #     est_A1 = np.sum(Y_A1_wtd) / np.sum(w_A1)
    #     est_A0 = np.sum(Y_A0_wtd) / np.sum(w_A0)
    # else:
    #     est_full = np.sum(Y_wtd) / n # correct division
    #     est_A1 = np.sum(Y_A1_wtd) / n
    #     est_A0 = np.sum(Y_A0_wtd) / n
    est_full = np.average(Y_s, weights=W_s)
    est_A1 = np.average(Y_A1_s, weights=W_A1_s)
    est_A0 = np.average(Y_A0_s, weights=W_A0_s)
    # Difference in means
    diff_est = est_A1 - est_A0
    # Deviation from Equal Representation (DER)
    frac_mean_A1 = est_A1 / (est_A1 + est_A0)
    frac_mean_A0 = 1 - frac_mean_A1
    der = 2 * ((frac_mean_A1 - 1/2)**2 + (frac_mean_A0 - 1/2)**2)
    theils = 2*frac_mean_A0*np.log(2*frac_mean_A0) + 2*frac_mean_A1*np.log(2*frac_mean_A1)
    
    frac_A1 = np.average(A_s==1, weights=W_s)
    return est_full, est_A1, est_A0, diff_est, der, theils, frac_A1

def uniform_estimate(X, P, S, Y):
    X, P, S, Y = np.array(X), np.array(P), np.array(S), np.array(Y)
    A = X[:,1]  # TODO handle group information at arbitrary index
    W = X[:,2]
    n = S.shape[0]
    n_A1 = np.sum(A==1)
    n_A0 = np.sum(A!=1)
    P_s, Y_s, A_s, W_s = P[S==1], Y[S==1], A[S==1], W[S==1]
    Y_A1_s, Y_A0_s = Y_s[A_s==1], Y_s[A_s!=1]
    W_A1_s, W_A0_s = W_s[A_s==1], W_s[A_s!=1]
    est_A1 = np.average(Y_A1_s, weights=W_A1_s)  # TODO: handle empty weights vector
    est_A0 = np.average(Y_A0_s, weights=W_A0_s)
    est_full = np.average(Y_s, weights=W_s)
    frac_A1 = np.average(A_s==1, weights=np.ones_like(W_s))
    return est_full, est_A0, est_A1, frac_A1

def stratified_estimate(X, P, S, Y):
    X, P, S, Y = np.array(X), np.array(P), np.array(S), np.array(Y)
    A = X[:,1]  # TODO handle group information at arbitrary index
    W = X[:,2]
    n = S.shape[0]
    n_A1 = np.sum(A==1)
    n_A0 = np.sum(A!=1)
    P_s, Y_s, A_s, W_s = P[S==1], Y[S==1], A[S==1], W[S==1]
    Y_A1_s, Y_A0_s = Y_s[A_s==1], Y_s[A_s!=1]
    W_A1_s, W_A0_s = W_s[A_s==1], W_s[A_s!=1]
    est_A1 = np.average(Y_A1_s, weights=W_A1_s)  # TODO: handle empty weights vector
    est_A0 = np.average(Y_A0_s, weights=W_A0_s)
    est_full = n_A1/n * est_A1 + n_A0/n * est_A0
    frac_A1 = np.average(A_s==1, weights=np.ones_like(W_s))
    return est_full, est_A0, est_A1, frac_A1

def metrics_stratified(X, P, S, Y, X_pilot, P_pilot, S_pilot, Y_pilot, sampling_type='stratified'):
    # Main
    if sampling_type=='stratified':
        est_full_main, est_A0_main, est_A1_main, frac_A1_main = stratified_estimate(X, P, S, Y)
    elif sampling_type=='uniform':
        est_full_main, est_A0_main, est_A1_main, frac_A1_main = uniform_estimate(X, P, S, Y)
    else:
        raise NotImplementedError(f'Sampling type {sampling_type} unknown')
    if False:
        # Difference in means
        diff_est_main = est_A1_main - est_A0_main
        # Deviation from Equal Representation (DER)
        frac_mean_A1_main = est_A1_main / (est_A1_main + est_A0_main)
        frac_mean_A0_main = 1 - frac_mean_A1_main
        der_main = 2 * ((frac_mean_A1_main - 1/2)**2 + (frac_mean_A0_main - 1/2)**2)

    # Pilot
    if sampling_type=='stratified':
        est_full_pilot, est_A0_pilot, est_A1_pilot, _ = stratified_estimate(X_pilot, P_pilot, S_pilot, Y_pilot)
    elif sampling_type=='uniform':
        est_full_pilot, est_A0_pilot, est_A1_pilot, _ = uniform_estimate(X_pilot, P_pilot, S_pilot, Y_pilot)
    else:
        raise NotImplementedError(f'Method {sampling_type} unknown')
    
    number_pilot = S_pilot.sum()
    number_main = S.sum()
    weight = number_pilot/(number_pilot+number_main)
    est_full = weight*est_full_pilot + (1-weight)*est_full_main
    est_A0 = weight*est_A0_pilot + (1-weight)*est_A0_main
    est_A1 = weight*est_A1_pilot + (1-weight)*est_A1_main
    # Difference in means
    diff_est = est_A1 - est_A0
    # Deviation from Equal Representation (DER)
    frac_mean_A1 = est_A1 / (est_A1 + est_A0)
    frac_mean_A0 = 1 - frac_mean_A1
    der = 2 * ((frac_mean_A1 - 1/2)**2 + (frac_mean_A0 - 1/2)**2)
    theils = 2*frac_mean_A0*np.log(2*frac_mean_A0) + 2*frac_mean_A1*np.log(2*frac_mean_A1)

    return est_full, est_A1, est_A0, diff_est, der, theils, frac_A1_main

def metrics_inverse_propensity_weighted(X, P, S, Y):
    X, P, S, Y = np.array(X), np.array(P), np.array(S), np.array(Y)
    A = X[:,1]  # TODO handle group information at arbitrary index
    W = X[:,2]
    n = S.shape[0]
    n_A1 = np.sum(A==1)
    n_A0 = np.sum(A!=1)
    P_s, Y_s, A_s, W_s = P[S==1], Y[S==1], A[S==1], W[S==1]
    P_A1_s, P_A0_s = P_s[A_s==1], P_s[A_s!=1]
    Y_A1_s, Y_A0_s = Y_s[A_s==1], Y_s[A_s!=1]
    W_A1_s, W_A0_s = W_s[A_s==1], W_s[A_s!=1]

    w =  W_s / P_s
    w_A1 = W_A1_s / P_A1_s
    w_A0 = W_A0_s / P_A0_s
    
    Y_wtd = w * Y_s
    Y_A1_wtd = w_A1 * Y_A1_s
    Y_A0_wtd = w_A0 * Y_A0_s
   
    est_full = np.sum(Y_wtd) / np.sum(W)
    est_A1 = np.sum(Y_A1_wtd) / np.sum(W[A==1])
    est_A0 = np.sum(Y_A0_wtd) / np.sum(W[A==0])

    # Difference in means
    diff_est = est_A1 - est_A0
    # Deviation from Equal Representation (DER)
    frac_mean_A1 = est_A1 / (est_A1 + est_A0)
    frac_mean_A0 = 1 - frac_mean_A1
    der = 2 * ((frac_mean_A1 - 1/2)**2 + (frac_mean_A0 - 1/2)**2)
    ratio = est_A0/est_A1
    theils = 2*frac_mean_A0*np.log(2*frac_mean_A0) + 2*frac_mean_A1*np.log(2*frac_mean_A1)
    
    frac_A1 = np.average(A_s==1, weights=np.ones_like(W_s))
    return est_full, est_A1, est_A0, diff_est, der, theils, frac_A1

SMALL_NUMBER = 1e-6

def compute_mean_actual(X, Y):
    P = np.ones((X.shape[0],))
    S = P
    Y_mean, Y_A1_mean, Y_A0_mean, diff_mean, der, theils, frac_A1 = metrics_uniform(X, P, S, Y)
    
    if False:
        Y_mean_std_boot = np.std([np.mean(np.random.choice(Y, size=10000)) for _ in range(100)])
        print(f"Y_mean_std_boot {Y_mean_std_boot}")

    return Y_mean, Y_A1_mean, Y_A0_mean, diff_mean, der, theils, frac_A1

def get_sample_uniform(rng, args, nsample, X_full, Y_full):
    npop = X_full.shape[0]
    rate = nsample / npop
    sigma_uniform = sigma_uniform_sample(rng, rate)
    train_data_generator = Simulate_data(X_full, Y_full)
    data = gendata_single(npop, sigma_uniform, train_data_generator)
    X, P, S, Y = zip(*data)
    X, P, S, Y = np.array(X), np.array(P), np.array(S), np.array(Y)
    if len(X.shape)==1:
        X = X[:,np.newaxis]
    print("Uniform sampled. Achieved rate={}, desired rate={}".format(np.mean(S), rate))
    assert X.shape[0]==npop, "Population incorrectly sampled in Uniform"
    return X, P, S, Y

def get_sample_group_equal(rng, args, X_full, Y_full, X_pilot, P_pilot, S_pilot, Y_pilot):
    n_full = X_full.shape[0]  # population size
    n_full_A1 = np.sum(X_full[:,1]==1)  # TODO handle group information at arbitrary index
    n_full_A0 = n_full - n_full_A1
    desired_each_group = int(args.nsample * 0.5)
    if (n_full_A0 >= desired_each_group) and (n_full_A1 >= desired_each_group):
        rate_A0 = desired_each_group / n_full_A0
        rate_A1 = desired_each_group / n_full_A1
    elif n_full_A0 < desired_each_group:
        rate_A0 = 1.0
        rate_A1 = (args.nsample - n_full_A0) / n_full_A1
    elif n_full_A1 < desired_each_group:
        rate_A0 = (args.nsample - n_full_A1) / n_full_A0
        rate_A1 = 1.0
    else:
      raise RuntimeError("Incorrect logic in Equal")
    sigma_equal = lambda x: sigma_group_sample(rng, x, rate_A0, rate_A1)
    train_data_generator = Simulate_data(X_full, Y_full)
    data = gendata_single(n_full, sigma_equal, train_data_generator)
    X, P, S, Y = zip(*data)
    X, P, S, Y = np.array(X), np.array(P), np.array(S), np.array(Y)
    if len(X.shape)==1:
        X = X[:,np.newaxis]
    print(f"Equal sampled. Achieved rate A0,A1={(np.mean(S[X[:,1]!=1]),np.mean(S[X[:,1]==1]))}, desired rate A0,A1={(rate_A0,rate_A1)}")
    return X, P, S, Y

def get_weighted_stdev(args, Y, weights):
    '''
    Formula for stdev of weighted Mean in page 19 of
    https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2014_2018AccuracyPUMS.pdf
    '''
    if not args.use_weights:
        return np.std(Y)
    else:
        numerator = np.sum(weights * Y**2) - (np.sum(weights*Y))**2/np.sum(weights)
        denominator = np.sum(weights) - 1
        return np.sqrt(numerator/denominator)
    
def get_sample_stdev_group(rng, args, X_full, Y_full, X_pilot, P_pilot, S_pilot, Y_pilot, target_metric):
    """
    Optimal allocation for overall, difference, der, ratio metrics
    Stdev, means estimated from the full data
    """
    n_full = X_full.shape[0]  # population size
    n_full_A1 = np.sum(X_full[:,1]==1)  # TODO handle group information at arbitrary index
    n_full_A0 = n_full - n_full_A1

    # Compute stdev by looking only at the SAMPLED outcomes
    X_pilot_sampled, Y_pilot_sampled = X_pilot[S_pilot==1], Y_pilot[S_pilot==1]
    W_pilot_sampled = X_pilot_sampled[:,2]
    stdev_A0 = get_weighted_stdev(args, Y_pilot_sampled[X_pilot_sampled[:,1]!=1], weights=W_pilot_sampled[X_pilot_sampled[:,1]!=1])
    stdev_A1 = get_weighted_stdev(args, Y_pilot_sampled[X_pilot_sampled[:,1]==1], weights=W_pilot_sampled[X_pilot_sampled[:,1]==1])
    mean_A0 = np.average(Y_pilot_sampled[X_pilot_sampled[:,1]!=1], weights=W_pilot_sampled[X_pilot_sampled[:,1]!=1])
    mean_A1 = np.average(Y_pilot_sampled[X_pilot_sampled[:,1]==1], weights=W_pilot_sampled[X_pilot_sampled[:,1]==1])
    
    # Compute stdev by looking at both the SAMPLED and UNSAMPLED outcomes
    W_full = X_full[:,2]
    true_stdev_A0 = get_weighted_stdev(args, Y_full[X_full[:,1]!=1], weights=W_full[X_full[:,1]!=1])
    true_stdev_A1 = get_weighted_stdev(args, Y_full[X_full[:,1]==1], weights=W_full[X_full[:,1]==1])
    true_mean_A0 = np.average(Y_full[X_full[:,1]!=1], weights=W_full[X_full[:,1]!=1])
    true_mean_A1 = np.average(Y_full[X_full[:,1]==1], weights=W_full[X_full[:,1]==1])

    print(f"\n\nmean std A0 full {true_mean_A0,true_stdev_A0}, A1 {true_mean_A1,true_stdev_A1}")
    print(f"mean std A0 pilot {mean_A0,stdev_A0}, A1 {mean_A1,stdev_A1}\n\n")
    if (mean_A0==0) or (mean_A1==0) or (stdev_A0==0) or (stdev_A1==0):
        print(f'\n****Zero mean or stdev in pilot. {args.dataset_name, args.us_state, args.outcome, args.group}****\n')
    if target_metric in ['stdev_overall', 'stdev_est_denom']:
      sample_ratio = stdev_A0*n_full_A0/(stdev_A0*n_full_A0+stdev_A1*n_full_A1+SMALL_NUMBER)
    elif target_metric == 'stdev_diff':
      sample_ratio = stdev_A0/(stdev_A0+stdev_A1+SMALL_NUMBER)
    elif target_metric in ['stdev_der', 'stdev_ratio', 'stdev_theils']:
      sample_ratio = np.abs(mean_A1)*stdev_A0/(np.abs(mean_A1)*stdev_A0+np.abs(mean_A0)*stdev_A1+SMALL_NUMBER)
    elif target_metric == 'oracle_stdev_diff':
      sample_ratio = true_stdev_A0/(true_stdev_A0+true_stdev_A1+SMALL_NUMBER)
    elif target_metric == 'oracle_stdev_der':
      sample_ratio = np.abs(true_mean_A1)*true_stdev_A0/(np.abs(true_mean_A1)*true_stdev_A0+np.abs(true_mean_A0)*true_stdev_A1+SMALL_NUMBER)
    else:
      raise NotImplementedError(f'Stdev method for {target_metric}')
    desired_A0 = int(sample_ratio * args.nsample)
    desired_A1 = args.nsample - desired_A0
    desired_A0 = max(desired_A0, 1)  # sample 1 response at the least
    desired_A1 = max(desired_A1, 1)
    if (n_full_A0 >= desired_A0) and (n_full_A1 >= desired_A1):
        rate_A0 = desired_A0 / n_full_A0
        rate_A1 = desired_A1 / n_full_A1
    elif n_full_A0 < desired_A0:
        rate_A0 = 1.0
        rate_A1 = (args.nsample - n_full_A0) / n_full_A1
    elif n_full_A1 < desired_A1:
        rate_A0 = (args.nsample - n_full_A1) / n_full_A0
        rate_A1 = 1.0
    else:
      raise RuntimeError("Incorrect logic in Stdev")
    sigma_stdev = lambda x: sigma_group_sample(rng, x, rate_A0, rate_A1)
    train_data_generator = Simulate_data(X_full, Y_full)
    data = gendata_single(n_full, sigma_stdev, train_data_generator)
    X, P, S, Y = zip(*data)
    X, P, S, Y = np.array(X), np.array(P), np.array(S), np.array(Y)
    if len(X.shape)==1:
        X = X[:,np.newaxis]
    print(f"{target_metric} sampled. Stdev A0,A1={(stdev_A0,stdev_A1)}, group sizes population={(n_full_A0,n_full_A1)}, mean A0,A1={(mean_A0,mean_A1)}, fraction={sample_ratio}, number of samples A0,A1={S[X[:,1]!=1].sum(),S[X[:,1]==1].sum()}, achieved rate A0,A1={(np.mean(S[X[:,1]!=1]),np.mean(S[X[:,1]==1]))}, desired rate A0,A1={(rate_A0,rate_A1)}")
    return X, P, S, Y

def run_exp(rng, args, X_full, Y_full, expid):
    print('\nRun:{}'.format(expid+1))

    # Sample pilot data
    X_pilot, P_pilot, S_pilot, Y_pilot = get_sample_uniform(rng, args, args.nsample_pilot, X_full, Y_full)
    print("pilot", S_pilot.sum(), args.nsample_pilot, X_pilot.shape, Y_pilot.shape)

    if args.sampling_method == 'uniform':
        X_sampled, P_sampled, S_sampled, Y_sampled = get_sample_uniform(rng, args, args.nsample, X_full, Y_full)
        Y_mean, Y_A1_mean, Y_A0_mean, diff_mean, der, theils, frac_A1 = metrics_stratified(X_sampled, P_sampled, S_sampled, Y_sampled,
                                                                                X_pilot, P_pilot, S_pilot, Y_pilot, sampling_type='uniform')
    elif args.sampling_method == 'equal':
        X_sampled, P_sampled, S_sampled, Y_sampled = get_sample_group_equal(rng, args, X_full, Y_full,
                                                                            X_pilot, P_pilot, S_pilot, Y_pilot)
        Y_mean, Y_A1_mean, Y_A0_mean, diff_mean, der, theils, frac_A1 = metrics_stratified(X_sampled, P_sampled, S_sampled, Y_sampled,
                                                                                X_pilot, P_pilot, S_pilot, Y_pilot, sampling_type='stratified')
    elif args.sampling_method in ['stdev_overall', 'stdev_diff', 'stdev_der', 'stdev_theils', 'oracle_stdev_diff', 'oracle_stdev_der']:
        X_sampled, P_sampled, S_sampled, Y_sampled = get_sample_stdev_group(rng, args, X_full, Y_full,
                                                                            X_pilot, P_pilot, S_pilot, Y_pilot,
                                                                            target_metric=args.sampling_method)
        Y_mean, Y_A1_mean, Y_A0_mean, diff_mean, der, theils, frac_A1 = metrics_stratified(X_sampled, P_sampled, S_sampled, Y_sampled,
                                                                                   X_pilot, P_pilot, S_pilot, Y_pilot, sampling_type='stratified')
    elif args.sampling_method == 'uniform_est_denom':
        X_sampled, P_sampled, S_sampled, Y_sampled = get_sample_uniform(rng, args, args.nsample, X_full, Y_full)
        Y_mean, Y_A1_mean, Y_A0_mean, diff_mean, der, theils, frac_A1 = metrics_inverse_propensity_weighted(X_sampled, P_sampled, S_sampled, Y_sampled)
    elif args.sampling_method == 'equal_est_denom':
        X_sampled, P_sampled, S_sampled, Y_sampled = get_sample_group_equal(rng, args, X_full, Y_full,
                                                                            X_pilot, P_pilot, S_pilot, Y_pilot)
        Y_mean, Y_A1_mean, Y_A0_mean, diff_mean, der, theils, frac_A1 = metrics_inverse_propensity_weighted(X_sampled, P_sampled, S_sampled, Y_sampled)
    elif args.sampling_method == 'stdev_est_denom':
        X_sampled, P_sampled, S_sampled, Y_sampled = get_sample_stdev_group(rng, args, X_full, Y_full, 
                                                                            X_pilot, P_pilot, S_pilot, Y_pilot,
                                                                            target_metric=args.sampling_method)
        Y_mean, Y_A1_mean, Y_A0_mean, diff_mean, der, theils, frac_A1 = metrics_inverse_propensity_weighted(X_sampled, P_sampled, S_sampled, Y_sampled)
    else:
      raise NotImplementedError(args.sampling_method)
    
    result = {'Y_mean':Y_mean, 'Y_A1_mean':Y_A1_mean, 'Y_A0_mean':Y_A0_mean,
              'diff_mean':diff_mean, 'der':der, 'theils': theils,
              'frac_A1':frac_A1, 'nsample_pilot':np.sum(S_pilot==1), 'nsample':np.sum(S_sampled==1), 'npop':args.npop}

    return result

def variance_metrics(samples_A0, samples_A1, mean_A0, mean_A1, stdev_A0, stdev_A1):
    nsample = samples_A0 + samples_A1
    var_A0 = stdev_A0**2
    var_A1 = stdev_A1**2
    overall = 1 # samples_A0**2/nsample**2*var_A0 + samples_A1**2/nsample**2*var_A1
    diff =var_A0/samples_A0 + var_A1/samples_A1
    der = 16 * (mean_A0 - mean_A1)**2 / (mean_A0 + mean_A1)**6 * (mean_A1**2*var_A0/samples_A0 + mean_A0**2*var_A1/samples_A1)
    ratio = var_A0/(mean_A1**2*samples_A0) + mean_A0**2*var_A1/(mean_A1**4*samples_A1)
    var_dict = {
        'overall': overall,
        'diff': diff,
        'der': der,
        'ratio': ratio,
    }
    return var_dict

def get_sample_size_for_method(args, X_full, Y_full):
    '''
    Returns desired sample sizes per group for a method and dataset
    '''
    n_full = X_full.shape[0]  # population size
    n_full_A1 = np.sum(X_full[:,1]==1)  # TODO handle group information at arbitrary index
    n_full_A0 = n_full - n_full_A1
    W_full = X_full[:,2]
    stdev_A0 = get_weighted_stdev(args, Y_full[X_full[:,1]!=1], weights=W_full[X_full[:,1]!=1])
    stdev_A1 = get_weighted_stdev(args, Y_full[X_full[:,1]==1], weights=W_full[X_full[:,1]==1])
    mean_A0 = np.average(Y_full[X_full[:,1]!=1], weights=W_full[X_full[:,1]!=1])
    mean_A1 = np.average(Y_full[X_full[:,1]==1], weights=W_full[X_full[:,1]==1])
    print(f"mean stdev: A0 {mean_A0, stdev_A0}, A1 {mean_A1, stdev_A1}")
    if args.sampling_method == 'uniform':
        rate_A0 = args.nsample / n_full
        rate_A1 = rate_A0
        desired_A0 = int(rate_A0*n_full_A0)
        desired_A1 = int(rate_A1*n_full_A1)
    elif args.sampling_method == 'equal':
        desired_each_group = int(args.nsample * 0.5)
        desired_A0 = desired_each_group
        desired_A1 = desired_each_group
    elif args.sampling_method == 'stdev_overall':
        sample_ratio = stdev_A0*n_full_A0/(stdev_A0*n_full_A0+stdev_A1*n_full_A1+SMALL_NUMBER)
        desired_A0 = int(sample_ratio * args.nsample)
        desired_A1 = args.nsample - desired_A0
        print(f"{args.sampling_method} sample ratio {sample_ratio}, A0 gets {desired_A0} out of {args.nsample}")
    elif args.sampling_method == 'stdev_diff':
        sample_ratio = stdev_A0/(stdev_A0+stdev_A1+SMALL_NUMBER)
        desired_A0 = int(sample_ratio * args.nsample)
        desired_A1 = args.nsample - desired_A0
        print(f"{args.sampling_method} sample ratio {sample_ratio}, A0 gets {desired_A0} out of {args.nsample}")
    elif args.sampling_method in ['stdev_der', 'stdev_ratio']:
        sample_ratio = np.abs(mean_A1)*stdev_A0/(np.abs(mean_A1)*stdev_A0+np.abs(mean_A0)*stdev_A1+SMALL_NUMBER)
        desired_A0 = int(sample_ratio * args.nsample)
        desired_A1 = args.nsample - desired_A0
        print(f"{args.sampling_method} sample ratio {sample_ratio}, A0 gets {desired_A0} out of {args.nsample}")
    else:
      raise NotImplementedError(f'Method for {args.sampling_method}')
    group_stats = {
        'mean_A0': mean_A0, 'stdev_A0': stdev_A0,
        'mean_A1': mean_A1, 'stdev_A1': stdev_A1,
        'n_full_A0': n_full_A0, 'n_full_A1': n_full_A1,
    }
    var_dict = variance_metrics(desired_A0, desired_A1, mean_A0, mean_A1, stdev_A0, stdev_A1)
    return desired_A0, desired_A1, group_stats, var_dict

def create_outdir(args):
    now = datetime.now()
    dt = now.strftime("%Y-%m-%d %Hh%Mm%Ss%fms")
    args.dir = os.path.join(os.path.join(STORAGE_PATH, args.results_outdir), dt)
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    torch.save(args, os.path.join(args.dir, 'args.pt'))
    return args

def run(rng, args, X_full, Y_full, metrics_true):
    results = []
    Y_mean_true, Y_A1_mean_true, Y_A0_mean_true, diff_mean_true, der_true, theils_true, frac_A1_true = metrics_true

    for expid in range(args.repeat):
        results.append(run_exp(rng, args, X_full, Y_full, expid))  # dataset is different for each sampling method

    save_runs = {}
    Y_runs = -999*np.ones((args.repeat,9))

    for expid, result in enumerate(results):

        Y_runs[expid,0] = result['Y_mean']
        Y_runs[expid,1] = result['Y_A1_mean']
        Y_runs[expid,2] = result['Y_A0_mean']
        Y_runs[expid,3] = result['diff_mean']
        Y_runs[expid,4] = result['der']
        Y_runs[expid,5] = result['theils']
        Y_runs[expid,6] = result['frac_A1']
        Y_runs[expid,7] = result['nsample']
        Y_runs[expid,8] = result['nsample_pilot']
        save_runs[expid] = result
      
    mse_full = evaluate_rmse(Y_runs[:,0], Y_mean_true)
    mse_A1 = evaluate_rmse(Y_runs[:,1], Y_A1_mean_true)
    mse_A0 = evaluate_rmse(Y_runs[:,2], Y_A0_mean_true)
    mse_diff = evaluate_rmse(Y_runs[:,3], diff_mean_true)
    mse_der = evaluate_rmse(Y_runs[:,4], der_true)
    mse_theils = evaluate_rmse(Y_runs[:,5], theils_true)

    metrics = {}
    metrics['ErrorOverall'] = float(mse_full)
    metrics['ErrorMajGroup'] = float(mse_A1)
    metrics['ErrorMinGroup'] = float(mse_A0)
    metrics['ErrorDifference'] = float(mse_diff)
    metrics['ErrorDER'] = float(mse_der)
    metrics['ErrorTheils'] = float(mse_theils)
    metrics['FractMajGroupPopulation'] = np.round(frac_A1_true, 2)
    metrics['NumberSamplesMainDesired'] = args.nsample
    metrics['NumberSamplesPilotDesired'] = args.nsample_pilot
    metrics['SamplingMethod'] = args.sampling_method
    metrics['Dataset'] = args.dataset_name
    metrics['State'] = args.us_state
    metrics['Outcome'] = args.outcome
    metrics['Group'] = args.group

    df = pd.DataFrame(Y_runs, columns=['Overall','MajGroup','MinGroup','Difference','DER','Theils','FractMajGroupSample','NumberSamplesMain','NumberSamplesPilot'])
    df['Run'] = np.arange(args.repeat)+1
    df['FractMajGroupPopulation'] = np.round(frac_A1_true, 2)
    df['NumberSamplesMainDesired'] = args.nsample
    df['NumberSamplesPilotDesired'] = args.nsample_pilot
    df['SamplingMethod'] = args.sampling_method
    df['Dataset'] = args.dataset_name
    df['State'] = args.us_state
    df['Outcome'] = args.outcome
    df['Group'] = args.group
    df['TrueDifference'] = diff_mean_true
    df['TrueDER'] = der_true
    df['TrueOverall'] = Y_mean_true
    df['TrueMajGroup'] = Y_A1_mean_true
    df['TrueMinGroup'] = Y_A0_mean_true
    df['TrueTheils'] = theils_true

    # Save
    torch.save(save_runs, os.path.join(args.dir, f'save_runs_{args.dataset_id}id.pt'))
    torch.save(metrics, os.path.join(args.dir, f'metrics_{args.dataset_id}id.pt'))
    torch.save(df, os.path.join(args.dir, f'df_{args.dataset_id}id.pt'))
    return df, metrics

def plot_sample(X, P, S, Y):
    Ys = Y[S==1]
    As = X[:,1][S==1]
    plt.hist(Ys[As==1],alpha=0.5,label='A1')
    plt.hist(Ys[As!=1],alpha=0.5,label='A0')
    plt.legend()
    print(Ys[As!=1].mean(), Ys[As==1].mean())

def plot_hist_markers(Y_list, true_Y_mean):
    plt.figure()
    plt.hist(Y_list,alpha=0.5,label='samples')
    plt.vlines(x=np.mean(Y_list), label='sample mean', ymin=0, ymax=10, linestyle='--')
    plt.vlines(x=true_Y_mean, label='true mean', ymin=0, ymax=10, linestyle='-')
    plt.legend()

def main(args, datasets, sampling_opts, use_weights):
    dfs = []
    errors = []

    for (dataset_name, us_state, outcome, group, nsample_pilot, nsample, sim_fract) in datasets:
        # Get population data
        args.nsample_pilot = nsample_pilot
        args.nsample = nsample
        args.sim_fract = sim_fract
        args.dataset_name = dataset_name
        args.us_state = us_state
        args.outcome = outcome
        args.group = group
        args.use_weights = use_weights

        rng = np.random.default_rng(seed=1)
        X_full, Y_full = Dataset(rng, args).population_data()
        metrics_true = compute_mean_actual(X_full, Y_full)

        for sampling_method in sampling_opts:
            args.sampling_method = sampling_method
            args = create_outdir(args)
            print(f"\n======Running {sampling_method, dataset_name, us_state, outcome, group, nsample_pilot, nsample, sim_fract}=========\n")
            assert args.nsample <= X_full.shape[0], "Sample size should be less than population size"
            
            df, metrics = run(rng, args, X_full, Y_full, metrics_true)
            
            errors.append(metrics)
            print(f"\nDesired nsample: {metrics['NumberSamplesMainDesired']}, sim_fract: {sim_fract}, frac_A1_true {metrics['FractMajGroupPopulation']}")
            dfs.append(df)
            df_concat = pd.concat(dfs, ignore_index=True)
            df_concat.to_csv(os.path.join(os.path.join(STORAGE_PATH, args.results_outdir), f'results_mean_{args.dataset_id}id.csv')) 
            pd.DataFrame(errors).to_csv(os.path.join(os.path.join(STORAGE_PATH, args.results_outdir), f'results_error_{args.dataset_id}id.csv'))

    df_concat = pd.concat(dfs, ignore_index=True)
    df_concat.to_csv(os.path.join(os.path.join(STORAGE_PATH, args.results_outdir), f'results_mean_{args.dataset_id}id.csv'))
    pd.DataFrame(errors).to_csv(os.path.join(os.path.join(STORAGE_PATH, args.results_outdir), f'results_error_{args.dataset_id}id.csv'))
    return dfs, errors

def plot_means(args, dfs, datasets):
    df_concat = pd.concat(dfs, ignore_index=True)
    metrics_to_plot = ['Overall','MajGroup','MinGroup','Difference','DER','Theils']

    for (dataset_name, us_state, outcome, group, nsample_pilot, nsample, sim_fract) in datasets:
        for metric in metrics_to_plot:
            plt.figure()
            if us_state is not None:
                df_ = df_concat[
                        (df_concat['NumberSamplesMainDesired']==nsample) &
                        (df_concat['NumberSamplesPilotDesired']==nsample_pilot) &
                        (df_concat["State"]==us_state) &
                        (df_concat["Outcome"]==outcome) &
                        (df_concat["Group"]==group)
                ]
            else:
                df_ = df_concat[
                    (df_concat['NumberSamplesMainDesired']==nsample) &
                    (df_concat['NumberSamplesPilotDesired']==nsample_pilot) &
                    (df_concat["Outcome"]==outcome) &
                    (df_concat["Group"]==group)
                ]
            sns.catplot(data=df_, y=metric, x='FractMajGroupPopulation', hue='SamplingMethod', kind='box')
            metric_true_value_map = {'Overall':'TrueOverall',
                                    'MajGroup':'TrueMajGroup',
                                    'MinGroup':'TrueMinGroup',
                                    'Difference':'TrueDifference',
                                    'DER':'TrueDER',
                                    'Theils':'TrueTheils'}
            metric_name = metric_true_value_map[metric]
            true_value = np.unique(df_[metric_name])[0]
            plt.axhline(y=true_value, linestyle='--')
            plt.title(f'{metric}\nn={nsample}\nPilotSamplesDesired={args.nsample_pilot}')
            plt.savefig(os.path.join(os.path.join(STORAGE_PATH, args.results_outdir), f'{metric}_{args.dataset_id}id_n{nsample_pilot}_{nsample}_{dataset_name}_{us_state}_{outcome}_{group}.png'), bbox_inches='tight')

def plot_errors(args, errors, datasets):
    errors = pd.DataFrame(errors)
    metrics_to_plot = ['ErrorOverall','ErrorMajGroup','ErrorMinGroup','ErrorDifference','ErrorDER','ErrorTheils']

    for (dataset_name, us_state, outcome, group, nsample_pilot, nsample, sim_fract) in datasets:
        for metric in metrics_to_plot:
            plt.figure()
            if us_state is not None:
                errors_ = errors[
                            (errors['NumberSamplesMainDesired']==nsample) &
                            (errors['NumberSamplesPilotDesired']==nsample_pilot) &
                            (errors['State']==us_state) &
                            (errors['Outcome']==outcome) &
                            (errors['Group']==group)
                        ]
            else:
                errors_ = errors[
                        (errors['NumberSamplesMainDesired']==nsample) &
                        (errors['NumberSamplesPilotDesired']==nsample_pilot) &
                        (errors['Outcome']==outcome) &
                        (errors['Group']==group)
                    ]
            sns.catplot(data=errors_, y=metric, x='FractMajGroupPopulation', hue='SamplingMethod',
                        col='Dataset', kind='bar')
            plt.title(f'{metric}\n n={nsample}')
            plt.savefig(os.path.join(os.path.join(STORAGE_PATH, args.results_outdir), f'{metric}_{args.dataset_id}id_n{nsample_pilot}_{nsample}_{dataset_name}_{us_state}_{outcome}_{group}.png'), bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_outdir', default='output_synthetic_metrics/')
    parser.add_argument('--in_filename', default='')
    parser.add_argument('--dataset_id', default=-1, type=int)
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--npop', default=10000, type=int)
    parser.add_argument('--sampling_method', default='uniform', type=str)  # uniform, equal, stdev
    parser.add_argument('--sim_noise', default=1.0, type=float)
    parser.add_argument('--sim_fract', default=0.55, type=float)
    parser.add_argument('--nprocs', default=4, type=int)
    parser.add_argument('--true_mean_samples', default=1000000, type=int)
    parser.add_argument('--dir', default='')

    args = parser.parse_args()
    print(args)

    args = create_outdir(args)

    sampling_opts = [
        'uniform', 'equal',
        'stdev_overall',
        'stdev_diff',
        'oracle_stdev_diff',
        'stdev_der',
        'oracle_stdev_der',
    ]
    # ACS model
    # samples in (pilot, main)
    # nsample_opts = [
    #     (500, 2000),
    #     (500, 500),
    #     (500, 1000),
    #     (500, 1500),
    #     (250, 1000),
    # ]
    # fractmaj_opts = [None]
    # use_weights = False
    # OUTCOMES = [
    #     'income_binary',
    #     'income_real',
    #     'travel_binary',
    #     'travel_real',
    #     # # 'coverage_binary',
    # ]
    # STATES = [
    #     'NY',
    #     # 'TX',
    #     'CA',
    # ]
    # DATASET_OPTS = [
    #     ('acs_model', state, outcome, 'BlackorAA', nsample_pilot, nsample_main, sim_fract)\
    #         for state in STATES for outcome in OUTCOMES for (nsample_pilot, nsample_main) in nsample_opts for sim_fract in fractmaj_opts
    # ]

    # # ACS
    # # samples in (pilot, main)
    # nsample_opts = [
    #     (200, 200),
    #     (200, 500),
    #     (200, 1000),
    #     # (200, 1500),
    #     (500, 1000),
    #     (500, 1500),
    #     (500, 2000),
    #     (500, 2500),
    # ]
    # fractmaj_opts = [None]
    # use_weights = True
    # OUTCOMES = [
    #     'PINCP',
    #     'WAGP',
    #     'PERNP',
    # ]
    # STATES = [
    #     'NY',
    #     'NV', 'AR', 'CT', 'IL',
    #     'TN', 'VA',
    #     # 'CO',
    # ]
    # DATASET_OPTS = [
    #     ('acs', state, outcome, 'BlackorAA', nsample_pilot, nsample_main, sim_fract)\
    #         for state in STATES for outcome in OUTCOMES for (nsample_pilot, nsample_main) in nsample_opts for sim_fract in fractmaj_opts
    # ]

    # BRFSS
    # # samples in (pilot, main)
    # nsample_opts = [
    #     (500, 2000),
    #     (500, 500),
    #     (500, 1000),
    #     (500, 1500),
    #     (500, 2500),
    # ]
    # fractmaj_opts = [None]
    # use_weights = True
    # OUTCOMES = [
    #     'diabindicator',
    #     'sleeptime',
    #     'diabage',
    # ]
    # STATES = [
    #     'all',
    # ]
    # GROUPS = [
    #     'BlackorAA',
    #     # 'Hispanic',
    #     # 'Asian',
    # ]
    # DATASET_OPTS = [
    #     ('brfss', state, outcome, group, nsample_pilot, nsample_main, sim_fract)\
    #         for state in STATES for outcome in OUTCOMES for group in GROUPS for (nsample_pilot, nsample_main) in nsample_opts for sim_fract in fractmaj_opts
    # ]
    # DATASET_OPTS = [x for x in DATASET_OPTS if not ((x[2]=='diabindicator') and (x[3]=='Asian')) and not ((x[2]=='diabage') and (x[3]=='Asian') and (x[4]==250))]

    # Synthetic
    # samples in (pilot, main)
    nsample_opts = [
        (100, 100),
        # (100, 500),
        (100, 1000),
        # (100, 1500),
        (200, 1000),
    ]
    fractmaj_opts = [
        0.5, 0.9,
    ]
    use_weights = False
    OUTCOMES = ['outcome']
    STATES = ['state']
    DATASET_OPTS = [
        ('synthetic', state, outcome, 'Group1', nsample_pilot, nsample_main, sim_fract)\
            for state in STATES for outcome in OUTCOMES for (nsample_pilot, nsample_main) in nsample_opts for sim_fract in fractmaj_opts
    ]

    print(len(DATASET_OPTS), DATASET_OPTS)

    if args.dataset_id==-1:
        datasets = DATASET_OPTS
    else:
        datasets = DATASET_OPTS[args.dataset_id-1 : args.dataset_id]

    means, errors = main(args, datasets, sampling_opts, use_weights)
    print(means)
    print(errors)
    plot_means(args, means, datasets)
    plot_errors(args, errors, datasets)

    args.datasets = datasets
    torch.save(args, os.path.join(args.dir, 'args.pt'))

    print('Ends')

    sys.exit(0)

