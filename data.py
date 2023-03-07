#!/usr/bin/env python
# coding: utf-8

'''
Code for reading ACS, BRFSS datasets
'''

import numpy as np
from functools import partial
from joblib import Parallel, delayed
from pathlib import Path
import torch

from folktables import ACSDataSource, ACSPublicCoverage
from folktables import ACSEmployment
from folktables import BasicProblem
from folktables import adult_filter

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

NOT_SAMP = -999

"""
ACS

Numeric variables
PWGTP - person weight
FINCP - family income
HINCP - household income
PAP - public assistance income past 12 months
RETP - retirement income past 12 months
VALP - property value

CITWP - Year of naturalization write-in
YOEP - Year of entry
MARHYP - Year last married

JWMNP - travel time to work
INTP - Interest, dividends, and net rental income past 12 months (signed, use ADJINC to adjust to constant dollars)
JWRIP - Vehicle occupancy
OIP - All other income past 12 months (use ADJINC to adjust to constant dollars)
PAP - Public assistance income past 12 months (use ADJINC to adjust to constant dollars)
RETP - Retirement income past 12 months (use ADJINC to adjust to constant dollars)
SEMP - Self-employment income past 12 months (signed, use ADJINC to adjust SEMP to constant dollars)
SSIP - Supplementary Security Income past 12 months (use ADJINC to adjust SSIP to constant dollars)
SSP - Social Security income past 12 months (use ADJINC to adjust SSP to constant dollars)
WAGP - Wages or salary income past 12 months (use ADJINC to adjust WAGP to constant dollars)
WKHP - usual hours worked per week past 12 months
PERNP - Total person's earnings (use ADJINC to adjust to constant dollars)
PINCP - Total person's income (signed, use ADJINC to adjust to constant dollars)
POVPIP - Income-to-poverty ratio recode

Race variable
Recoded detailed race code 
1 .White alone 
2 .Black or African American alone 
3 .American Indian alone 
4 .Alaska Native alone 
5 .American Indian and Alaska Native tribes specified; or .American Indian or Alaska Native, not specified and no other .races 
6 .Asian alone 
7 .Native Hawaiian and Other Pacific Islander alone 
8 .Some Other Race alone 
9 .Two or More Races

BRFSS

Codebook https://www.cdc.gov/brfss/annual_data/2014/pdf/codebook14_llcp.pdf
diabindicator DIABETE3==1 - diabetes indicator except when during pregnancy for female respondents
sleeptime SLEPTIM1 - average sleep in 24 hours
diabage DIABAGE2 - age when diagnosed with diabetes
"""

ACS_RACE_CODE = {
    'White': 1,
    'BlackorAA': 2,
    'Asian': 6,
}

def adult_filter_income_quantile(data):
    """Mimic the filters in place for Adult data.
    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    """
    df = data
    df = df[df['AGEP'] > 16]
    # df = df[df['PINCP'] > 100]
    df = df[
        (df['PINCP'] > df['PINCP'].quantile(0.25)) &
        (df['PINCP'] < df['PINCP'].quantile(0.75))
    ]
    df = df[df['WKHP'] > 0]
    df = df[df['PWGTP'] >= 1]
    return df

def public_coverage_filter(data):
    """
    Filters for the public health insurance prediction task; focus on low income Americans, and those not eligible for Medicare
    """
    df = data
    df = df[df['AGEP'] < 65]
    df = df[df['PINCP'] <= 30000]
    return df

ACSPublicCoverageRACE = BasicProblem(
    features=[
        'PWGTP',
        'AGEP',
        'PINCP',
        'RAC1P',
        'RACAIAN',
        'RACASN',
        'RACBLK',
        'RACNH',
        'RACPI',
        'RACSOR',
        'RACWHT',
        'FHISP',
    ],
    target='PRIVCOV',
    target_transform=lambda x: x == 1,
    group='RAC1P',
    preprocess=lambda df: public_coverage_filter(df),
    postprocess=lambda x: np.nan_to_num(x, -1),
)

def create_acs_query_object(outcome):
    if outcome == 'PRIVCOV':
        return ACSPublicCoverageRACE
    else:
        ACSRaceOutcome = BasicProblem(
            features=[
                'PWGTP',
                'RAC1P',
                'RACAIAN',
                'RACASN',
                'RACBLK',
                'RACNH',
                'RACPI',
                'RACSOR',
                'RACWHT',
                'FHISP',
            ],
            target=outcome,
            target_transform=lambda x: x,
            group='RAC1P',
            preprocess=lambda x: x,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        return ACSRaceOutcome

def travel_time_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['PWGTP'] >= 1]
    df = df[df['ESR'] == 1]
    df = df[df['JWMNP'].notna()]
    return df

def create_acs_query_object_model(outcome):
    if outcome == 'income_binary':
        ACSIncomeClassification = BasicProblem(
            features=[
                'RAC1P',
                'AGEP',
                'COW',
                'SCHL',
                'MAR',
                'OCCP',
                'POBP',
                'RELP',
                'WKHP',
                'SEX',
                'PWGTP',
                'FHISP',
            ],
            target='PINCP',
            target_transform=lambda x: x > 50000,    
            group='RAC1P',
            preprocess=adult_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        return ACSIncomeClassification
    elif outcome == 'income_real':
        ACSIncomeRegression = BasicProblem(
            features=[
                'RAC1P',
                'AGEP',
                'COW',
                'SCHL',
                'MAR',
                'OCCP',
                'POBP',
                'RELP',
                'WKHP',
                'SEX',
                'PWGTP',
                'FHISP',
            ],
            target='PINCP',
            target_transform=lambda x: x,    
            group='RAC1P',
            preprocess=adult_filter_income_quantile,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        return ACSIncomeRegression
    elif outcome == 'coverage_binary':
        ACSPublicCoverageClassification = BasicProblem(
            features=[
                'RAC1P',
                'AGEP',
                'SCHL',
                'MAR',
                'SEX',
                'DIS',
                'ESP',
                'CIT',
                'MIG',
                'MIL',
                'ANC',
                'NATIVITY',
                'DEAR',
                'DEYE',
                'DREM',
                'PINCP',
                'ESR',
                'ST',
                'FER',
                'PWGTP',
                'FHISP',
            ],
            target='PUBCOV',
            target_transform=lambda x: x == 1,
            group='RAC1P',
            preprocess=public_coverage_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        return ACSPublicCoverageClassification
    elif outcome == 'travel_binary':
        ACSTravelTimeClassification = BasicProblem(
            features=[
                'RAC1P',
                'AGEP',
                'SCHL',
                'MAR',
                'SEX',
                'DIS',
                'ESP',
                'MIG',
                'RELP',
                'PUMA',
                'ST',
                'CIT',
                'OCCP',
                'JWTR',
                'POWPUMA',
                'POVPIP',
                'PWGTP',
                'FHISP',
            ],
            target="JWMNP",
            target_transform=lambda x: x > 20,
            group='RAC1P',
            preprocess=travel_time_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        return ACSTravelTimeClassification
    elif outcome == 'travel_real':
        ACSTravelTimeRegression = BasicProblem(
            features=[
                'RAC1P',
                'AGEP',
                'SCHL',
                'MAR',
                'SEX',
                'DIS',
                'ESP',
                'MIG',
                'RELP',
                'PUMA',
                'ST',
                'CIT',
                'OCCP',
                'JWTR',
                'POWPUMA',
                'POVPIP',
                'PWGTP',
                'FHISP',
            ],
            target="JWMNP",
            target_transform=lambda x: x,
            group='RAC1P',
            preprocess=travel_time_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        return ACSTravelTimeRegression
    else:
        raise NotImplementedError(outcome)
    
def get_acs_data(rng, npop, frac, outcome, us_state, group, use_weights=False):
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person',
                                root_dir='data')
    data = data_source.get_data(states=[us_state], download=False)
    ACSQuery = create_acs_query_object(outcome)
    X_acs, Y_acs, G_acs = ACSQuery.df_to_pandas(data)
    Y_acs = Y_acs.to_numpy().squeeze()
    G_acs = G_acs.to_numpy().squeeze()
    if use_weights:
        W_acs = X_acs['PWGTP'].to_numpy().squeeze()
    else:
        W_acs = np.ones((X_acs.shape[0],))

    # Remove rows with nan target value
    nan_indices = np.isnan(Y_acs)
    X_acs = X_acs[~nan_indices]
    Y_acs = Y_acs[~nan_indices]
    G_acs = G_acs[~nan_indices]
    W_acs = W_acs[~nan_indices]

    Y_acs = Y_acs.astype(float)
    print("Counts and Y mean per group", [(i, np.sum(G_acs==i), Y_acs[G_acs==i].mean()) for i in np.unique(G_acs)])

    # Select individuals with race variable as 1 .White alone or 2 .Black or African American alone or 6 .Asian alone
    if group!='Hispanic':
        group_code = ACS_RACE_CODE[group]
        subset_by_race = np.isin(G_acs, [1,group_code])
        G_acs = G_acs[subset_by_race]
        Y_acs = Y_acs[subset_by_race]
        W_acs = W_acs[subset_by_race]
        G_acs = (G_acs==group_code).astype(int)
    else:
        G_acs = (X_acs['FHISP']==1.0).to_numpy().squeeze()

    # Subsample to given group proportion
    if frac is not None:
        desired_A1 = int(frac*npop)
        desired_A0 = npop - desired_A1
        indices = np.arange(G_acs.shape[0])
        indices_A0 = indices[G_acs!=1]
        indices_A1 = indices[G_acs==1]
        assert (len(indices_A0)>=desired_A0) and (len(indices_A1)>=desired_A1), "Desired fraction not possible"
        random_indices_A0 = rng.choice(indices_A0, size=desired_A0)
        random_indices_A1 = rng.choice(indices_A1, size=desired_A1)
        Y_acs = np.concatenate([
            Y_acs[random_indices_A0], Y_acs[random_indices_A1],
        ], axis=0)
        G_acs = np.concatenate([
            G_acs[random_indices_A0], G_acs[random_indices_A1],
        ], axis=0)
        W_acs = np.concatenate([
            W_acs[random_indices_A0], W_acs[random_indices_A1],
        ], axis=0)
    G_acs_ = G_acs[:, np.newaxis]
    W_acs_ = W_acs[:, np.newaxis]
    X_acs = np.concatenate([np.ones_like(G_acs_), G_acs_, W_acs_], axis=1)
  
    # Shuffle rows
    shuffled_indices = rng.permutation(X_acs.shape[0])
    X_acs, Y_acs = X_acs[shuffled_indices], Y_acs[shuffled_indices]
    P_acs = np.ones((X_acs.shape[0],))
    S_acs = P_acs

    # Keep npop samples
    X_acs = X_acs[:npop]
    Y_acs = Y_acs[:npop]
    P_acs = P_acs[:npop]
    S_acs = S_acs[:npop]
    print("Achieved counts and Y mean per group", [(i, np.sum(X_acs[:,1]==i), Y_acs[X_acs[:,1]==i].mean()) for i in np.unique(X_acs[:,1])])

    return X_acs, P_acs, S_acs, Y_acs

BRFSS_RACE_CODE = {
    'White': 1.0,
    'BlackorAA': 2.0,
    'Asian': 4.0,
    'AmericanIndianAlaskan': 3.0,
}

def get_brfss_data(rng, npop, frac, outcome, us_state, group, use_weights=False):
    '''
    Codebook for 2014 BRFSS https://www.cdc.gov/brfss/annual_data/2014/pdf/codebook14_llcp.pdf
    '''
    data_source = pd.read_csv('data/2014_diabage_sleeptime.csv')
    data_source = data_source[data_source['_RACE']!=9.0]  # remove responses Don’t know/Not sure/Refused
    if us_state=='all':
        data = data_source
    else:
        state_fips_codes = pd.read_csv('data/us-state-ansi-fips.csv')
        us_state_code = state_fips_codes[state_fips_codes[' stusps']==f' {us_state}'][' st'].item()
        data = data_source[data_source['_STATE']==us_state_code]
    if outcome=='diabage':
        data = data[~data['DIABAGE2'].isin([98.0,99.0])]
        Y_acs = data['DIABAGE2']
    elif outcome=='diabindicator':
        data = data[~data['DIABETE3'].isin([7.0,9.0])]
        Y_acs = (data['DIABETE3']==1.0).astype(int)
    elif outcome=='sleeptime':
        data = data[~data['SLEPTIM1'].isin([77.0,99.0])]
        Y_acs = data['SLEPTIM1']
    else:
        raise NotImplementedError(outcome)
    X_acs = data[['_LLCPWT','SEX','_RACE']]
    G_acs = data['_RACE']
    Y_acs = Y_acs.to_numpy().squeeze()
    G_acs = G_acs.to_numpy().squeeze()
    if use_weights:
        W_acs = X_acs['_LLCPWT'].to_numpy().squeeze()
    else:
        W_acs = np.ones((X_acs.shape[0],))

    # Remove rows with nan target or race feature value
    nan_indices = np.isnan(Y_acs) | np.isnan(G_acs)
    X_acs = X_acs[~nan_indices]
    Y_acs = Y_acs[~nan_indices]
    G_acs = G_acs[~nan_indices]
    W_acs = W_acs[~nan_indices]

    Y_acs = Y_acs.astype(float)
    print("Counts and Y mean per group", [(i, np.sum(G_acs==i), Y_acs[G_acs==i].mean()) for i in np.unique(G_acs)])
  
    # Select individuals with race variable as 1 .White alone or 2 .Black or African American alone or 6 .Asian alone
    if group!='Hispanic':
        group_code = BRFSS_RACE_CODE[group]
        subset_by_race = np.isin(G_acs, [1.0,group_code])
        G_acs = G_acs[subset_by_race]
        Y_acs = Y_acs[subset_by_race]
        W_acs = W_acs[subset_by_race]
        G_acs = (G_acs==group_code).astype(int)
    else:
        G_acs = (X_acs['_RACE']==8.0).to_numpy().squeeze()

    # Subsample to given group proportion
    if frac is not None:
        desired_A1 = int(frac*npop)
        desired_A0 = npop - desired_A1
        indices = np.arange(G_acs.shape[0])
        indices_A0 = indices[G_acs!=1]
        indices_A1 = indices[G_acs==1]
        assert (len(indices_A0)>=desired_A0) and (len(indices_A1)>=desired_A1), "Desired fraction not possible"
        random_indices_A0 = rng.choice(indices_A0, size=desired_A0)
        random_indices_A1 = rng.choice(indices_A1, size=desired_A1)
        Y_acs = np.concatenate([
            Y_acs[random_indices_A0], Y_acs[random_indices_A1],
        ], axis=0)
        G_acs = np.concatenate([
            G_acs[random_indices_A0], G_acs[random_indices_A1],
        ], axis=0)
        W_acs = np.concatenate([
            W_acs[random_indices_A0], W_acs[random_indices_A1],
        ], axis=0)
    G_acs_ = G_acs[:, np.newaxis]
    W_acs_ = W_acs[:, np.newaxis]
    X_acs = np.concatenate([np.ones_like(G_acs_), G_acs_, W_acs_], axis=1)
  
    # Shuffle rows
    shuffled_indices = rng.permutation(X_acs.shape[0])
    X_acs, Y_acs = X_acs[shuffled_indices], Y_acs[shuffled_indices]
    P_acs = np.ones((X_acs.shape[0],))
    S_acs = P_acs

    # Keep npop samples
    X_acs = X_acs[:npop]
    Y_acs = Y_acs[:npop]
    P_acs = P_acs[:npop]
    S_acs = S_acs[:npop]
    print("Achieved counts and Y mean per group", [(i, np.sum(X_acs[:,1]==i), Y_acs[X_acs[:,1]==i].mean()) for i in np.unique(X_acs[:,1])])

    return X_acs, P_acs, S_acs, Y_acs

def get_acs_data_model(rng, npop, frac, outcome, us_state, group, use_weights=False):
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person',
                              root_dir='data')
    data = data_source.get_data(states=[us_state], download=False)
    ACSQuery = create_acs_query_object_model(outcome)
    X_acs, Y_acs, G_acs = ACSQuery.df_to_pandas(data)
    Y_acs = Y_acs.to_numpy().squeeze()
    G_acs = G_acs.to_numpy().squeeze()
    if use_weights:
        W_acs = X_acs['PWGTP'].to_numpy().squeeze()
    else:
        W_acs = np.ones((X_acs.shape[0],))

    # Remove rows with nan target value
    nan_indices = np.isnan(Y_acs)
    X_acs = X_acs[~nan_indices]
    Y_acs = Y_acs[~nan_indices]
    G_acs = G_acs[~nan_indices]
    W_acs = W_acs[~nan_indices]

    Y_acs = Y_acs.astype(float)
    print("Counts and Y mean per group", [(i, np.sum(G_acs==i), Y_acs[G_acs==i].mean()) for i in np.unique(G_acs)])

    # Select individuals with race variable as 1 .White alone or 2 .Black or African American alone or 6 .Asian alone
    if group!='Hispanic':
        group_code = ACS_RACE_CODE[group]
        subset_by_race = np.isin(G_acs, [1,group_code])
        X_acs = X_acs[subset_by_race]
        G_acs = G_acs[subset_by_race]
        Y_acs = Y_acs[subset_by_race]
        W_acs = W_acs[subset_by_race]
        G_acs = (G_acs==group_code).astype(int)
    else:
        G_acs = (X_acs['FHISP']==1.0).to_numpy().squeeze()
    X_acs = X_acs.drop(['PWGTP','FHISP'], axis=1)
    
    X_train, X_test, Y_train, Y_test, G_train, G_test, W_train, W_test = train_test_split(
        X_acs, Y_acs, G_acs, W_acs, test_size=0.3, random_state=0)
    
    if outcome in ['income_binary','coverage_binary','travel_binary']:
        model = make_pipeline(StandardScaler(), GradientBoostingClassifier())
    elif outcome in ['income_real','travel_real']:
        model = make_pipeline(StandardScaler(), GradientBoostingRegressor())
    else:
        raise NotImplementedError(f'Model for {outcome}')
    model.fit(X_train, Y_train)
    
    Yhat = model.predict(X_test)
    
    if outcome in ['income_binary','coverage_binary','travel_binary']:
        Y_acs = Yhat[Y_test==1]
        G_acs = G_test[Y_test==1]
        W_acs = W_test[Y_test==1]
    elif outcome in ['income_real','travel_real']:
        Y_acs = (Yhat - Y_test)**2
        G_acs = G_test
        W_acs = W_test
    else:
        raise NotImplementedError(f'Loss for {outcome}')
    
    # Subsample to given group proportion
    if frac is not None:
        desired_A1 = int(frac*npop)
        desired_A0 = npop - desired_A1
        indices = np.arange(G_acs.shape[0])
        indices_A0 = indices[G_acs!=1]
        indices_A1 = indices[G_acs==1]
        assert (len(indices_A0)>=desired_A0) and (len(indices_A1)>=desired_A1), "Desired fraction not possible"
        random_indices_A0 = rng.choice(indices_A0, size=desired_A0)
        random_indices_A1 = rng.choice(indices_A1, size=desired_A1)
        Y_acs = np.concatenate([
            Y_acs[random_indices_A0], Y_acs[random_indices_A1],
        ], axis=0)
        G_acs = np.concatenate([
            G_acs[random_indices_A0], G_acs[random_indices_A1],
        ], axis=0)
        W_acs = np.concatenate([
            W_acs[random_indices_A0], W_acs[random_indices_A1],
        ], axis=0)
    G_acs_ = G_acs[:, np.newaxis]
    W_acs_ = W_acs[:, np.newaxis]
    X_acs = np.concatenate([np.ones_like(G_acs_), G_acs_, W_acs_], axis=1)

    # Shuffle rows
    shuffled_indices = rng.permutation(X_acs.shape[0])
    X_acs, Y_acs = X_acs[shuffled_indices], Y_acs[shuffled_indices]
    P_acs = np.ones((X_acs.shape[0],))
    S_acs = P_acs

    # Keep npop samples
    X_acs = X_acs[:npop]
    Y_acs = Y_acs[:npop]
    P_acs = P_acs[:npop]
    S_acs = S_acs[:npop]
    print("Achieved counts and Y mean per group", [(i, np.sum(X_acs[:,1]==i), Y_acs[X_acs[:,1]==i].mean()) for i in np.unique(X_acs[:,1])])

    return X_acs, P_acs, S_acs, Y_acs

def get_synthetic_data(rng, args, npop):
    sigma_all = lambda x: (1,1)
    train_data_generator = Simulate_data_synthetic(rng, args.sim_noise, args.sim_fract)
    data = gendata_single(npop, sigma_all, train_data_generator)
    X, P, S, Y = zip(*data)
    X, P, S, Y = np.array(X), np.array(P), np.array(S), np.array(Y)
    if len(X.shape)==1:
        X = X[:,np.newaxis]
    print("Population generated. Size={}".format(npop))
    assert X.shape[0]==npop, "Population incorrectly sampled"
    return X, P, S, Y

def gendata_single(n_0, sigma_0, gen_xy):
    data_0 = []
    gen_xy.reset()
    for _ in range(n_0):
        x = gen_xy.observe()
        y_act = gen_xy.reward(None, None)
        p_0, s_0 = sigma_0(x)
        y = y_act if s_0==1 else NOT_SAMP
        data_0.append((x,p_0,s_0,y))
    return data_0


def gendata_multiple(n_0, sigmas, gen_xy):
    data = [[] for _ in range(len(sigmas))]
    gen_xy.reset()
    for _ in range(n_0):
        x = gen_xy.observe()
        y_act = gen_xy.reward(None, None)
        for i in range(len(sigmas)):
            sigma = sigmas[i]
            p, s = sigma(x)
            y = y_act if s==1 else NOT_SAMP
            data[i].append((x,p,s,y))
    return data

def sigma_uniform_sample(rng, rate):
    return lambda x: (rate, rng.binomial(1, p=rate))

def sigma_group_sample(rng, x, rate_A0, rate_A1):
    if x[1]==1:  # TODO allow group information at arbitrary index
        return rate_A1, rng.binomial(1, p=rate_A1)
    else:
        return rate_A0, rng.binomial(1, p=rate_A0)

class Simulate_data_synthetic(object):
    def __init__(self, rng, noise, fract, mu=200, delta_mu=80):
        self.rng = rng
        self.dim = 6 # including intercept
        self.step = -1
        self.noise = noise
        self.fract = fract
        self.mu = mu
        self.delta_mu = delta_mu
    def reset(self):
        self.step = 0
        return None
    def observe(self):
        self.step+=1
        # A is gender
        A = self.rng.binomial(1, self.fract)
        # X1,X2,X3 are therapy, antidepressant use, regular physical exercise
        X1 = self.rng.binomial(1, 0.3)
        X2 = self.rng.binomial(1, 0.25)
        X3 = self.rng.binomial(1, 1/(1+np.exp(-1*(-0.5 + 0.75*A + 1*X1 + 1.5*X2))))
        # Y is CES-D score
        # self.Y = 24 - 3*A + 3*X1 - 4*X2 - 6*X3 - 1.5*A*X3 + self.rng.normal(loc=0,scale=4.5)
        
        self.Y = self.mu + self.delta_mu*(1-A) + self.rng.normal(loc=0,scale=10+(1-A)*self.noise)
        # self.Y = self.mu + self.delta_mu*(1-A) + self.rng.standard_t(df= 1 + int(A*self.noise))
        # self.Y = 24 + 3*A - 4*X1 - 6*X2 - 3*X3 - 1.5*X2*X3 + self.rng.normal(loc=0,scale=4 + (1-A)*self.noise)
        # self.Y = 24 + 3*A - 4*X1 - 6*X2 - 3*X3 - 1.5*X2*X3 + self.rng.standard_t(df= 2 + int(A*self.noise))
        self.X = np.array([1,A,1,X1,X2,X3])

        # # Variance depending on X
        # # X1 = self.rng.uniform(-1,1)
        # X1 = self.rng.normal(0,1)
        # self.Y = 8*X1 + 2*np.exp(X1)*self.rng.normal(0,1)
        # # self.Y = 8*X1 + 2*np.exp(X1)*self.rng.standard_t(4)
        # self.X = np.array([1,X1])

        # # Variance depending on A
        # X1 = self.rng.normal(0,1)
        # A = self.rng.binomial(1, 0.5)
        # self.Y = 8*X1 + np.exp(A*X1)*self.rng.normal(0,2)
        # self.X = np.array([1,A,X1])

        return self.X
    def reward(self, context, action):
        return self.Y

class Simulate_data(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.step = -1
    def reset(self):
        self.step = 0
        return self.x[self.step,:]
    def observe(self):
        self.step+=1
        if self.step==self.x.shape[0]:
            self.step=0 # loop again through x
        return self.x[self.step,:]
    def reward(self, context, action):
        return self.y[self.step]

class Dataset():
    def __init__(self, rng, args):
        self.rng = rng
        self.args = args
    def population_data(self, npop):
        if self.args.dataset_name=='synthetic':
            X_full, _, _, Y_full = get_synthetic_data(self.rng, self.args, npop)
        elif self.args.dataset_name=='acs':
            X_full, _, _, Y_full = get_acs_data(self.rng, npop, self.args.sim_fract, self.args.outcome, 
                                                us_state=self.args.us_state, group=self.args.group,
                                                use_weights=self.args.use_weights)
        elif self.args.dataset_name=='brfss':
            X_full, _, _, Y_full = get_brfss_data(self.rng, npop, self.args.sim_fract, self.args.outcome, 
                                                us_state=self.args.us_state, group=self.args.group,
                                                use_weights=self.args.use_weights)
        elif self.args.dataset_name=='acs_model':
            X_full, _, _, Y_full = get_acs_data_model(self.rng, npop, self.args.sim_fract, self.args.outcome, 
                                                us_state=self.args.us_state, group=self.args.group,
                                                use_weights=self.args.use_weights)
        else:
            raise NotImplementedError(f"{self.args.dataset_name}")
        return X_full, Y_full