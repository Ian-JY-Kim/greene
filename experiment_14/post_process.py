import csv
import os
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import random
from scipy.stats import truncnorm
from scipy.optimize import fsolve
from scipy.optimize import minimize
import math
from scipy.stats import norm
from statistics import NormalDist
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.integrate import tplquad
from itertools import product
from parameter import *


# import data
df_master = pd.read_csv("MC_master_data.csv")

# read computed result
path_0731 = './result/'
path_0731_list = os.listdir(path_0731)

def data_read(path):
    file_list = os.listdir(path)

    job_num_list = []
    xi_1_star_list = []
    xi_2_star_list = []

    for i in file_list:
        file_name = path + i
        f = open(file_name, 'r', encoding='utf-8')
        rdr = csv.reader(f, delimiter='\t')
        r = list(rdr)

        job_num_list.append(int(r[0][0]))
        xi_1_star_list.append(float(r[0][1]))
        xi_2_star_list.append(float(r[0][2]))
    
    df_return = df({"job_num": job_num_list, "xi_1_star": xi_1_star_list, "xi_2_star": xi_2_star_list})

    return df_return

df_xi_star = data_read(path_0731)
df_xi_star = df_xi_star.loc[df_xi_star['xi_1_star'] != 100]

# merge the computed result to the original data
df_master = pd.merge(df_master, df_xi_star, left_on="job_array_num", right_on='job_num')

def monopoly(x, w, xi, omega):
    mc = np.exp(gamma*w + omega)
    T = np.exp(beta*x - alpha*mc + xi)

    #(1) find Y
    def monopoly_eqn(var):
        Y = var
        eq = 1 - Y + T*np.exp(-Y)
        return eq
    Y = fsolve(monopoly_eqn, 1)[0]
    
    pi = (1/alpha)*(Y-1) 
    price = Y/alpha + mc
    share = pi/(price-mc)

    return pi, price, share
vec_monopoly = np.vectorize(monopoly)

def duopoly(x_1, x_2, w_1, w_2, xi_1, xi_2, omega_1, omega_2):
    try:
        mc_1 = np.exp(gamma*w_1 + omega_1) 
        mc_2 = np.exp(gamma*w_2 + omega_2)
        T_1 = np.exp(beta*x_1 - alpha*mc_1 + xi_1)
        T_2 = np.exp(beta*x_2 - alpha*mc_2 + xi_2)
        
        def duopoly_fun(Y):
            Y_1, Y_2 = Y
            eqn1 = Y_1 - math.log(T_1*(Y_2-1)) + math.log(1-Y_2+T_2*np.exp(-Y_2))        
            return abs(eqn1)
        
        def c1(Y):
            'Y_1 exp term greater than 0'
            Y_1, Y_2 = Y
            return 1-Y_1+T_1*np.exp(-Y_1)

        def c2(Y):
            'Y_2 exp term greater than 0'
            Y_1, Y_2 = Y 
            return 1-Y_2+T_2*np.exp(-Y_2)
        
        def c3(Y):
            Y_1, Y_2 = Y
            return Y_2 - math.log(T_2*(Y_1-1)) + math.log(1-Y_1+T_1*np.exp(-Y_1))

        bnds = ((1.000001, None), (1.000001, None))
        cons = ({'type': 'ineq', 'fun': c1}, 
                {'type': 'ineq', 'fun': c2},
                {'type': 'eq', 'fun': c3})
        initial_point = (1.0001, 1.0001)
        res = minimize(duopoly_fun, initial_point, method = 'SLSQP', bounds=bnds, constraints=cons)
        Y_1 = res.x[0]
        Y_2 = res.x[1]
        
        pi_1 = (1/alpha)*(Y_1-1)
        pi_2 = (1/alpha)*(Y_2-1)

        price_1 = Y_1/alpha + mc_1
        price_2 = Y_2/alpha + mc_2

        share_1 = pi_1/(price_1 - mc_1)
        share_2 = pi_2/(price_2 - mc_2)

        return pi_1, pi_2, price_1, price_2, share_1, share_2
    
    except:
        return 100, 100, 100, 100, 100, 100
vec_duopoly = np.vectorize(duopoly)

df_master['MS'] = (df_master['xi_1'].values >= df_master['xi_1_star'].values).astype(int) + 2*(df_master['xi_2'].values >= df_master['xi_2_star'].values).astype(int)


# generate subsamples according to the market structure
df_master_ms0 = df_master.loc[df_master['MS']==0]
df_master_ms1 = df_master.loc[df_master['MS']==1]
df_master_ms2 = df_master.loc[df_master['MS']==2]
df_master_ms3 = df_master.loc[df_master['MS']==3]


df_master_ms0['price_1'] = 100
df_master_ms0['price_2'] = 100
df_master_ms0['share_1'] = 100
df_master_ms0['share_2'] = 100

f1_temp = vec_monopoly(df_master_ms1['x_1'].values, df_master_ms1['w_1'].values, df_master_ms1['xi_1'].values, df_master_ms1['omega_1'].values)
df_master_ms1['price_1'] = f1_temp[1]
df_master_ms1['price_2'] = 100
df_master_ms1['share_1'] = f1_temp[2]
df_master_ms1['share_2'] = 100

f2_temp = vec_monopoly(df_master_ms2['x_2'].values, df_master_ms2['w_2'].values, df_master_ms2['xi_2'].values, df_master_ms2['omega_2'].values)
df_master_ms2['price_1'] = 100
df_master_ms2['price_2'] = f2_temp[1]
df_master_ms2['share_1'] = 100
df_master_ms2['share_2'] = f2_temp[2]

f1f2_temp = vec_duopoly(df_master_ms3['x_1'].values, df_master_ms3['x_2'].values, df_master_ms3['w_1'].values, df_master_ms3['w_2'].values, df_master_ms3['xi_1'].values, df_master_ms3['xi_2'].values, df_master_ms3['omega_1'].values, df_master_ms3['omega_2'].values)
df_master_ms3['price_1'] = f1f2_temp[2]
df_master_ms3['price_2'] = f1f2_temp[3]
df_master_ms3['share_1'] = f1f2_temp[4]
df_master_ms3['share_2'] = f1f2_temp[5]

df_master_ms3 = df_master_ms3.loc[df_master_ms3['price_1'] != 100]


# generate final data
df_final = df_master_ms0.copy()
df_final = df_final.append(df_master_ms1)
df_final = df_final.append(df_master_ms2)
df_final = df_final.append(df_master_ms3)

df_final.to_csv("df_final.csv")
df_final_shuffled = df_final.sample(frac = 1)
df_final_shuffled['mc_num'] = range(1, df_final_shuffled.shape[0]+1)
df_final_shuffled.to_csv("df_final_shuffled.csv")