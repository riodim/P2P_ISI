import torch
import numpy as np

def RC_pulses_create():
    mu=7
    time_interval = mu/2

    samples = np.linspace(-time_interval,time_interval,3501)
    roll_off = np.arange(0.1,1,0.2)

    beta = 2*np.log(2)/roll_off
    RC_pulses_roll_off = np.zeros((len(roll_off),len(samples)))
    Exp_pulses_roll_off = np.zeros((len(roll_off),len(samples)))
    
    zeros_RC = 1 / (2*roll_off)
    
    for w1 in range(len(roll_off)):
        RC_pulses_roll_off[w1,:] = np.sinc(samples)*((np.cos(np.pi*roll_off[w1]*samples))/(1-np.square(2*roll_off[w1]*samples)))
        if np.isin(zeros_RC[w1],samples):
            index_pos_zero = np.where(zeros_RC[w1] == samples)[0]
            RC_pulses_roll_off[w1,index_pos_zero] = (np.pi / 4)*np.sinc(zeros_RC[w1])
        if np.isin(-zeros_RC[w1],samples):
            index_neg_zero = np.where(-zeros_RC[w1] == samples)[0]
            RC_pulses_roll_off[w1,index_neg_zero] = (np.pi / 4)*np.sinc(zeros_RC[w1])

        term1 = 2*beta[w1]**2*np.cos(np.pi*roll_off[w1]*samples) - beta[w1]**2
        term2 = 4*np.pi*beta[w1]*samples*np.sin(np.pi*roll_off[w1]*samples)
        term3 = beta[w1]**2 + np.square(2*np.pi*samples)
    
        Exp_pulses_roll_off[w1,:] = np.sinc(samples)*(term1+term2)/term3
    
    RC_pulses = torch.tensor(RC_pulses_roll_off)
    Exp_pulses = torch.tensor(Exp_pulses_roll_off)
    
    return RC_pulses, Exp_pulses

"""
Array will keep the mean square values at ISI points plus 0 and mean value at 0 
"""
def pulse_statistics(pulse,mu,pmf):
    statistics_interest = torch.zeros(mu+1)
    # Half period
    T = np.floor(len(pulse)/(2*mu)).astype(int)
    center = np.floor(mu/2).astype(int)
    mid_sample = np.floor(len(pulse)/2).astype(int)
    
    # for m in range(mu):
    #     statistics_interest[m] = torch.sum(torch.square(pulse[m*2*T:(m+1)*2*T+1])*pmf[mid_sample-T:mid_sample+T+1])
    
    statistics_interest[3] = torch.sum(torch.square(pulse[3*2*T:(3+1)*2*T+1])*pmf[mid_sample-T:mid_sample+T+1])
    statistics_interest[mu] = torch.sum(pulse[center*2*T:(center+1)*2*T+1]*pmf[mid_sample-T:mid_sample+T+1])
    return statistics_interest
