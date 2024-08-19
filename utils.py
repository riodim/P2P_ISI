# -*- coding: utf-8 -*-

import numpy as np
import torch

def generate_qam_symbols(num_symbols, M):
    # Calculate the number of points along each axis
    sqrt_M = int(np.sqrt(M))
    
    # Generate all possible combinations of psi and zeta
    psi = np.arange(-(sqrt_M - 1), sqrt_M, 2)
    zeta = np.arange(-(sqrt_M - 1), sqrt_M, 2)
    
    # Create a grid of the combinations
    Re_ck_j, Im_ck_j = np.meshgrid(psi, zeta)
    
    # Flatten the grids and combine to create all possible symbols
    Re_ck_j = Re_ck_j.flatten()
    Im_ck_j = Im_ck_j.flatten()
    
    # Combine real and imaginary parts to form complex symbols
    symbols = Re_ck_j + 1j * Im_ck_j
    
    # Randomly choose num_symbols from the 64-QAM symbols
    chosen_indices = np.random.choice(len(symbols), num_symbols, replace=True)
    symbols = symbols[chosen_indices]
    
    # Normalize symbols to ensure average power is 1
    normalization_factor = np.sqrt(np.mean(np.abs(symbols) ** 2))
    symbols /= normalization_factor
    
    return symbols

import matplotlib.pyplot as plt

def plot_constellation(symbols, file_name=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(symbols.real, symbols.imag, color='blue', s=10)
    plt.grid(True)
    plt.title("64-QAM Constellation Diagram")
    plt.xlabel("In-phase (Real)")
    plt.ylabel("Quadrature (Imaginary)")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    if file_name:
        plt.savefig(file_name)
        print(f"Constellation diagram saved as {file_name}")
    else:
        plt.show()

def MSE_sampling_ISI(mu, b, x_real, x_imag, x_ISI_real, x_ISI_imag, channels, ISI_channels, sample_time, T, dnn_out, batch_size, device):
    num_ISI = np.floor(mu / 2).astype(int)

    y_ISI_real = torch.zeros(batch_size).to(device)
    y_ISI_imag = torch.zeros(batch_size).to(device)
    y_rec_real = torch.zeros(batch_size).to(device)
    y_rec_imag = torch.zeros(batch_size).to(device)

    index = 0
    for w2 in range(mu):
        dnn_out_index = sample_time + (w2 - num_ISI) * 2 * T
        if dnn_out_index >= dnn_out.size(1) or dnn_out_index < 0:
            #print(f"Skipping out-of-bounds index {dnn_out_index}")
            continue  # Skip if the index is out of bounds
        
        if w2 != num_ISI:
            dnn_out_val = dnn_out[:, dnn_out_index].unsqueeze(1)  # Shape [100, 1]
            
            ISI_channel_current = ISI_channels[:, index].view(1, -1).expand(batch_size, -1)  # Shape [100, 6]
            x_ISI_real_current = x_ISI_real[:, index].unsqueeze(-1)  # Shape [100, 1]
            x_ISI_imag_current = x_ISI_imag[:, index].unsqueeze(-1)  # Shape [100, 1]
                        
            y_ISI_real += (b * ISI_channel_current * x_ISI_real_current * dnn_out_val).sum(dim=1)
            y_ISI_imag += (b * ISI_channel_current * x_ISI_imag_current * dnn_out_val).sum(dim=1)
            
            index += 1
        else:
            dnn_out_val = dnn_out[:, sample_time].unsqueeze(1)
            y_rec_real = b * channels * x_real * dnn_out_val.squeeze(1)
            y_rec_imag = b * channels * x_imag * dnn_out_val.squeeze(1)

    y_ISI_total_real = y_ISI_real + y_rec_real
    y_ISI_total_imag = y_ISI_imag + y_rec_imag

    return y_ISI_total_real, y_ISI_total_imag
        
def calculate_mse_with_isi(K, mu, bk, x_real, x_imag, x_ISI_real, x_ISI_imag, channels, ISI_channels, sample_time, T, dnn_out, batch_size, device, y_target_real, y_target_imag):
    y_ISI_total_real, y_ISI_total_imag = MSE_sampling_ISI(K, mu, bk, x_real, x_imag, x_ISI_real, x_ISI_imag, channels, ISI_channels, sample_time, T, dnn_out, batch_size, device)
    
    mse_real = torch.mean(torch.square(y_ISI_total_real - y_target_real))
    mse_imag = torch.mean(torch.square(y_ISI_total_imag - y_target_imag))
    
    mse_total = mse_real + mse_imag
    
    return mse_total

"""
Calculates the pmf given the distribution of the sync error and depending on the 
number of points taken on the time instances of the pulse we wish to create.
"""
def pmf_extract(num_points,mu,var_sample,num_err_samples,samples):
    
    error_samples = np.random.normal(0,var_sample,num_err_samples).astype(np.float32)

    pdf = np.zeros(num_points,dtype=np.float32)
    counter_sam = 0
    while counter_sam < num_err_samples:
        dist = np.abs(samples - error_samples[counter_sam])
        ind = np.argmin(dist)
        if ind < np.floor(num_points/2) + np.floor(num_points/mu) and ind > np.floor(num_points/2) - np.floor(num_points/mu):
            pdf[ind] = pdf[ind] + 1
            counter_sam = counter_sam + 1
        else:
            error_samples[counter_sam] = var_sample*np.random.randn(1)
        
    pdf = torch.tensor(pdf/num_err_samples)
    return pdf