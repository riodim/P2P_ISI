# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def get_value_for_roll_off(roll_off):
    roll_off_table = {
        0.1: 12,
        0.2: 13,
        0.3: 14,
        0.4: 15,
        0.5: 16,
        0.6: 17,
        0.7: 18,
        0.8: 19,
        0.9: 20,
        1.0: 21
    }
    return roll_off_table.get(roll_off, "Invalid roll_off value")

def get_index_for_roll_off(roll_off):
    roll_off_index_table = {
        0.1: 0,
        0.3: 1,
        0.5: 2,
        0.7: 3,
        0.9: 4,
    }
    import pdb; pdb.set_trace()
    return roll_off_index_table.get(roll_off, "Invalid roll_off value")

output_fft_folder = './results/data/fft/'
if not os.path.exists(output_fft_folder):
    os.makedirs(output_fft_folder)

def save_fft_data(fft_data, M_loss, pulse_type, roll_off):

    roll_off_folder = os.path.join(output_fft_folder, f'roll_off_{roll_off}')
    if not os.path.exists(roll_off_folder):
        os.makedirs(roll_off_folder)

    if M_loss:  # Only add M_loss if it's provided
        filename = f"M_loss_{M_loss}_{pulse_type}.txt"
    else:
        filename = f"{pulse_type}.txt"
    
    filepath = os.path.join(roll_off_folder, filename)

    # Construct the file name and path
    fft_data_np = fft_data.cpu().numpy() if isinstance(fft_data, torch.Tensor) else fft_data
    np.savetxt(filepath, fft_data_np, delimiter=',')    
    print(f"Saved FFT data to: {filepath}")

def prepare_probability_distribution(sigma_error, num_points, mu):
    prob = torch.zeros(len(sigma_error), num_points)
    for iterSig in range(len(sigma_error)):
        var_sample = sigma_error[iterSig]
        prob[iterSig] = pmf_extract(
            num_points, mu, var_sample, 10000, np.linspace(-mu / 2, mu / 2, num_points)
        )

    return prob

def MSE_sampling_ISI(mu, b, x_real, x_imag, x_ISI_real, x_ISI_imag, channels, ISI_channels, sample_time, T, dnn_out, batch_size, device):
    num_ISI = np.floor(mu / 2).astype(int)

    y_ISI_real = torch.zeros(batch_size, 1).to(device)
    y_ISI_imag = torch.zeros(batch_size, 1).to(device)
    y_rec_real = torch.zeros(batch_size, 1).to(device)
    y_rec_imag = torch.zeros(batch_size, 1).to(device)

    index = 0
    b = torch.reshape(b,(batch_size,1))
    channels = torch.reshape(channels,(batch_size,1))

    for w2 in range(mu):        
        if w2 != num_ISI:
            
            y_ISI_real = y_ISI_real + b*ISI_channels[:,index]*x_ISI_real[:,index]*(dnn_out[:,sample_time+(w2-num_ISI)*2*T].unsqueeze(1)).repeat(1,1)
            y_ISI_imag = y_ISI_imag + b*ISI_channels[:,index]*x_ISI_imag[:,index]*(dnn_out[:,sample_time+(w2-num_ISI)*2*T].unsqueeze(1)).repeat(1,1)

        else:
            y_rec_real = b*channels*x_real*(dnn_out[:,sample_time].unsqueeze(1))
            y_rec_imag = b*channels*x_imag*(dnn_out[:,sample_time].unsqueeze(1))

    y_ISI_total_real = y_ISI_real + y_rec_real
    y_ISI_total_imag = y_ISI_imag + y_rec_imag
 
    return y_ISI_total_real, y_ISI_total_imag

"""
Calculates the pmf given the distribution of the sync error and depending on the 
number of points taken on the time instances of the pulse we wish to create.
"""
def pmf_extract(num_points, mu, var_sample, num_err_samples, samples):
    error_samples = np.random.normal(0, var_sample, num_err_samples).astype(np.float32)

    pdf = np.zeros(num_points, dtype=np.float32)
    counter_sam = 0

    while counter_sam < num_err_samples:
        dist = np.abs(samples - error_samples[counter_sam])
        ind = np.argmin(dist)

        if ind < np.floor(num_points / 2) + np.floor(num_points / mu) and ind > np.floor(num_points / 2) - np.floor(num_points / mu):
            pdf[ind] = pdf[ind] + 1
            counter_sam = counter_sam + 1
        else:
            error_samples[counter_sam] = var_sample * np.random.randn(1)
        
    pdf = torch.tensor(pdf / num_err_samples)
    
    return pdf

def generate_h(num_points):
    h_data = np.zeros(num_points, dtype=np.float32)  

    for m in range(num_points):
        h = (1/np.sqrt(2)) * (np.random.randn(1) + 1j * np.random.randn(1))
        
        h_data[m] = np.abs(h).item()

    return h_data

def calculate_b(simRuns, P, h_data):
    b_data = np.zeros(simRuns, dtype=np.float32)
    sqrt_P = np.sqrt(P)

    for m in range(simRuns):
        h = h_data[m]

        if 1/h < sqrt_P:
            b = 1/h
        else:
            b = sqrt_P

        b_data[m] = b

    return b_data

def save_results_to_file(filename, M_loss, M_sym, M_power, M_bandwidth, pulse_new, output_folder='results'):
    os.makedirs(output_folder, exist_ok=True)
    
    # Combine folder path with the filename
    filepath = os.path.join(output_folder, filename)

    # Save the results to a text file
    with open(filepath, 'w') as f:
        f.write(f"M_loss: {M_loss}, M_sym: {M_sym}, M_power: {M_power}, M_bandwidth: {M_bandwidth}\n")
        np.savetxt(f, pulse_new.cpu().numpy(), fmt='%.6f')
        f.write("\n\n")

def plot_and_save_pulse(filename, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the file and read the M values and pulse_new array
    with open(filename, 'r') as f:
        first_line = f.readline().strip()  # Read the first line for M values
        pulse_new = np.loadtxt(f)  # Read the pulse_new array from the rest of the file
    
    # Extract the base name of the file (without extension)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Plot the pulse_new array
    plt.figure()
    plt.plot(pulse_new)
    plt.title(base_name)
    plt.xlabel('Index')
    plt.ylabel('Pulse Amplitude')
    
    # Save the plot as a PNG in the results folder
    output_path = os.path.join(output_folder, f"{base_name}.png")
    plt.savefig(output_path)
    plt.close()

def process_all_files_in_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all the files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            plot_and_save_pulse(file_path, output_folder)

def prepare_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    device = "cpu"
    return device