# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def generate_qam_symbols(num_symbols, M):
    # Calculate the number of points along each axis (sqrt_M should be an integer)
    sqrt_M = int(np.sqrt(M))
    
    # Generate all possible psi and zeta values symmetrically around zero
    psi = np.arange(-(sqrt_M - 1), sqrt_M, 2)
    zeta = np.arange(-(sqrt_M - 1), sqrt_M, 2)
    
    # Create a grid of the combinations
    Re_ck_j_grid, Im_ck_j_grid = np.meshgrid(psi, zeta)
    
    # Calculate Delta for normalization
    Delta = np.sqrt(2 * (M - 1) / 3)
    
    # Compute the real and imaginary parts according to the provided formulas
    Re_ck_j = Re_ck_j_grid / Delta
    Im_ck_j = Im_ck_j_grid / Delta
    
    # Combine real and imaginary parts to form complex symbols
    symbols = Re_ck_j + 1j * Im_ck_j
    
    # Flatten the symbols grid to get M symbols
    symbols = symbols.flatten()
    
    # Randomly choose num_symbols from the generated symbols
    chosen_indices = np.random.choice(len(symbols), num_symbols, replace=True)
    symbols = symbols[chosen_indices]
    
    # Normalize symbols to ensure average power is 1
    normalization_factor = np.sqrt(np.mean(np.abs(symbols) ** 2))
    symbols /= normalization_factor
    
    return symbols

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

def gray_code(n):
    """Generate n-bit Gray code."""
    return n ^ (n >> 1)

def assign_bits_to_symbols(symbols, M):
    # Create a Gray-coded bit mapping
    # M = 64, so we will need 6 bits for each symbol (3 for I and 3 for Q)
    bit_mapping = {}
    sqrt_M = int(np.sqrt(M))
    num_bits = int(np.log2(sqrt_M))  # Bits for each axis (I and Q)

    # Find the unique values for real and imaginary parts
    real_vals = np.unique([np.real(sym) for sym in symbols])
    imag_vals = np.unique([np.imag(sym) for sym in symbols])

    for symbol in symbols:
        real_part = np.real(symbol)
        imag_part = np.imag(symbol)

        # Find the index of the real and imaginary parts
        real_index = np.where(real_vals == real_part)[0][0]
        imag_index = np.where(imag_vals == imag_part)[0][0]

        # Get the Gray code for the real and imaginary parts
        real_gray = gray_code(real_index)
        imag_gray = gray_code(imag_index)

        # Convert the Gray codes to binary strings
        real_bits = np.binary_repr(real_gray, width=num_bits)
        imag_bits = np.binary_repr(imag_gray, width=num_bits)

        # Concatenate the bits for the symbol
        bit_mapping[symbol] = real_bits + imag_bits

    return bit_mapping

def calculate_bit_distance(correct_symbol, received_symbol, bit_mapping):
    # Find the closest symbol for both correct and received symbols
    correct_symbol_closest = find_closest_symbol(correct_symbol, bit_mapping)
    received_symbol_closest = find_closest_symbol(received_symbol, bit_mapping)

    # Retrieve the corresponding bit strings from the bit mapping
    correct_bits = bit_mapping[correct_symbol_closest]
    error_bits = bit_mapping[received_symbol_closest]

    # Calculate Hamming distance between the bit strings
    distance = sum(c1 != c2 for c1, c2 in zip(correct_bits, error_bits))
    return distance

def find_closest_symbol(received_symbol, bit_mapping):
    # Find the closest symbol in the bit_mapping using the minimum Euclidean distance
    closest_symbol = min(bit_mapping.keys(), key=lambda x: abs(x - received_symbol))
    return closest_symbol

def nearest_qam_symbols(received_symbols, qam_symbols):
    # Ensure received_symbols is a NumPy array for compatibility with qam_symbols
    if isinstance(received_symbols, torch.Tensor):
        received_symbols = received_symbols.cpu().numpy()  # Convert tensor to NumPy

    nearest_symbols = []
    for symbol in received_symbols:
        distances = np.abs(qam_symbols - symbol)
        nearest_symbol = qam_symbols[np.argmin(distances)]
        nearest_symbols.append(nearest_symbol)

    return nearest_symbols

def plot_scatter(x_axis, y_axis):
    plt.figure(figsize=(6, 6))
    plt.scatter(x_axis, y_axis, color='blue', s=10)
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    import pdb; pdb.set_trace()

def calculate_ber(bit_mapping, transmitted_symbols, received_symbols):
    total_bit_errors = 0
    total_bits = 0
    for transmitted_symbol, received_symbol in zip(transmitted_symbols, received_symbols):
        transmitted_bits = bit_mapping[transmitted_symbol]
        received_bits = bit_mapping[received_symbol]
        
        # Calculate the number of bit errors
        bit_errors = sum(t_bit != r_bit for t_bit, r_bit in zip(transmitted_bits, received_bits))
        total_bit_errors += bit_errors
        total_bits += len(transmitted_bits)
    
    ber = total_bit_errors / total_bits
    return ber

def prepare_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    device = "cpu"
    return device