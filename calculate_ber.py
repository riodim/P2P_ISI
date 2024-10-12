import time
import torch
import numpy as np
import utils as utils
import model_loss as model_loss
import torch.optim as optim
import pulse_generation as pulse_gen
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import os
import symbol_utils as sym_utils

num_symbols = 10000  # Total number of symbols
M = 64 
P = 10
mu = 7

def MSE_sampling_ISI(mu, b, x_real, x_imag, x_ISI_real, x_ISI_imag, channels, ISI_channels, sample_time, T, dnn_out, device, ISI_contribution=1):
    num_ISI = np.floor(mu / 2).astype(int)

    # Handle all data directly, no batch dimension needed
    y_ISI_real = torch.zeros_like(x_real).to(device)
    y_ISI_imag = torch.zeros_like(x_imag).to(device)
    y_rec_real = torch.zeros_like(x_real).to(device)
    y_rec_imag = torch.zeros_like(x_imag).to(device)
    
    # Downsample ISI_channels or adjust based on the expected length
    ISI_channels = ISI_channels[:len(x_ISI_real)]
    # No batch dimension, so we process the data directly
    for w2 in range(mu):
        if w2 != num_ISI:
            sample_index = sample_time + (w2 - num_ISI) * 2 * T
            y_ISI_real += b * ISI_channels * x_ISI_real * (dnn_out[sample_index].unsqueeze(0))
            y_ISI_imag += b * ISI_channels * x_ISI_imag * (dnn_out[sample_index].unsqueeze(0))
        else:
            y_rec_real = b * channels * x_real * (dnn_out[1750].unsqueeze(0))
            y_rec_imag = b * channels * x_imag * (dnn_out[1750].unsqueeze(0))

    y_ISI_total_real = (y_ISI_real*ISI_contribution + y_rec_real)
    y_ISI_total_imag = (y_ISI_imag*ISI_contribution + y_rec_imag)

    return y_ISI_total_real, y_ISI_total_imag

def process_file(file_path, bit_mapping, qam_symbols, h_data, b, x_real, x_imag, ISI_symbols_real, ISI_symbols_imag, ISI_channels, mu, device, ISI_contribution = 1):

    if isinstance(b, np.ndarray):
        b = torch.tensor(b, dtype=torch.float32, device=device)

    # Convert h_data, x_real, x_imag to tensors
    x_real = torch.tensor(x_real, dtype=torch.float32).to(device)
    x_imag = torch.tensor(x_imag, dtype=torch.float32).to(device)
    h_data = torch.tensor(h_data, dtype=torch.float32).to(device)

    # Load data from file and convert to tensor
    with open(file_path, 'r') as file:
        dnn_out = []
        for line in file:
            try:
                dnn_out.append(float(line.strip()))
            except ValueError:
                continue  # Skip metadata lines

    RC, EXP = pulse_gen.RC_pulses_create()
    dnn_out = torch.tensor(np.array(dnn_out), dtype=torch.float32).to(device)

    cur_pulse = torch.nn.functional.pad(dnn_out, (0, 6500))
    fft_cur_pulse = torch.abs(torch.fft.fft(cur_pulse))
    
    cur_pulse_rc = torch.nn.functional.pad(RC[0], (0, 6500))
    fft_cur_pulse_rc = torch.abs(torch.fft.fft(cur_pulse_rc))
    
    cur_pulse_exp = torch.nn.functional.pad(EXP[0], (0, 6500))
    fft_cur_pulse_exp = torch.abs(torch.fft.fft(cur_pulse_exp))
       
    plt.figure()
    plt.plot(fft_cur_pulse[0:40])
    plt.plot(fft_cur_pulse_rc[0:40])
    plt.plot(fft_cur_pulse_exp[0:40])
    # import pdb; pdb.set_trace()
    mid_sample = np.floor(len(dnn_out) / 2).astype(int)
    T = np.floor(np.floor(len(dnn_out) / mu) / 2).astype(int)

    y_signal_real, y_signal_imag = MSE_sampling_ISI(
        mu=mu,
        b=b,
        x_real=x_real,
        x_imag=x_imag,
        x_ISI_real=ISI_symbols_real,
        x_ISI_imag=ISI_symbols_imag,
        channels=h_data,
        ISI_channels=ISI_channels,
        sample_time=mid_sample,
        T=T,
        dnn_out=dnn_out,
        device=device,
        ISI_contribution=ISI_contribution
    )
    # M = 64
    # delta = np.sqrt(2 * (M - 1) / 3)
    
    noise_real = torch.randn(num_symbols, device=device)
    noise_real = 1/np.sqrt(2)*noise_real
    noise_real = noise_real*np.sqrt(0.01)
    noise_real = noise_real/b

    noise_imag = torch.randn(num_symbols, device=device)
    noise_imag = 1/np.sqrt(2)*noise_imag
    noise_imag = noise_imag*np.sqrt(0.01)
    noise_imag = noise_imag/b

    y_total_real = y_signal_real + noise_real
    y_total_imag = y_signal_imag + noise_imag

    # sym_utils.plot_scatter(y_total_real, y_total_imag)

    transmitted_symbols = sym_utils.nearest_qam_symbols(x_real + 1j*x_imag, qam_symbols)
    received_symbols = sym_utils.nearest_qam_symbols(y_total_real + 1j*y_total_imag, qam_symbols)

    ber = sym_utils.calculate_ber(bit_mapping, transmitted_symbols, received_symbols)
    return ber

def main():

    input_folder = './results/data/'

    ##python calculate_ber 0 will not include the ISI
    ##python calculate_ber 1 will include the ISI
    ##python calculate_ber will include the ISI

    if len(sys.argv) > 1 and sys.argv[1] == '0':
        ISI_contribution = 0  # Set to 0 if '0' is passed as the argument
        output_folder = './results/data/ber/without_ISI/'
    else:
        ISI_contribution = 1  # Default value if no argument or any other value is passed
        output_folder = './results/data/ber/with_ISI/'

    device = utils.prepare_device()

    # Generate QAM symbols and bit mappings
    qam_symbols = sym_utils.generate_qam_symbols(num_symbols, M)
    bit_mapping = sym_utils.assign_bits_to_symbols(qam_symbols, M)

    # Real and Imaginary parts of the QAM symbols
    x_real = np.real(qam_symbols)
    x_imag = np.imag(qam_symbols)

    # Generate channel data and ISI symbols
    h_data = utils.generate_h(num_symbols)
    b = utils.calculate_b(h_data=h_data, P=P, simRuns=num_symbols)
 
    ISI_symbols_real = np.roll(x_real, shift=1)  # Shift the real parts by one to simulate ISI from neighboring symbols
    ISI_symbols_imag = np.roll(x_imag, shift=1)  # Shift the imaginary parts similarly

    # Generate ISI channels for the whole data (direct processing, not batches)
    ISI_channels = torch.tensor(utils.generate_h(num_points=num_symbols * (mu - 1)), dtype=torch.float32).to(device)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    ber_results = []  

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_name.endswith(".txt"):
            ber = process_file(
                file_path=file_path,
                bit_mapping=bit_mapping,
                qam_symbols=qam_symbols,
                h_data=h_data,
                b=b,
                x_real=x_real,
                x_imag=x_imag,
                ISI_symbols_real=ISI_symbols_real,
                ISI_symbols_imag=ISI_symbols_imag,
                ISI_channels=ISI_channels,
                mu=mu,
                device=device,
                ISI_contribution=ISI_contribution
            )

            output_file_path = os.path.join(output_folder, file_name)
            ber_results.append(ber)
            with open(output_file_path, 'w') as output_file:
                output_file.write(f"BER: {ber}\n")
    
    if ber_results:
        average_ber = np.mean(ber_results)
        print(f"Average BER for the run: {average_ber}")

    print("BER calculations completed and saved to: ", output_folder)
    return 0

if __name__ == "__main__":
    main()
