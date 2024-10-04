import time
import torch
import numpy as np
import utils as utils
import model_loss as model_loss
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

def MSE_sampling_ISI(mu, b, x_real, x_imag, x_ISI_real, x_ISI_imag, channels, ISI_channels, sample_time, T, dnn_out, device):
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
            y_ISI_real += b * ISI_channels * x_ISI_real * (dnn_out[sample_index].unsqueeze(0)).repeat(len(x_ISI_real))
            y_ISI_imag += b * ISI_channels * x_ISI_imag * (dnn_out[sample_index].unsqueeze(0)).repeat(len(x_ISI_imag))
        else:
            y_rec_real = b * channels * x_real * (dnn_out[sample_time].unsqueeze(0)).repeat(len(x_real))
            y_rec_imag = b * channels * x_imag * (dnn_out[sample_time].unsqueeze(0)).repeat(len(x_imag))

    y_ISI_total_real = y_ISI_real * 0 + y_rec_real
    y_ISI_total_imag = y_ISI_imag * 0 + y_rec_imag

    return y_ISI_total_real, y_ISI_total_imag

def prepare_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    device = "cpu"
    return device

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

def process_file(file_path, bit_mapping, qam_symbols, h_data, b, x_real, x_imag, ISI_symbols_real, ISI_symbols_imag, ISI_channels, mu, device, P):
    # Load 'b' as tensor
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

    dnn_out = torch.tensor(np.array(dnn_out), dtype=torch.float32).to(device)

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
        device=device
    )

    y_total_real = y_signal_real
    y_total_imag = y_signal_imag

    received_symbols = y_total_real + 1j * y_total_imag
    received_symbols = utils.nearest_qam_symbols(received_symbols, qam_symbols)

    ber = calculate_ber(bit_mapping, qam_symbols[:len(received_symbols)], received_symbols)
    return ber

def main():
    num_symbols = 10000  # Total number of symbols
    M = 64  # QAM order
    P = 10  # Some parameter related to h_data (as used in your utils)
    mu = 7
    device = prepare_device()  # Make sure device is initialized (CPU or GPU)

    # Generate QAM symbols and bit mappings
    qam_symbols = utils.generate_qam_symbols(num_symbols, M)
    bit_mapping = utils.assign_bits_to_symbols(qam_symbols, M)

    # Real and Imaginary parts of the QAM symbols
    x_real = np.real(qam_symbols)
    x_imag = np.imag(qam_symbols)

    # Generate channel data (h_data) and ISI symbols
    h_data = utils.generate_h(num_symbols)
    b = utils.calculate_b(h_data=h_data, P=P, simRuns=num_symbols)

    # ISI symbol real and imaginary parts (adjust based on ISI length)
    ISI_symbols_real = np.random.choice(x_real, size=(num_symbols,))
    ISI_symbols_imag = np.random.choice(x_imag, size=(num_symbols,))

    # Generate ISI channels for the whole data (direct processing, not batches)
    ISI_channels = torch.tensor(utils.generate_h(num_points=num_symbols * (mu - 1)), dtype=torch.float32).to(device)

    # Process all files in the folder ./results/data/
    input_folder = './results/data/'
    output_folder = './results/data/ber/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_name.endswith(".txt"):
            # Calculate BER for each file
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
                P=P
            )

            # Save the BER result to the output folder
            output_file_path = os.path.join(output_folder, file_name)
            with open(output_file_path, 'w') as output_file:
                output_file.write(f"BER: {ber}\n")

    print("BER calculations completed and saved to ./results/data/ber/")
    return 0


if __name__ == "__main__":
    main()
