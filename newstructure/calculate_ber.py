import time
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

import utils as utils
import pulse_generation as pulse_gen
import symbol_utils as sym_utils
import config 


def MSE_sampling_ISI(x_ISI_real, x_ISI_imag, sample_time, T, dnn_out, ISI_contribution=1):
    num_ISI = np.floor(config.mu / 2).astype(int)
    x_real = torch.tensor(config.x_real, dtype=torch.float32).to(config.device)
    x_imag = torch.tensor(config.x_imag, dtype=torch.float32).to(config.device)
    channels = torch.tensor(config.h, dtype=torch.float32).to(config.device)

    y_ISI_real = torch.zeros_like(x_real).to(config.device)
    y_ISI_imag = torch.zeros_like(x_imag).to(config.device)
    y_rec_real = torch.zeros_like(x_real).to(config.device)
    y_rec_imag = torch.zeros_like(x_imag).to(config.device)
    
    b = torch.tensor(config.b, dtype=torch.float32, device=config.device)
    
    ISI_channels = torch.tensor(utils.generate_h(num_points=config.num_symbols * (config.mu - 1)), dtype=torch.float32).to(config.device)
    ISI_channels = ISI_channels[:len(x_ISI_real)]

    for w2 in range(config.mu):
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

def process_file(file_path, ISI_symbols_real, ISI_symbols_imag, ISI_contribution):
    RC, EXP = pulse_gen.RC_pulses_create()

    roll_off_index = utils.get_index_for_roll_off(config.roll_off)

    fft_cur_pulse_rc = torch.abs(torch.fft.fft(torch.nn.functional.pad(RC[roll_off_index], (0, 6500))))
    fft_cur_pulse_exp = torch.abs(torch.fft.fft(torch.nn.functional.pad(EXP[roll_off_index], (0, 6500))))
    
    output_folder = f'./test_results/test_results_learn_rate_{config.learning_rate}/roll_off_{config.roll_off}/data/fft/'

    utils.save_fft_data(output_folder, fft_cur_pulse_rc,'', "RC", config.roll_off)
    utils.save_fft_data(output_folder,fft_cur_pulse_exp,'', "EXP", config.roll_off)
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        dnn_out = []
        
        # Extract M_loss from the first line
        if first_line.startswith("M_loss"):
            M_loss_str = first_line.split(",")[0].split(":")[1].strip()
            M_loss = float(M_loss_str)
        else:
            M_loss = None  # Handle the case if the format is unexpected

        # Process the remaining lines to get dnn_out
        for line in file:
            try:
                dnn_out.append(float(line.strip()))
            except ValueError:
                continue  # Skip metadata lines

    dnn_out = torch.tensor(np.array(dnn_out), dtype=torch.float32).to(config.device)
    # Generate the FFT for RC, EXP, and dnn_out
    fft_cur_pulse = torch.abs(torch.fft.fft(torch.nn.functional.pad(dnn_out, (0, 6500))))
    utils.save_fft_data(output_folder, fft_cur_pulse, M_loss, "DNN", config.roll_off)

    plt.figure()
    plt.plot(fft_cur_pulse_rc[:40].cpu().numpy(), label='RC Pulse')
    plt.plot(fft_cur_pulse_exp[:40].cpu().numpy(), label='EXP Pulse')
    plt.plot(fft_cur_pulse[:40].cpu().numpy(), label='DNN Pulse')           
    
    plt.title(f'RC, EXP, and DNN Output (roll_off = {config.roll_off})')
    plt.xlabel('DFT Sample')
    plt.ylabel('Magnitude (FFT)')
    plt.legend()
    if ISI_contribution == 1:
        output_folder = f'./test_results/test_results_learn_rate_{config.learning_rate}/roll_off_{config.roll_off}/plots/P = {config.P}/'
    else:
        output_folder = f'./test_results/test_results_learn_rate_{config.learning_rate}/roll_off_{config.roll_off}/plots/without_ISI/'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)   
    
    output_plot_path = os.path.join(output_folder, f"{os.path.basename(file_path).split('.')[0]}_plot.png")
    plt.savefig(output_plot_path) 
    
    plt.close()

    mid_sample = np.floor(len(dnn_out) / 2).astype(int)
    T = np.floor(np.floor(len(dnn_out) / config.mu) / 2).astype(int)

    y_signal_real, y_signal_imag = MSE_sampling_ISI(
        x_ISI_real=ISI_symbols_real,
        x_ISI_imag=ISI_symbols_imag,
        sample_time=mid_sample,
        T=T,
        dnn_out=dnn_out,
        ISI_contribution=ISI_contribution
    )
    # M = 64
    # delta = np.sqrt(2 * (M - 1) / 3)

    noise_real = torch.randn(config.num_symbols, device=config.device)
    noise_real = 1/np.sqrt(2)*noise_real
    noise_real = noise_real*np.sqrt(0.01)
    noise_real = noise_real/config.b

    noise_imag = torch.randn(config.num_symbols, device=config.device)
    noise_imag = 1/np.sqrt(2)*noise_imag
    noise_imag = noise_imag*np.sqrt(0.01)
    noise_imag = noise_imag/config.b

    y_total_real = y_signal_real + noise_real
    y_total_imag = y_signal_imag + noise_imag

    # sym_utils.plot_scatter(y_total_real, y_total_imag)
    transmitted_symbols = sym_utils.nearest_qam_symbols(config.x_real + 1j*config.x_imag, config.qam_symbols)
    received_symbols = sym_utils.nearest_qam_symbols(y_total_real + 1j*y_total_imag, config.qam_symbols)

    ber = sym_utils.calculate_ber(config.bit_mapping, transmitted_symbols, received_symbols)
    return ber

def calculate_ber():

    input_folder = f'./test_results/test_results_learn_rate_{config.learning_rate}/roll_off_{config.roll_off}/data/'

    ##python calculate_ber 0 will not include the ISI
    ##python calculate_ber 1 will include the ISI
    ##python calculate_ber will include the ISI

    if len(sys.argv) > 1 and sys.argv[1] == '0':
        ISI_contribution = 0  # Set to 0 if '0' is passed as the argument
        output_folder = f'./test_results/test_results_learn_rate_{config.learning_rate}/roll_off_{config.roll_off}/data/ber/without_ISI/'
    else:
        ISI_contribution = 1  # Default value if no argument or any other value is passed
        output_folder = f'./test_results/test_results_learn_rate_{config.learning_rate}/roll_off_{config.roll_off}/data/ber/with_ISI/'
 
    ISI_symbols_real = np.roll(config.x_real, shift=1)  # Shift the real parts by one to simulate ISI from neighboring symbols
    ISI_symbols_imag = np.roll(config.x_imag, shift=1)  # Shift the imaginary parts similarly

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    ber_results = []  

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_name.endswith(".txt"):
            ber = process_file(
                file_path=file_path,
                ISI_symbols_real=ISI_symbols_real,
                ISI_symbols_imag=ISI_symbols_imag,
                ISI_contribution=ISI_contribution
            )

            output_file_path = os.path.join(output_folder, file_name)
            ber_results.append(ber)
            with open(output_file_path, 'w') as output_file:
                output_file.write(f"BER: {ber}\n")
    
    print("BER calculations completed and saved to: ", output_folder)       
    return 0

if __name__ == "__main__":
    calculate_ber()
