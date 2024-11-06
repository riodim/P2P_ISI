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
            y_ISI_real += b * ISI_channels * x_ISI_real * (dnn_out[sample_index])
            y_ISI_imag += b * ISI_channels * x_ISI_imag * (dnn_out[sample_index])
        else:
            y_rec_real = b * channels * x_real * (dnn_out[1750])
            y_rec_imag = b * channels * x_imag * (dnn_out[1750])

    y_ISI_total_real = (y_ISI_real*ISI_contribution + y_rec_real)
    y_ISI_total_imag = (y_ISI_imag*ISI_contribution + y_rec_imag)

    return y_ISI_total_real, y_ISI_total_imag

def process_file(file_path, ISI_symbols_real, ISI_symbols_imag, ISI_contribution):
    # RC, EXP = pulse_gen.RC_pulses_create()

    # roll_off_index = utils.get_index_for_roll_off(config.roll_off)

    # fft_cur_pulse_rc = torch.abs(torch.fft.fft(torch.nn.functional.pad(RC[roll_off_index], (0, 6500))))
    # fft_cur_pulse_exp = torch.abs(torch.fft.fft(torch.nn.functional.pad(EXP[roll_off_index], (0, 6500))))
    
    # # Define the output directory
    # output_dir = "./fft_data"
    # os.makedirs(output_dir, exist_ok=True)

    # Save FFT data as text files
    # fft_rc_path = os.path.join(output_dir, "fft_cur_pulse_rc.txt")
    # fft_exp_path = os.path.join(output_dir, "fft_cur_pulse_exp.txt")

    # Save each FFT tensor to a text file
    # fft_data_np = fft_cur_pulse_exp.cpu().numpy() if isinstance(fft_cur_pulse_exp, torch.Tensor) else fft_cur_pulse_exp
    # np.savetxt(fft_exp_path, fft_data_np, delimiter=',')    
    # print(f"Saved FFT data to: {fft_exp_path}")

    # import pdb; pdb.set_trace()
    # output_folder = f'./test_results/test_results_learn_rate_{config.learning_rate}/roll_off_{config.roll_off}/data/fft/'

    # utils.save_fft_data(output_folder, fft_cur_pulse_rc,'', "RC", config.roll_off)
    # utils.save_fft_data(output_folder,fft_cur_pulse_exp,'', "EXP", config.roll_off)
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

    # dnn_out = torch.tensor(np.array(dnn_out), dtype=torch.float32).to(config.device)
    # # Generate the FFT for RC, EXP, and dnn_out
    # fft_cur_pulse = torch.abs(torch.fft.fft(torch.nn.functional.pad(dnn_out, (0, 6500))))
    # utils.save_fft_data(output_folder, fft_cur_pulse, M_loss, "DNN", config.roll_off)

    # plt.figure()
    # plt.plot(fft_cur_pulse_rc[:40].cpu().numpy(), label='RC Pulse')
    # plt.plot(fft_cur_pulse_exp[:40].cpu().numpy(), label='EXP Pulse')
    # plt.plot(fft_cur_pulse[:40].cpu().numpy(), label='DNN Pulse')           
    
    # plt.title('RC, EXP, and DNN Output')
    # plt.legend()
    # if ISI_contribution == 1:
    #     output_folder = f'./test_results/test_results_learn_rate_{config.learning_rate}/roll_off_{config.roll_off}/plots/P = {config.P}/'
    # else:
    #     output_folder = f'./test_results/test_results_learn_rate_{config.learning_rate}/roll_off_{config.roll_off}/plots/without_ISI/'
    # output_folder = ''
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)   
    
    # output_plot_path = os.path.join(output_folder, f"{os.path.basename(file_path).split('.')[0]}_plot.png")
    # plt.savefig(output_plot_path) 
    
    # plt.close()

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

    noise_real = torch.randn(config.num_symbols, device=config.device)
    noise_real = 1/np.sqrt(2)*noise_real
    noise_real = noise_real*np.sqrt(config.x)
    noise_real = noise_real/config.b

    noise_imag = torch.randn(config.num_symbols, device=config.device)
    noise_imag = 1/np.sqrt(2)*noise_imag
    noise_imag = noise_imag*np.sqrt(config.x)
    noise_imag = noise_imag/config.b

    y_total_real = y_signal_real + noise_real
    y_total_imag = y_signal_imag + noise_imag

    # sym_utils.plot_scatter(y_total_real, y_total_imag)
    transmitted_symbols = sym_utils.nearest_qam_symbols(config.x_real + 1j*config.x_imag, config.qam_symbols)
    received_symbols = sym_utils.nearest_qam_symbols(y_total_real + 1j*y_total_imag, config.qam_symbols)

    ber = sym_utils.calculate_ber(config.bit_mapping, transmitted_symbols, received_symbols)
    return ber

def calculate_ber():

    # input_folder = f'./test_results/test_results_learn_rate_{config.learning_rate}/roll_off_{config.roll_off}/data/'
    input_folder = './data/'
    ##python calculate_ber 0 will not include the ISI
    ##python calculate_ber 1 will include the ISI
    ##python calculate_ber will include the ISI

    if len(sys.argv) > 1 and sys.argv[1] == '0':
        ISI_contribution = 0  # Set to 0 if '0' is passed as the argument
        output_folder = f'./test_results/test_results_learn_rate_{config.learning_rate}/roll_off_{config.roll_off}/data/ber/without_ISI/'
    else:
        ISI_contribution = 1  # Default value if no argument or any other value is passed
        output_folder = f'./test_results/test_results_learn_rate_{config.learning_rate}/roll_off_{config.roll_off}/data/ber/with_ISI/'
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    x_real = torch.tensor(config.x_real, dtype=torch.float32).to(config.device)
    x_imag = torch.tensor(config.x_imag, dtype=torch.float32).to(config.device)
    ISI_symbols_real = torch.stack([torch.roll(x_real, shifts=i, dims=0) for i in range(1, config.mu)], dim=1)
    ISI_symbols_imag = torch.stack([torch.roll(x_imag, shifts=i, dims=0) for i in range(1, config.mu)], dim=1)

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
    return ber

def calculate_ber_for_dB_range():
    dB_values = list(range(0, 41, 4))  # dB range from 0 to 40 in steps of 5
    M_values = [16]  # Different M values to iterate over
    input_folder = f'./test_results/roll_off_{config.roll_off}/data/'

    # To store DNN results for each file and M values for comparison later
    dnn_ber_results_all_files = {file_name: {M: [] for M in M_values} for file_name in os.listdir(input_folder) if file_name.endswith(".txt")}

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_folder, file_name)

            for M in M_values:
                
                # x_real = torch.tensor(config.x_real, dtype=torch.float32).to(config.device)
                # x_imag = torch.tensor(config.x_imag, dtype=torch.float32).to(config.device)
                # ISI_symbols_real = torch.stack([torch.roll(x_real, shifts=i, dims=0) for i in range(1, config.mu)], dim=1)
                # ISI_symbols_imag = torch.stack([torch.roll(x_imag, shifts=i, dims=0) for i in range(1, config.mu)], dim=1)

                ISI_symbols_real = np.roll(config.x_real, shift=1)
                ISI_symbols_imag = np.roll(config.x_imag, shift=1)
                # roll_off_index = utils.get_index_for_roll_off(config.roll_off)

                dnn_ber_results = []
                # rc_ber_results = []
                # exp_ber_results = []

                for dB in dB_values:
                    x = 10 ** (dB / 10)
                    config.x = 1 / x
                    print(f"Calculating BER for {file_name} at {dB} dB with M = {M}...")
    
                    # Calculate BER for DNN
                    dnn_ber = process_file(file_path, ISI_symbols_real, ISI_symbols_imag, ISI_contribution=1)
                    dnn_ber_results.append(dnn_ber)

                    # RC, EXP = pulse_gen.RC_pulses_create()
                    # mid_sample = np.floor(len(RC[roll_off_index]) / 2).astype(int)
                    # T = np.floor(np.floor(len(RC[roll_off_index]) / config.mu) / 2).astype(int)

                    # y_signal_real_rc, y_signal_imag_rc = MSE_sampling_ISI(
                    #     x_ISI_real=ISI_symbols_real,
                    #     x_ISI_imag=ISI_symbols_imag,
                    #     sample_time=mid_sample,
                    #     T=T,
                    #     dnn_out=RC[roll_off_index],
                    #     ISI_contribution=1
                    # )

                    # noise_real = torch.randn(config.num_symbols, device=config.device) / np.sqrt(2)
                    # noise_real = noise_real * np.sqrt(config.x) / config.b

                    # noise_imag = torch.randn(config.num_symbols, device=config.device) / np.sqrt(2)
                    # noise_imag = noise_imag * np.sqrt(config.x) / config.b

                    # y_total_real_rc = y_signal_real_rc + noise_real
                    # y_total_imag_rc = y_signal_imag_rc + noise_imag

                    # transmitted_symbols = sym_utils.nearest_qam_symbols(config.x_real + 1j * config.x_imag, config.qam_symbols)
                    # received_symbols_rc = sym_utils.nearest_qam_symbols(y_total_real_rc + 1j * y_total_imag_rc, config.qam_symbols)

                    # rc_ber = sym_utils.calculate_ber(config.bit_mapping, transmitted_symbols, received_symbols_rc)
                    # rc_ber_results.append(rc_ber)

                    # Calculate BER for EXP pulse
                    # y_signal_real_exp, y_signal_imag_exp = MSE_sampling_ISI(
                    #     x_ISI_real=ISI_symbols_real,
                    #     x_ISI_imag=ISI_symbols_imag,
                    #     sample_time=mid_sample,
                    #     T=T,
                    #     dnn_out=EXP[roll_off_index],
                    #     ISI_contribution=1
                    # )

                    # y_total_real_exp = y_signal_real_exp + noise_real
                    # y_total_imag_exp = y_signal_imag_exp + noise_imag

                    # received_symbols_exp = sym_utils.nearest_qam_symbols(y_total_real_exp + 1j * y_total_imag_exp, config.qam_symbols)
                    # exp_ber = sym_utils.calculate_ber(config.bit_mapping, transmitted_symbols, received_symbols_exp)
                    # exp_ber_results.append(exp_ber)

                # Store DNN BER results for comparison across M values for the same file
                dnn_ber_results_all_files[file_name][M] = dnn_ber_results

                # Plotting BER vs dB for DNN, RC, and EXP for the current file and M value
                plt.figure()
                plt.plot(dB_values, dnn_ber_results, marker='o', label='DNN')
                # plt.plot(dB_values, rc_ber_results, marker='o', label='RC')
                # plt.plot(dB_values, exp_ber_results, marker='o', label='EXP')

                plt.title(f'BER vs dB for {file_name} with M = {M} (DNN)')
                plt.xlabel('dB')
                plt.ylabel('BER')
                plt.legend()
                plt.grid(True)

                # Save the plot for DNN, RC, and EXP for the current file and M value
                plot_output_folder = './results/plots/different_M/'
                if not os.path.exists(plot_output_folder):
                    os.makedirs(plot_output_folder)

                # plt.savefig(os.path.join(plot_output_folder, f'BER_vs_dB_{file_name}_M_{M}.png'))
                # plt.show()

                # Optionally, save the BER values for the current file to a text file
                with open(os.path.join(plot_output_folder, f'BER_results_{file_name}_M_{M}.txt'), 'w') as f:
                    for dB, dnn_ber in zip(dB_values, dnn_ber_results):
                        f.write(f"{dB} dB -> DNN BER: {dnn_ber}\n")

    # Create a comparison plot for all three M values (16, 64, 256) for each file
    for file_name in dnn_ber_results_all_files:
        # plt.figure()
        # for M in M_values:
        #     plt.plot(dB_values, dnn_ber_results_all_files[file_name][M], marker='o', label=f'DNN M={M}')

        # plt.title(f'Comparison of DNN BER vs dB for {file_name} with different M values')
        # plt.xlabel('dB')
        # plt.ylabel('BER')
        # plt.legend()
        # plt.grid(True)

        # Save the comparison plot for DNN with different M values for the current file
        comparison_plot_folder = './results/plots/comparison/'
        if not os.path.exists(comparison_plot_folder):
            os.makedirs(comparison_plot_folder)

        # plt.savefig(os.path.join(comparison_plot_folder, f'DNN_BER_comparison_{file_name}.png'))
        # plt.show()

if __name__ == "__main__":
    calculate_ber_for_dB_range()