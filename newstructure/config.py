import utils as u
import symbol_utils as sym_u
import numpy as np
import torch

num_symbols = 10000
num_points = 3501
mu = 7
P = 10
batch_size = 100
learning_rate = 0.001
num_epochs = 3
M = 64
roll_off = 0.1
input_size=4
pul_power = 0.1382
freq_resp = 0.2

h = u.generate_h(num_symbols)
b = u.calculate_b(h_data=h, P=P, simRuns=num_symbols)
qam_symbols = sym_u.generate_qam_symbols(num_symbols, M)
bit_mapping = sym_u.assign_bits_to_symbols(qam_symbols, M)

x_real = np.real(qam_symbols)
x_imag = np.imag(qam_symbols)

M_loss_values = [4]
M_sym_values = [4.3*10**3]
M_power_values = [9*10**3]
M_bandwidth_values = [10**1.35]

hidden_layers = [256, 256, 256]

sigma_error = np.array([0.1, 0.2])

prob = u.prepare_probability_distribution(sigma_error, num_points, mu)

device = u.prepare_device()

ISI_channels = torch.tensor(u.generate_h(num_points=batch_size * (mu - 1)), dtype=torch.float32).view(batch_size, mu - 1)

loss_save_interval=1