#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import torch
import torch.nn as nn
import numpy as np
import utils as utils


#OTA Pulse Model
class OTAPulse(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(OTAPulse, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(hidden_layers[2], output_size)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):

        x1 = self.fc1(x)
        x2 = self.relu1(x1)

        x3 = self.fc2(x2)
        x4 = self.relu2(x3)

        x5 = self.fc3(x4)
        x6 = self.relu3(x5)

        x7 = self.fc4(x6)

        return x7


class ISI_loss_function(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(ISI_loss_function, self).__init__()
        self.epsilon = epsilon
        
    def forward(self,
                batch_size,
                inputs,  
                mu,
                x_real,
                x_imag,
                ISI_symbols_real,
                ISI_symbols_imag,
                ISI_channels,
                num_points,
                noise,
                prob,
                device,
                M_sym,
                M_power,
                M_bandwidth,
                pul_power,
                freq_resp,
                P,
                test_bool=False,
                ):

        # Data related attributes from inputs 
        x_real = x_real.view(batch_size, 1)
        x_imag = x_imag.view(batch_size, 1)

        # Extract h from inputs and ensure it's positive
        h = inputs[:, 0:1]

        # Compute b and clamp its values to be within the valid range
        b = torch.where(1/h < torch.sqrt(torch.tensor(P, device=device)), 1/h, torch.sqrt(torch.tensor(P, device=device)))
        
        loss = torch.tensor(0.0, device=device, requires_grad=True) 
         
        mid_sample = np.floor(num_points / 2).astype(int)
        T = np.floor(np.floor(num_points / mu) / 2).astype(int)
                
        cur_pulse = torch.nn.functional.pad(inputs, (0, 6500, 0, 0))

        fft_cur_pulse = torch.abs(torch.fft.fft(cur_pulse, dim=1))

        for dist in range(-T, T):

            # Calls function to compute SE at sampling time and ISI for every error sample
            sample_time = mid_sample + dist

            y_signal_real, y_signal_imag = utils.MSE_sampling_ISI(mu, b, x_real, x_imag, ISI_symbols_real, ISI_symbols_imag, h, ISI_channels, sample_time, T, inputs, batch_size, device)
            y_target_real = torch.sum(x_real, dim=1)
            y_target_imag = torch.sum(x_imag, dim=1)
            y_total_real = y_signal_real + noise
            y_total_imag = y_signal_imag + noise

            # Adjust indexing for prob to avoid out-of-bounds errors
            prob_index = sample_time % prob.size(0)  # Adjusting sample_time to fit within prob's size

            # Calculate the loss using the adjusted prob index
            loss = loss + prob[prob_index] * (torch.mean(torch.square(y_total_real - y_target_real)) + torch.mean(torch.square(y_total_imag - y_target_imag)))
            
            if not test_bool:
                loss = 5 * loss \
                    + self.calculate_loss_sym(M_sym, mid_sample, inputs) \
                    + self.calculate_loss_power(M_power, num_points, pul_power, inputs) \
                    + self.calculate_loss_bandwidth(M_bandwidth, fft_cur_pulse, freq_resp)
                
        return loss
            
    def calculate_loss_sym(self, M_sym, mid_sample, inputs):
        loss_sym = M_sym * torch.mean(torch.square(inputs[:, 0:mid_sample] - inputs.flip(1)[:, 0:mid_sample]))
        return loss_sym

    def calculate_loss_power(self, M_power, num_points, pul_power, inputs):
        loss_power = M_power * torch.mean(torch.square((torch.sum(torch.square(inputs), dim=1) / num_points) - pul_power))
        return loss_power

    def calculate_loss_bandwidth(self, M_bandwidth, fft_cur_pulse, freq_resp):
        loss_bandwidth = M_bandwidth * torch.mean(torch.square(fft_cur_pulse[:, 12 :5001] - freq_resp))
        return loss_bandwidth