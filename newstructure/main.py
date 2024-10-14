import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import product

import utils as u
import config as config
import model.model_loss as model_loss
import model.model_train as model_train
import model.model_test as model_test
import calculate_ber as ber 

def prepare_dataloader():
    inputs = np.vstack((config.b, config.h, config.x_real, config.x_imag)).T
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(config.device)
    dataset = TensorDataset(inputs_tensor)

    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

def prepare_model():
    model = model_loss.OTAPulse(config.input_size, config.num_points, config.hidden_layers).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    return model, optimizer, scheduler

def main():    
    results_folder = 'test_results'
    roll_off_folder = os.path.join(results_folder, f'roll_off_{config.roll_off}')
    data_folder = os.path.join(roll_off_folder, 'data')
    plots_folder = os.path.join(roll_off_folder, 'plots')
    ##example path ./test_results/roll_off_0.1/data

    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    for M_loss, M_sym, M_power, M_bandwidth in product(config.M_loss_values, config.M_sym_values, config.M_power_values, config.M_bandwidth_values):
        
        print(f"Running with M_loss={M_loss}, M_sym={M_sym}, M_power={M_power}, M_bandwidth={M_bandwidth}")
        
        dataloader = prepare_dataloader()
        model, optimizer, scheduler = prepare_model()
        loss_function = model_loss.ISI_loss_function()
                
        model_train.train(
            model,
            dataloader,
            optimizer,
            loss_function,
            M_loss,
            M_sym,
            M_power,
            M_bandwidth,
            scheduler
        )
        
        average_loss, pulse_per_batch = model_test.test(
            model,
            dataloader,
            loss_function,
            M_loss,
        )    

        created_pulse = pulse_per_batch
        mid_point = 1750
        data_pulse_last = (created_pulse[0:mid_point] + created_pulse.flip(0)[0:mid_point]) / 2
        pulse_new = created_pulse 
        pulse_new[0:mid_point] = data_pulse_last
        pulse_new[mid_point+1:config.num_points] = data_pulse_last.flip(0)
        
        filename = f"pulse_Mloss{M_loss}_Msym_{M_sym}_Mpower_{M_power}_Mbandwidth_{M_bandwidth}_Loss_{average_loss}.txt"
        u.save_results_to_file(filename, M_loss, M_sym, M_power, M_bandwidth, pulse_new, data_folder)
                            
    u.process_all_files_in_folder(data_folder, plots_folder)
    ber.calculate_ber()

if __name__ == "__main__":
    roll_off_values = [0.1]
    for roll_off in roll_off_values:
        print(f"Running training for roll_off={roll_off}")
        
        config.roll_off = roll_off
        
        main()