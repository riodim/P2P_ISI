import time
import torch
import numpy as np
import utils as utils
import symbol_utils as sym_utils
import model_loss as model_loss
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from itertools import product
import os

results_folder = 'results'
data_folder = os.path.join(results_folder, 'data')
plots_folder = os.path.join(results_folder, 'plots')

num_points = 3501
mu = 7
P = 10
batch_size = 100
learning_rate = 0.001 #0.003
num_epochs = 3
freq_resp = 0.2
pul_power = 0.1382
num_symbols = 10000

def prepare_dataloader(num_symbols, M, P, batch_size, device):
    qam_symbols = sym_utils.generate_qam_symbols(num_symbols, M)

    # utils.plot_constellation(qam_symbols, "qam_constellation.png")

    h = utils.generate_h(num_symbols)
    b = utils.calculate_b(h_data=h, P=P, simRuns=num_symbols)
    x_real = np.real(qam_symbols)
    x_imag = np.imag(qam_symbols)

    inputs = np.vstack((b, h, x_real, x_imag)).T
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)

    dataset = TensorDataset(inputs_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def prepare_model(input_size, output_size, hidden_layers, learning_rate, device):
    model = model_loss.OTAPulse(input_size, output_size, hidden_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    return model, optimizer, scheduler

def train(
    model,
    dataloader,
    optimizer,
    loss_function,
    num_epochs,
    device,
    mu,
    num_points,
    prob,
    M_loss,
    M_sym,
    M_power,
    M_bandwidth,
    pul_power,
    freq_resp,
    batch_size,
    scheduler,
    ISI_channels
):
    
    for epoch in range(num_epochs):
        start_time = time.time()  # Start the timer for the epoch
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            batch = batch[0].to(device)
            optimizer.zero_grad()
            output = model(batch)

            x_real = batch[:, 2]  # Real parts of the symbols
            x_imag = batch[:, 3]  # Imaginary parts of the symbols

            # Apply roll operation to simulate ISI from neighboring symbols
            ISI_symbols_real = torch.stack([torch.roll(x_real, shifts=i, dims=0) for i in range(1, mu)], dim=1)
            ISI_symbols_imag = torch.stack([torch.roll(x_imag, shifts=i, dims=0) for i in range(1, mu)], dim=1)
            
            noise = torch.randn(batch_size, device=device)

            loss = loss_function(
                batch_size=batch_size,
                dnn_out=output,
                mu=mu,
                batch=batch,
                ISI_symbols_real=ISI_symbols_real,
                ISI_symbols_imag=ISI_symbols_imag,
                ISI_channels=ISI_channels,
                num_points=num_points,
                noise=noise,
                prob=prob.to(device),
                device=device,
                M_loss=M_loss,
                M_sym=M_sym,
                M_power=M_power,
                M_bandwidth=M_bandwidth,
                pul_power=pul_power,
                freq_resp=freq_resp,
            )
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            print(f" Loss: {loss.item()}")
            total_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Time: {epoch_time:.2f} seconds")


def test(
    model,
    dataloader,
    loss_function,
    device,
    mu,
    num_points,
    prob,
    M_loss,
    M_sym,
    M_power,
    M_bandwidth,
    pul_power,
    freq_resp,
    batch_size,
    ISI_channels):
    model.eval() 
    
    total_loss = 0

    index = -1
    pulse_per_batch = torch.zeros(len(dataloader), num_points)

    with torch.no_grad():  # Disable gradient calculation for testing
        for batch_idx, batch in enumerate(dataloader):
            index = index + 1

            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            batch = batch[0].to(device)
            output = model(batch)
            pulse_per_batch[index, :] = torch.mean(output,dim=0)

            x_real = batch[:, 2]  # Real parts of the symbols
            x_imag = batch[:, 3]  # Imaginary parts of the symbols

            # Apply roll operation to simulate ISI from neighboring symbols
            ISI_symbols_real = torch.stack([torch.roll(x_real, shifts=i, dims=0) for i in range(1, mu)], dim=1)
            ISI_symbols_imag = torch.stack([torch.roll(x_imag, shifts=i, dims=0) for i in range(1, mu)], dim=1)

            noise = torch.randn(batch_size, device=device)

            loss = loss_function(
                batch_size=batch_size,
                dnn_out=output,
                mu=mu,
                batch=batch,
                ISI_symbols_real=ISI_symbols_real,
                ISI_symbols_imag=ISI_symbols_imag,
                ISI_channels=ISI_channels,
                num_points=num_points,
                noise=noise,
                prob=prob.to(device),
                device=device,
                M_loss=M_loss,
                M_sym=M_sym,
                M_power=M_power,
                M_bandwidth=M_bandwidth,
                pul_power=pul_power,
                freq_resp=freq_resp,
                test_bool=True,
            )

            total_loss += loss.item()

    pulse_per_batch = torch.mean(pulse_per_batch, dim = 0)
    print(f"Test Loss: {total_loss / len(dataloader):.4f}")
    return total_loss / len(dataloader), pulse_per_batch

def main():
    device = utils.prepare_device()
    
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    M_loss_values = [10]
    M_sym_values = [4.3*10**3]
    M_power_values = [9*10**3]
    M_bandwidth_values = [10**1.35]

    for M_loss, M_sym, M_power, M_bandwidth in product(M_loss_values, M_sym_values, M_power_values, M_bandwidth_values):
        print(f"Running with M_loss={M_loss}, M_sym={M_sym}, M_power={M_power}, M_bandwidth={M_bandwidth}")
        dataloader = prepare_dataloader(
            num_symbols=num_symbols,
            M=64,
            P=P,
            batch_size=batch_size,
            device=device,
        )
        model, optimizer, scheduler = prepare_model(
            input_size=4,  # Assuming your input to the model is 4-dimensional
            output_size=num_points,  # The model should output 4-dimensional vectors
            hidden_layers=[256, 256, 256],
            learning_rate=learning_rate,
            device=device,
        )
        loss_function = model_loss.ISI_loss_function()

        # Define probability distribution for error samples
        sigma_error = np.array([0.1, 0.2])
        prob = utils.prepare_probability_distribution(sigma_error, num_points, mu)
        
        ISI_channels = torch.tensor(
            utils.generate_h(num_points=batch_size * (mu - 1)), dtype=torch.float32
        )
        ISI_channels = ISI_channels.view(batch_size, mu - 1)
        
        train(
            model,
            dataloader,
            optimizer,
            loss_function,
            num_epochs,
            device,
            mu,
            num_points,
            prob,
            M_loss,
            M_sym,
            M_power,
            M_bandwidth,
            pul_power,
            freq_resp,
            batch_size,
            scheduler,
            ISI_channels
        )

        average_loss, pulse_per_batch = test(
            model,
            dataloader,
            loss_function,
            device,
            mu,
            num_points,
            prob,
            M_loss,
            M_sym,
            M_power,
            M_bandwidth,
            pul_power,
            freq_resp,
            batch_size,
            ISI_channels
        )    
        created_pulse = pulse_per_batch
        mid_point = 1750
        data_pulse_last = (created_pulse[0:mid_point] + created_pulse.flip(0)[0:mid_point]) / 2
        pulse_new = created_pulse 
        pulse_new[0:mid_point] = data_pulse_last
        pulse_new[mid_point+1:num_points] = data_pulse_last.flip(0)
        
        filename = f"pulse_Mloss{M_loss}_Msym_{M_sym}_Mpower_{M_power}_Mbandwidth_{M_bandwidth}_Loss_{average_loss}.txt"
        utils.save_results_to_file(filename, M_loss, M_sym, M_power, M_bandwidth, pulse_new, data_folder)
                            
    utils.process_all_files_in_folder(data_folder, plots_folder)
                
if __name__ == "__main__":
    main()

