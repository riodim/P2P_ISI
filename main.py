import torch
import numpy as np
import utils as utils
import model_loss as model_loss
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def prepare_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def prepare_dataloader(num_symbols, M, P, batch_size):
    qam_symbols = utils.generate_qam_symbols(num_symbols, M)
    utils.plot_constellation(qam_symbols, "qam_constellation.png")

    x_real = np.real(qam_symbols)
    x_imag = np.imag(qam_symbols)
    h = np.random.randn(num_symbols)
    inputs = np.vstack((np.random.uniform(0, np.sqrt(P), num_symbols), h, x_real, x_imag)).T
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    dataset = TensorDataset(inputs_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def prepare_model(input_size, output_size, hidden_layers, learning_rate, device):
    model = model_loss.OTAPulse(input_size, output_size, hidden_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    return model, optimizer, scheduler

def prepare_loss_function():
    return model_loss.ISI_loss_function()

def prepare_probability_distribution(sigma_error, num_points, mu):
    prob = torch.zeros(len(sigma_error), num_points)
    for iterSig in range(len(sigma_error)):
        var_sample = sigma_error[iterSig]
        prob[iterSig] = utils.pmf_extract(num_points, mu, var_sample, 10**5, np.linspace(-mu/2, mu/2, num_points))

    return prob

def train(model, dataloader, optimizer, loss_function, num_epochs, device, P, mu, num_points, prob, M_sym, M_power, M_bandwidth, pul_power, freq_resp, batch_size, scheduler):    

    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            output = model(batch)

            # Simulate ISI symbols and channels
            ISI_symbols_real = torch.tensor(np.random.uniform(-np.sqrt(3), np.sqrt(3), (batch_size, mu-1)), device=device).float()
            ISI_symbols_imag = torch.tensor(np.random.uniform(-np.sqrt(3), np.sqrt(3), (batch_size, mu-1)), device=device).float()
            ISI_channels = torch.tensor(np.random.normal(0, 1/np.sqrt(2), (mu-1, 1)), device=device).float()
            noise = torch.randn(batch_size, device=device)

            loss = loss_function(batch_size = batch_size, inputs=output, mu=mu, x_real = batch[:, 2], x_imag = batch[:, 3], ISI_symbols_real = ISI_symbols_real, ISI_symbols_imag = ISI_symbols_imag, ISI_channels = ISI_channels, num_points = num_points, noise = noise, prob = prob.to(device), device = device, M_sym = M_sym, M_power = M_power, M_bandwidth = M_bandwidth, pul_power = pul_power, freq_resp = freq_resp, P = P)
            loss = loss.mean()
            # import pdb
            # pdb.set_trace
            loss.backward()
            optimizer.step()
        scheduler.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def test(model, dataloader, loss_function, device, P, mu, num_points, prob, M_sym, M_power, M_bandwidth, pul_power, freq_resp, batch_size, scheduler):
    test_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch[0].to(device)
            output = model(batch)

            h = batch[:, 1]
            b = torch.where(1/h < torch.sqrt(torch.tensor(P, device=device)), 1/h, torch.sqrt(torch.tensor(P, device=device)))
            # print(f'Test Batch b values: {b}') 

            # Simulate ISI symbols and channels
            ISI_symbols_real = torch.tensor(np.random.uniform(-np.sqrt(3), np.sqrt(3), (batch_size, mu-1)), device=device).float()
            ISI_symbols_imag = torch.tensor(np.random.uniform(-np.sqrt(3), np.sqrt(3), (batch_size, mu-1)), device=device).float()
            ISI_channels = torch.tensor(np.random.normal(0, 1/np.sqrt(2), (mu-1, 1)), device=device).float()
            noise = torch.randn(batch_size, device=device)

            loss = loss_function(batch_size, output, mu, batch[:, 2], batch[:, 3], ISI_symbols_real, ISI_symbols_imag, ISI_channels, num_points, noise, prob.to(device), device, M_sym, M_power, M_bandwidth, pul_power, freq_resp = freq_resp, P=P, test_bool=True)
            test_loss += loss.item()
    print(f'Test Loss: {test_loss/len(dataloader)}')

def main():
    device = prepare_device()

    # Define hyperparameters and constants
    num_points = 3501
    mu = 7
    P = 10
    batch_size = 100
    learning_rate = 0.01
    num_epochs = 2

    # Define MSE parameters
    M_sym = 2.3*10**1
    M_power = 9*10**1
    M_bandwidth = 15
    
    # delta_gap = torch.tensor([0.0002, 0.0005, 0.0008, 0.0013, 0.0008, 0.0005, 0.0002])
    freq_resp = 0.2
    pul_power = 10

    dataloader = prepare_dataloader(num_symbols=100000, M=64, P=P, batch_size=batch_size)
    model, optimizer, scheduler = prepare_model(input_size=4, output_size=1, hidden_layers=[64, 64, 64], learning_rate=learning_rate, device=device)
    loss_function = prepare_loss_function()

    # Define probability distribution for error samples
    sigma_error = np.array([0.1, 0.2])
    prob = prepare_probability_distribution(sigma_error, num_points, mu)

    train(model, dataloader, optimizer, loss_function, num_epochs, device, P, mu, num_points, prob, M_sym, M_power, M_bandwidth, pul_power, freq_resp, batch_size, scheduler)
    #test(model, dataloader, loss_function, device, P, mu, num_points, prob, M_sym, M_power, M_bandwidth, pul_power, freq_resp, batch_size, scheduler)

if __name__ == "__main__":
    main()
