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


def prepare_dataloader(num_symbols, M, P, batch_size, device, fixed_h):
    qam_symbols = utils.generate_qam_symbols(num_symbols, M)
    utils.plot_constellation(qam_symbols, "qam_constellation.png")

    x_real = np.real(qam_symbols)
    x_imag = np.imag(qam_symbols)

    # Generate h values using Rayleigh fading model (fixed for all epochs)
    h_data = fixed_h
    if np.any(h_data <= 0):
        raise ValueError(
            "h_data contains non-positive values, which could lead to invalid operations."
        )

    # Ensure b_tensor is computed for each symbol
    b = utils.calculate_b(h_data=h_data, P=P, simRuns=num_symbols)

    # Debugging: Print shapes to ensure consistency
    print(f"b shape: {b.shape}")
    print(f"h_data shape: {h_data.shape}")
    print(f"x_real shape: {x_real.shape}")
    print(f"x_imag shape: {x_imag.shape}")

    # Ensure all arrays have the correct size before stacking
    inputs = np.vstack((b, h_data, x_real, x_imag)).T

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
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
        prob[iterSig] = utils.pmf_extract(
            num_points, mu, var_sample, 10000, np.linspace(-mu / 2, mu / 2, num_points)
        )

    return prob


def train(
    model,
    dataloader,
    optimizer,
    loss_function,
    num_epochs,
    device,
    P,
    mu,
    num_points,
    prob,
    M_sym,
    M_power,
    M_bandwidth,
    pul_power,
    freq_resp,
    batch_size,
    scheduler,
):
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            output = model(batch)

            # Simulate ISI symbols and channels
            ISI_symbols_real = (
                -torch.sqrt(torch.tensor(3.0))
                + 2 * torch.sqrt(torch.tensor(3.0)) * torch.rand((batch_size, mu - 1))
            ).to(device)
            ISI_symbols_imag = (
                -torch.sqrt(torch.tensor(3.0))
                + 2 * torch.sqrt(torch.tensor(3.0)) * torch.rand((batch_size, mu - 1))
            ).to(device)
            ISI_channels = (
                torch.sqrt(torch.tensor(1 / 2.0)) * torch.randn((mu - 1, 1))
            ).to(device)
            
            noise = torch.randn(batch_size, device=device)

            loss = loss_function(
                batch_size=batch_size,
                dnn_out=output,
                mu=mu,
                batch = batch, 
                ISI_symbols_real=ISI_symbols_real,
                ISI_symbols_imag=ISI_symbols_imag,
                ISI_channels=ISI_channels,
                num_points=num_points,
                noise=noise,
                prob=prob.to(device),
                device=device,
                M_sym=M_sym,
                M_power=M_power,
                M_bandwidth=M_bandwidth,
                pul_power=pul_power,
                freq_resp=freq_resp,
                P=P,
            )
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

def test(
    model,
    dataloader,
    loss_function,
    device,
    P,
    mu,
    num_points,
    prob,
    M_sym,
    M_power,
    M_bandwidth,
    pul_power,
    freq_resp,
    batch_size,
):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation for testing
        for batch in dataloader:
            batch = batch[0].to(device)
            output = model(batch)

            # Simulate ISI symbols and channels
            ISI_symbols_real = (-torch.sqrt(torch.tensor(3.0)) + 2 * torch.sqrt(torch.tensor(3.0)) * torch.rand((batch_size, mu - 1))).to(device)
            ISI_symbols_imag = (-torch.sqrt(torch.tensor(3.0)) + 2 * torch.sqrt(torch.tensor(3.0)) * torch.rand((batch_size, mu - 1))).to(device)
            ISI_channels = (torch.sqrt(torch.tensor(1 / 2.0)) * torch.randn((mu - 1, 1))).to(device)
            noise = torch.randn(batch_size, device=device)

            loss = loss_function(
                batch_size=batch_size,
                inputs=output,
                mu=mu,
                x_real=batch[:, 2],
                x_imag=batch[:, 3],
                ISI_symbols_real=ISI_symbols_real,
                ISI_symbols_imag=ISI_symbols_imag,
                ISI_channels=ISI_channels,
                num_points=num_points,
                noise=noise,
                prob=prob.to(device),
                device=device,
                M_sym=M_sym,
                M_power=M_power,
                M_bandwidth=M_bandwidth,
                pul_power=pul_power,
                freq_resp=freq_resp,
                P=P,
                test_bool=True,  # Indicate that this is a test run
            )
            loss = loss.mean()
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Test Loss: {average_loss:.4f}")
    return average_loss

def main():
    device = prepare_device()

    # Define hyperparameters and constants
    num_points = 3501
    mu = 7
    P = 10
    batch_size = 100
    learning_rate = 0.003
    num_epochs = 5

    # Define MSE parameters
    M_sym = 1.3 * 10**2
    M_power = 1 * 10
    M_bandwidth = 12

    freq_resp = 0.2
    pul_power = 10

    num_symbols = 10000

    # Generate a fixed h for all epochs
    fixed_h = utils.generate_h(num_symbols)  # Assuming num_symbols = 10000

    dataloader = prepare_dataloader(
        num_symbols=10000,
        M=64,
        P=P,
        batch_size=batch_size,
        device=device,
        fixed_h=fixed_h,
    )
    model, optimizer, scheduler = prepare_model(
        input_size=4,  # Assuming your input to the model is 4-dimensional
        output_size=num_points,  # The model should output 4-dimensional vectors
        hidden_layers=[256, 256, 256],
        learning_rate=learning_rate,
        device=device,
    )
    loss_function = prepare_loss_function()

    # Define probability distribution for error samples
    sigma_error = np.array([0.1, 0.2])
    prob = prepare_probability_distribution(sigma_error, num_points, mu)

    train(
        model,
        dataloader,
        optimizer,
        loss_function,
        num_epochs,
        device,
        P,
        mu,
        num_points,
        prob,
        M_sym,
        M_power,
        M_bandwidth,
        pul_power,
        freq_resp,
        batch_size,
        scheduler,
    )

    test(
        model,
        dataloader,
        loss_function,
        device,
        P,
        mu,
        num_points,
        prob,
        M_sym,
        M_power,
        M_bandwidth,
        pul_power,
        freq_resp,
        batch_size,
    )

if __name__ == "__main__":
    main()
