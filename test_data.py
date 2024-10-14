import os
import numpy as np
import torch
import utils  # Ensure this imports the necessary utilities
import model_loss_testing  # Ensure this imports the loss function and model class
import symbol_utils as sym_utils  # For QAM symbol generation

# Define the folder where the trained data files are located
trained_data_folder = './trainedData'
results_folder = './results/test_data'
batch_size = 100
model_save_path = './model/trained_model.pth'  # Path to the trained model

# Create the folder for test results if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

# Function to load the model
def load_trained_model(device, model_class, input_size, output_size, hidden_layers):
    model = model_class(input_size, output_size, hidden_layers).to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()  # Set model to evaluation mode
    return model

# Function to load data, strip the first line, and run the test phase with batching
def run_test_on_saved_data(file_path, model, loss_function, device, mu, num_points, ISI_channels, prob, num_symbols, P, M):
    print(f"Loading data from {file_path}...")
    
    # Load the saved pulse data, skipping the first line
    with open(file_path, 'r') as f:
        _ = f.readline().strip()  # Strip the first line (M values)
        pulse_data = np.loadtxt(f)  # Load the pulse data (all symbols and points)

    print(f"Data loaded. Converting to tensor...")

    # Convert pulse_data to tensor (entire dataset of 10,000 symbols)
    pulse_tensor = torch.tensor(pulse_data, dtype=torch.float32).to(device)

    print("Re-generating QAM symbols and related values...")
    
    # Re-generate the necessary values used during training
    qam_symbols = sym_utils.generate_qam_symbols(num_symbols, M)  # Re-generate QAM symbols
    h = utils.generate_h(num_symbols)  # Generate the channel coefficients
    b = utils.calculate_b(h_data=h, P=P, simRuns=num_symbols)  # Calculate the b values

    x_real = torch.tensor(np.real(qam_symbols), dtype=torch.float32).to(device)  # Real part of the symbols
    x_imag = torch.tensor(np.imag(qam_symbols), dtype=torch.float32).to(device)  # Imaginary part of the symbols

    # Split the data into batches of size 100
    num_batches = num_symbols // batch_size
    total_loss = 0.0

    for batch_idx in range(num_batches):
        print(f"Processing batch {batch_idx + 1}/{num_batches}...")

        # Select the batch slice
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        pulse_batch = pulse_tensor[start_idx:end_idx]
        x_real_batch = x_real[start_idx:end_idx]
        x_imag_batch = x_imag[start_idx:end_idx]
        h_batch = torch.tensor(h[start_idx:end_idx], dtype=torch.float32).to(device)
        b_batch = torch.tensor(b[start_idx:end_idx], dtype=torch.float32).to(device)

        # Prepare ISI symbols for the batch
        ISI_symbols_real_batch = torch.stack([torch.roll(x_real_batch, shifts=i, dims=0) for i in range(1, mu)], dim=1)
        ISI_symbols_imag_batch = torch.stack([torch.roll(x_imag_batch, shifts=i, dims=0) for i in range(1, mu)], dim=1)

        print("Generating noise for the batch...")
        noise_batch = torch.randn(batch_size, device=device)  # Generate noise for the batch

        print("Calling the loss function for the batch...")
        # Compute the loss for the current batch
        batch_loss = loss_function(
            dnn_out=model(pulse_batch.unsqueeze(0)),  # Use the model's output here
            mu=mu,
            b=b_batch,
            h=h_batch,
            x_real=x_real_batch,
            x_imag=x_imag_batch,
            ISI_symbols_real=ISI_symbols_real_batch,
            ISI_symbols_imag=ISI_symbols_imag_batch,
            ISI_channels=ISI_channels,
            num_points=num_points,
            noise=noise_batch,
            prob=prob.to(device),
            device=device,
            batch_size=batch_size
        )

        print(f"Batch {batch_idx + 1}/{num_batches} loss: {batch_loss.item()}")
        total_loss += batch_loss.item()

    avg_loss = total_loss / num_batches
    print(f"Average Test Loss for {file_path}: {avg_loss}")

    # Save the new test results (entire dataset)
    output_filename = os.path.join(results_folder, os.path.basename(file_path).replace('.txt', '_test.txt'))
    np.savetxt(output_filename, pulse_tensor.cpu().numpy())
    print(f"Test data saved to: {output_filename}")
    return output_filename

def main():
    print("Preparing device...")
    device = utils.prepare_device()
    mu = 7  
    num_points = 3501  
    num_symbols = 10000
    P = 10  
    M = 64  
    
    # Load the loss function
    loss_function = model_loss_testing.ISI_loss_function()

    # Load the trained model
    model = load_trained_model(device, model_loss_testing.OTAPulse, input_size=4, output_size=num_points, hidden_layers=[256, 256, 256])

    # Generate probability distribution (adjust parameters as needed)
    print("Generating probability distribution...")
    sigma_error = np.array([0.1, 0.2])
    prob = utils.prepare_probability_distribution(sigma_error, num_points, mu)
    
    # Generate ISI channels (adjust parameters as needed)
    print("Generating ISI channels...")
    ISI_channels = torch.tensor(
        utils.generate_h(num_points=mu - 1), dtype=torch.float32
    ).view(1, mu - 1)

    # Iterate through all files in the trained data folder
    for filename in os.listdir(trained_data_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(trained_data_folder, filename)
            print(f"Processing file: {file_path}")
            output_file = run_test_on_saved_data(
                file_path=file_path,
                model=model,
                loss_function=loss_function,
                device=device,
                mu=mu,
                num_points=num_points,
                ISI_channels=ISI_channels,
                prob=prob,
                num_symbols=num_symbols,
                P=P,
                M=M,
            )
    
    # After processing all files, run the post-processing on the results
    print("Processing all files in folder...")
    utils.process_all_files_in_folder(results_folder, results_folder)

if __name__ == "__main__":
    main()
