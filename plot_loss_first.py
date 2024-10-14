import matplotlib.pyplot as plt
import csv
import os

def plot_loss_graph(loss_csv_path, batch_interval):
    # Lists to store data from CSV
    epochs = []
    batches = []
    losses = []

    # Read the CSV file
    with open(loss_csv_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for i, row in enumerate(reader):
            if i % batch_interval == 0:  # Only take every 10th batch (or the defined interval)
                epoch, batch, loss = int(row[0]), int(row[1]), float(row[2])
                epochs.append(epoch)
                batches.append(batch)
                losses.append(loss)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, marker='o', label='Loss')

    # Adding labels and title
    plt.xlabel(f'Batch (every {batch_interval} batches)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Loss Over Time (every {batch_interval}th batch)', fontsize=14)
    plt.grid(True)

    # Determine the directory from loss_csv_path
    csv_dir = os.path.dirname(loss_csv_path)
    
    # Define the plots folder inside the same directory
    plots_folder = os.path.join(csv_dir, "plots")
    
    # Create the plots folder if it doesn't exist
    os.makedirs(plots_folder, exist_ok=True)

    # Generate dynamic plot save path based on batch_interval inside the plots folder
    plot_save_path = os.path.join(plots_folder, f"loss_plot_{batch_interval}_interval.png")
    # Save the plot to file
    plt.savefig(plot_save_path)
    plt.close()  # Close the plot to avoid displaying it inline (for scripts)

    print(f"Loss plot saved to {plot_save_path}")

def main():
    loss_csv_file = "./model/loss_data.csv"

    # Call the function to plot the loss graph and save it as an image
    for interval in range(21, 50):
        plot_loss_graph(loss_csv_file, batch_interval=interval)

def plot_graph(loss_csv_file):

    for interval in range(1, 21):
        plot_loss_graph(loss_csv_file, batch_interval=interval)

if __name__ == "__main__":
    main()
