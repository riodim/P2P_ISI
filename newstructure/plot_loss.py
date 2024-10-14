import matplotlib.pyplot as plt
import csv
import os
import config

def plot_loss_graph(loss_csv_path, batch_interval =config.loss_save_interval):
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

def plot_graph(loss_csv_file):

    for interval in range(1, 21):
        plot_loss_graph(loss_csv_file, batch_interval=interval)

def main():
    for M_loss in config.M_loss_values:
        loss_csv_file = f"./test_results/roll_off_{config.roll_off}/model/M_loss_{M_loss}/loss_data.csv"

        for interval in range(5, 21):
            plot_loss_graph(loss_csv_file, batch_interval=interval)

def plot_multiple_graphs():
    # Paths to the two CSV files
    loss_csv_file1 = f"./test_results/test_results_learn_rate_0.002/roll_off_{config.roll_off}/model/M_loss_4/loss_data.csv"
    loss_csv_file2 = f"./test_results/test_results_learn_rate_0.001/roll_off_{config.roll_off}/model/M_loss_4/loss_data.csv"
    loss_csv_file3 = f"./test_results/test_results_learn_rate_0.003/roll_off_{config.roll_off}/model/M_loss_4/loss_data.csv"
    loss_csv_file4 = f"./test_results/test_results_learn_rate_0.005/roll_off_{config.roll_off}/model/M_loss_4/loss_data.csv"

    # Lists to store data for both files
    losses1 = []
    losses2 = []
    losses3 = []
    losses4 = []

    # Batch interval (you can set this value as desired)
    batch_interval = 3

    with open(loss_csv_file1, mode='r') as file1:
        reader1 = csv.reader(file1)
        next(reader1)  
        for i, row in enumerate(reader1):
            if i % batch_interval == 0:
                losses1.append(float(row[2]))  

    with open(loss_csv_file2, mode='r') as file2:
        reader2 = csv.reader(file2)
        next(reader2)  
        for i, row in enumerate(reader2):
            if i % batch_interval == 0:
                losses2.append(float(row[2]))  

    with open(loss_csv_file3, mode='r') as file3:
        reader3 = csv.reader(file3)
        next(reader3)  
        for i, row in enumerate(reader3):
            if i % batch_interval == 0:
                losses3.append(float(row[2]))  

    with open(loss_csv_file4, mode='r') as file4:
        reader4 = csv.reader(file4)
        next(reader4) 
        for i, row in enumerate(reader4):
            if i % batch_interval == 0:
                losses4.append(float(row[2]))  

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot both loss graphs
    plt.plot(range(len(losses1)), losses1, marker='o', label='Loss (LR=0.002)', color='blue')
    plt.plot(range(len(losses2)), losses2, marker='x', label='Loss (LR=0.001)', color='green')
    plt.plot(range(len(losses3)), losses3, marker='+', label='Loss (LR=0.003)', color='red')
    plt.plot(range(len(losses4)), losses4, marker='s', label='Loss (LR=0.005)', color='orange')

    # Adding labels and title
    plt.xlabel(f'Batch (every {batch_interval} batches)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Loss Over Time (every {batch_interval}th batch)', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plot_save_path = f"./test_results/learning_rate_comparison/loss_comparison_lr_0_005_lr_0_003_lr_0_002_lr_0_001.png"
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    plt.close()  # Close the plot

    print(f"Comparison plot saved to {plot_save_path}")
if __name__ == "__main__":
    plot_multiple_graphs()
