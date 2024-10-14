import time
import config
import torch 
import csv
import os 
import plot_loss
 
def train(
    model,
    dataloader,
    optimizer,
    loss_function,
    M_loss,
    M_sym,
    M_power,
    M_bandwidth,
    scheduler
):
    loss_data = []
    for epoch in range(config.num_epochs):
        start_time = time.time()  # Start the timer for the epoch
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            batch = batch[0].to(config.device)
            optimizer.zero_grad()
            output = model(batch)

            x_real = batch[:, 2]
            x_imag = batch[:, 3] 

            # Apply roll operation to simulate ISI from neighboring symbols
            ISI_symbols_real = torch.stack([torch.roll(x_real, shifts=i, dims=0) for i in range(1, config.mu)], dim=1)
            ISI_symbols_imag = torch.stack([torch.roll(x_imag, shifts=i, dims=0) for i in range(1, config.mu)], dim=1)
            
            noise = torch.randn(config.batch_size, device=config.device)

            loss = loss_function(
                dnn_out=output,
                batch=batch,
                ISI_symbols_real=ISI_symbols_real,
                ISI_symbols_imag=ISI_symbols_imag,
                noise=noise,
                M_loss=M_loss,
                M_sym=M_sym,
                M_power=M_power,
                M_bandwidth=M_bandwidth,
            )
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            print(f" Loss: {loss.item()}")
            total_loss += loss.item()
            
            if (batch_idx + 1) % config.loss_save_interval == 0:
                loss_data.append([epoch + 1, batch_idx + 1, loss.item()]) 

        scheduler.step()
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {loss.item()}, Time: {epoch_time:.2f} seconds")

    # Save the trained model's state_dict
    model_save_path = f"./test_results/roll_off_{config.roll_off}/model/M_loss_{M_loss}/trained_model.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    loss_save_path=f"./test_results/roll_off_{config.roll_off}/model/M_loss_{M_loss}/loss_data.csv" 
    os.makedirs(os.path.dirname(loss_save_path), exist_ok=True)
    with open(loss_save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Batch", "Loss"])
        writer.writerows(loss_data)
    print(f"Loss data saved to {loss_save_path}")

    plot_loss.plot_loss_graph( loss_csv_path = loss_save_path)

