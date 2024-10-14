import torch 
import config

def test(
    model,
    dataloader,
    loss_function,
    M_loss):
    model.eval() 
    
    total_loss = 0

    index = -1
    pulse_per_batch = torch.zeros(len(dataloader), config.num_points)

    with torch.no_grad(): 
        for batch_idx, batch in enumerate(dataloader):
            index = index + 1

            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            batch = batch[0].to(config.device)
            output = model(batch)
            pulse_per_batch[index, :] = torch.mean(output,dim=0)

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
                M_sym=None,
                M_power=None,
                M_bandwidth=None,
                test_bool=True
            )

            total_loss += loss.item()

    pulse_per_batch = torch.mean(pulse_per_batch, dim = 0)
    print(f"Test Loss: {total_loss / len(dataloader):.4f}")
    return total_loss / len(dataloader), pulse_per_batch
