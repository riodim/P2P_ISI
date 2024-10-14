import torch
import torch.nn as nn
import numpy as np
import utils as utils

def MSE_sampling_ISI(mu, b, x_real, x_imag, x_ISI_real, x_ISI_imag, channels, ISI_channels, sample_time, T, dnn_out, batch_size, device):
    num_ISI = np.floor(mu / 2).astype(int)

    y_ISI_real = torch.zeros(batch_size, 1).to(device)
    y_ISI_imag = torch.zeros(batch_size, 1).to(device)
    y_rec_real = torch.zeros(batch_size, 1).to(device)
    y_rec_imag = torch.zeros(batch_size, 1).to(device)

    index = 0
    b = torch.reshape(b, (batch_size, 1))
    channels = torch.reshape(channels, (batch_size, 1))

    for w2 in range(mu):        
        if w2 != num_ISI:

            y_ISI_real = y_ISI_real + b*ISI_channels[:,index]*x_ISI_real[:,index]*(dnn_out[:,sample_time+(w2-num_ISI)*2*T].unsqueeze(1)).repeat(1,1)
            y_ISI_imag = y_ISI_imag + b*ISI_channels[:,index]*x_ISI_imag[:,index]*(dnn_out[:,sample_time+(w2-num_ISI)*2*T].unsqueeze(1)).repeat(1,1)

        else:
            y_rec_real = b*channels*x_real*(dnn_out[:,sample_time].unsqueeze(1))
            y_rec_imag = b*channels*x_imag*(dnn_out[:,sample_time].unsqueeze(1))

    y_ISI_total_real = y_ISI_real + y_rec_real
    y_ISI_total_imag = y_ISI_imag + y_rec_imag
 
    return y_ISI_total_real, y_ISI_total_imag

class ISI_loss_function(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(ISI_loss_function, self).__init__()
        self.epsilon = epsilon

    def forward(
        self,
        dnn_out,
        mu,
        b,
        h,
        x_real,
        x_imag,
        ISI_symbols_real,
        ISI_symbols_imag,
        ISI_channels,
        num_points,
        noise,
        prob,
        batch_size,
        device,
    ):
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        mid_sample = np.floor(num_points / 2).astype(int)
        T = np.floor(np.floor(num_points / mu) / 2).astype(int)

        # Loop through the sample time offsets to compute the loss over a range of samples
        for dist in range(-T, T):
            # Compute the sample time
            sample_time = mid_sample + dist

            # Compute the signal for the current batch
            y_signal_real, y_signal_imag = MSE_sampling_ISI(
                mu=mu,
                b=b,
                x_real=x_real,
                x_imag=x_imag,
                x_ISI_real=ISI_symbols_real,
                x_ISI_imag=ISI_symbols_imag,
                channels=h,
                ISI_channels=ISI_channels,
                sample_time=sample_time,
                T=T,
                dnn_out=dnn_out,
                batch_size=batch_size,  # Using batch_size for this computation
                device=device,
            )

            # Set the target values (real and imaginary parts of x)
            y_target_real = x_real
            y_target_imag = x_imag

            # Compute the total signal with noise added
            y_total_real = y_signal_real + 1 / np.sqrt(2) * noise
            y_total_imag = y_signal_imag + 1 / np.sqrt(2) * noise

            # Calculate the loss using the probability index
            prob_index = sample_time
            loss = loss + prob[0, prob_index] * (
                torch.mean(torch.square(y_total_real - y_target_real)) +
                torch.mean(torch.square(y_total_imag - y_target_imag))
            )

        return loss
