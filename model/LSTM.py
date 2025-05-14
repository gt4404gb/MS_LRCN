
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()

        # LSTM Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)

        # Estimation network
        self.estimation_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # Use the last hidden state as the context
        h,_ = self.encoder(x)
        h = h.squeeze(0)  # Remove the num_layers dimension

        # Use the estimation network for classification
        y_hat = self.estimation_net(h)

        return h,y_hat


def lstm_loss(y, y_hat):
    diff_loss = nn.CrossEntropyLoss()(y_hat, y)
    return diff_loss