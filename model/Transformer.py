import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TransformerClassifier, self).__init__()

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        # Transformer Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_size, nhead=1)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=3)
        # Estimation network
        self.estimation_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            #nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (seq_len, batch_size, input_size)
        x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)  # Change to (batch_size, seq_len, input_size)

        # Use the transformer encoder
        h = self.transformer_encoder(x)
        h = self.transformer_decoder(h,h)
        h = h.permute(1, 0, 2)  # Change back to (batch_size, seq_len, input_size)

        # Use the last token as the context
        h = h[:, -1, :]

        # Use the estimation network for classification
        #y_hat = self.estimation_net(h)
        #不使用分类器

        return h, h

def transformer_loss(y, y_hat):
    diff_loss = nn.CrossEntropyLoss()(y_hat, y)
    return diff_loss