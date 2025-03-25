import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=96, use_attention=True, 
                 dropout_rate=0.5, residual_layers=2, filters=(32, 64, 128)):
        super(Autoencoder, self).__init__()
        
        # Encoder construction
        encoder_layers = []
        in_channels = 3
        
        for i, out_channels in enumerate(filters):
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            if i < residual_layers:
                encoder_layers.append(ResidualBlock(out_channels))
                
            in_channels = out_channels
        
        encoder_layers.extend([
            nn.Conv2d(filters[-1], latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder construction
        decoder_layers = []
        in_channels = latent_dim
        reversed_filters = list(reversed(filters))
        
        decoder_layers.extend([
            nn.ConvTranspose2d(latent_dim, reversed_filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(reversed_filters[0]),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        if 0 < residual_layers:
            decoder_layers.append(ResidualBlock(reversed_filters[0]))
        
        for i in range(len(reversed_filters)-1):
            decoder_layers.extend([
                nn.ConvTranspose2d(reversed_filters[i], reversed_filters[i+1], 
                                  kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(reversed_filters[i+1]),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            if i+1 < residual_layers:
                decoder_layers.append(ResidualBlock(reversed_filters[i+1]))
        
        decoder_layers.extend([
            nn.ConvTranspose2d(reversed_filters[-1], 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(latent_dim)
        
        self.channel_dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        encoded = self.encoder(x)
        if self.use_attention:
            encoded = self.attention(encoded)
        if self.training:
            encoded = self.channel_dropout(encoded)
        decoded = self.decoder(encoded)
        return decoded, encoded