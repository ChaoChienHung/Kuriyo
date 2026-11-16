"""Residual-quantized autoencoder for item-content embeddings."""

import torch
from torch import nn
from torch.nn import functional as F


class RQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256, 128), latent_dim=32,
                 levels=3, codebook_size=256, commitment_weight=0.25):
        super().__init__()
        if input_dim < 1 or latent_dim < 1 or levels < 1 or codebook_size < 2:
            raise ValueError("invalid RQ-VAE dimensions")
        dims = (input_dim, *hidden_dims, latent_dim)
        encoder = []
        for index in range(len(dims) - 1):
            encoder.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)
        reverse = (latent_dim, *reversed(hidden_dims), input_dim)
        decoder = []
        for index in range(len(reverse) - 1):
            decoder.append(nn.Linear(reverse[index], reverse[index + 1]))
            if index < len(reverse) - 2:
                decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder)
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.empty(codebook_size, latent_dim)) for _ in range(levels)
        ])
        for codebook in self.codebooks:
            nn.init.uniform_(codebook, -1.0 / codebook_size, 1.0 / codebook_size)
        self.commitment_weight = commitment_weight

    def quantize(self, latent):
        residual = latent
        selected = []
        codes = []
        codebook_loss = latent.new_zeros(())
        commitment_loss = latent.new_zeros(())
        for codebook in self.codebooks:
            distances = (residual.square().sum(-1, keepdim=True)
                         + codebook.square().sum(-1).unsqueeze(0)
                         - 2 * residual @ codebook.T)
            index = distances.argmin(-1)
            vector = F.embedding(index, codebook)
            codebook_loss = codebook_loss + F.mse_loss(vector, residual.detach())
            commitment_loss = commitment_loss + F.mse_loss(residual, vector.detach())
            selected.append(vector)
            codes.append(index)
            residual = residual - vector.detach()
        quantized = torch.stack(selected).sum(0)
        straight_through = latent + (quantized - latent).detach()
        return (straight_through, torch.stack(codes, dim=-1), codebook_loss,
                commitment_loss)

    def forward(self, inputs):
        latent = self.encoder(inputs)
        quantized, codes, codebook_loss, commitment_loss = self.quantize(latent)
        reconstruction = self.decoder(quantized)
        reconstruction_loss = F.mse_loss(reconstruction, inputs)
        loss = reconstruction_loss + codebook_loss + self.commitment_weight * commitment_loss
        return loss, codes, {
            "reconstruction": reconstruction_loss.detach(),
            "codebook": codebook_loss.detach(),
            "commitment": commitment_loss.detach(),
        }

    @torch.no_grad()
    def encode_codes(self, inputs):
        return self.quantize(self.encoder(inputs))[1]
