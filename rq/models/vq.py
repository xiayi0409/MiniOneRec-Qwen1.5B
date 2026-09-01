import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.003, sk_iters=100,
                 cvq=False, cvq_anchor="probrandom", cvq_decay=0.99, cvq_scale=10.0):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        # CVQ-VAE (Zheng & Vedaldi, ICCV 2023): online clustered codebook.
        self.cvq = cvq
        self.cvq_anchor = cvq_anchor
        self.cvq_decay = cvq_decay
        self.cvq_scale = cvq_scale

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()
        self.register_buffer("embed_prob", torch.zeros(self.n_e))

    @torch.no_grad()
    def _cvq_online_update(self, latent, distances, indices):
        """Core online-CVQ update, adapted from lyndonzheng/CVQ-VAE.

        Active entries are still optimized by the usual VQ codebook loss.
        Low-EMA-usage entries instead move toward anchors sampled from the
        current encoded-feature distribution.
        """
        one_hot = F.one_hot(indices, num_classes=self.n_e).to(latent.dtype)
        avg_probs = one_hot.mean(dim=0)
        self.embed_prob.mul_(self.cvq_decay).add_(avg_probs, alpha=1.0-self.cvq_decay)
        if self.cvq_anchor == "closest":
            anchors = latent[torch.argmin(distances, dim=0)]
        elif self.cvq_anchor == "random":
            anchors = latent[torch.randint(latent.shape[0], (self.n_e,), device=latent.device)]
        elif self.cvq_anchor == "probrandom":
            # The paper samples one feature for each code from p(feature|code).
            probs = torch.softmax((-distances).t(), dim=1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
            anchors = latent[sampled]
        else:
            raise ValueError(f"Unsupported CVQ anchor: {self.cvq_anchor}")
        alpha = torch.exp(
            -(self.embed_prob * self.n_e * self.cvq_scale) / (1.0-self.cvq_decay) - 1e-3
        ).unsqueeze(1)
        old = self.embedding.weight.data.clone()
        self.embedding.weight.data.copy_(old * (1.0-alpha) + anchors * alpha)

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight.t())
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        # Match CVQ-VAE ordering: form the current quantized output/loss first,
        # then update only low-use entries for the following iteration.
        if self.training and self.cvq:
            self._cvq_online_update(latent.detach(), d.detach().float(), indices)

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices


