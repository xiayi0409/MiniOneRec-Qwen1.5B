import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 0.25,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100,
                 cvq=False, cvq_anchor="probrandom", cvq_decay=0.99, cvq_scale=10.0):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.cvq = cvq
        self.cvq_anchor = cvq_anchor
        self.cvq_decay = cvq_decay
        self.cvq_scale = cvq_scale
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        beta=self.beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters,
                                                        cvq=self.cvq,
                                                        cvq_anchor=self.cvq_anchor,
                                                        cvq_decay=self.cvq_decay,
                                                        cvq_scale=self.cvq_scale)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])

    @torch.no_grad()
    def reset_dead_codes(self, z, indices, threshold=1):
        """Replace rarely-used codewords with residual encoder features."""
        residual = z.detach().clone()
        total_reset = 0
        for level, quantizer in enumerate(self.vq_layers):
            ids = indices[..., level].reshape(-1).to(z.device)
            counts = torch.bincount(ids, minlength=quantizer.n_e)
            dead = torch.nonzero(counts <= threshold, as_tuple=False).flatten()
            if dead.numel():
                candidates = residual.reshape(-1, self.e_dim)
                if candidates.shape[0] == 0:
                    break
                pick = torch.randint(candidates.shape[0], (dead.numel(),), device=z.device)
                quantizer.embedding.weight.data[dead] = candidates[pick]
                total_reset += int(dead.numel())
            q = quantizer.embedding(ids).view_as(residual)
            residual = residual - q
        return total_reset

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices