import torch
from torch import nn
from .base_model import CriticsBaseModel
from utils.utils import random_sample_from_est_mel_mag,dependent_sample_from_est_mel_mag,random_sample_components_from_bank
from torch.nn.utils.spectral_norm import SpectralNorm


class Memorybank(nn.Module):
    def __init__(self, N=1000, F=256, T=256):
        super().__init__()
        self.memory_bank=torch.randn([N, F, T], requires_grad=False)
        self.idx = 0
        self.N = N

    def enqueue(self, components):
        self.memory_bank = self.memory_bank#.to("cpu")
        components = components.detach()#.to("cpu")
        for i in range(components.size(0)):
            self.memory_bank[self.idx] = components[i]
            self.idx += 1
            if self.idx >= self.N:
                self.idx -= self.N


class PairSampleModule(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        if opts.use_memory_bank:
            self.memory_bank = Memorybank(opts.bank_size, F=opts.input_f_dim, T=opts.input_t_dim)

    def forward(self, inputs):

        est_mel_mag1 = inputs["est_mel_mag"].flatten(end_dim=2)  # (N,F,T)
        valid_num1 = inputs["components_valid_nums"].flatten()  # (N)
        N, F, T = est_mel_mag1.size()
        independent_pair = []
        for i in range(N):
            if valid_num1[i]:
                if "sample_2_from_bank" in self.opts and self.opts["sample_2_from_bank"]:
                    independent_pair.append(torch.stack([random_sample_components_from_bank(1,
                                                                                            self.memory_bank.memory_bank,
                                                                                            est_mel_mag1.device).view(F,T),
                                                         random_sample_components_from_bank(1,
                                                                                            self.memory_bank.memory_bank,
                                                                                            est_mel_mag1.device).view(F,T)]))  # (2,F,T)
                else:
                    independent_pair.append(torch.stack([est_mel_mag1[i],
                                                         random_sample_components_from_bank(1,
                                                                                            self.memory_bank.memory_bank,
                                                                                            est_mel_mag1.device).view(F,
                                                                                                                      T)]))  # (2,F,T)
                self.memory_bank.enqueue(est_mel_mag1[i].unsqueeze(0))
        independent_pair = torch.stack(independent_pair)  # (N,2,F,T)


        est_mel_mag2 = inputs["est_mel_mag"].flatten(start_dim=1, end_dim=2)  # (N1,N2,F,T)
        valid_num2 = inputs["components_valid_nums"].flatten(start_dim=1, end_dim=2).bool()  # (N1,N2)
        N1, N2, F, T = est_mel_mag2.size()
        dependent_pair = []
        for i in range(N1):
            i_valid_components = est_mel_mag2[i][valid_num2[i]]  # (N,F,T)
            dependent_sampled_components = dependent_sample_from_est_mel_mag(i_valid_components.unsqueeze(0))[
                0]  # (N,F,T)
            dependent_pair.append(
                torch.stack([i_valid_components, dependent_sampled_components], dim=1))  # (N,2,F,T)
        dependent_pair = torch.cat(dependent_pair, dim=0)  # (N,2,F,T)

        return independent_pair, dependent_pair