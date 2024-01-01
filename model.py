import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NACCDataset

# dataset = NACCDataset("./data/investigator_nacc57.csv", "./features/combined")
# len(dataset.features)

class NACCEmbedder(nn.Module):
    def __init__(self, input_dim, latent=128, nhead=4, nlayers=3):
        # call early initializers
        super(NACCEmbedder, self).__init__()

        # the entry network ("linear embedding")
        # bigger than 80 means that its going to be out of bounds and therefore
        # be masked out; so hard code 81
        self.linear0 = nn.Linear(1, latent)
        
        # the encoder network
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers, enable_nested_tensor=False)

        # the MLP
        self.linear1 = nn.Linear(latent, latent)
        self.linear2 = nn.Linear(latent, latent)

        # tanh
        self.tanh = nn.Tanh()

    # tau is the temperature parameter to normalize the encodings
    def forward(self, x, mask, tau = 0.05):
        net = self.linear0(torch.unsqueeze(x, dim=2))

        # pass through the model twice
        # Because we don't have a [CLS] token that's constant
        # we perform embedding by averaging all the tokens together

        first_encoding = self.encoder(net.transpose(0,1), src_key_padding_mask=mask).transpose(0,1)
        first_latent_state = torch.mean(first_encoding, dim=1)
        first_latent_state = self.tanh(self.linear1(first_latent_state))
        first_latent_state = self.tanh(self.linear2(first_latent_state))

        if not self.training:
            return {
                "latent": first_latent_state,
            }

        second_encoding = self.encoder(net.transpose(0,1)).transpose(0,1)
        second_latent_state = torch.mean(second_encoding, dim=1)
        second_latent_state = self.tanh(self.linear1(second_latent_state))
        second_latent_state = self.tanh(self.linear2(second_latent_state))

        # compute pairwise consine similarity
        all_sims = []
        agreeing_sims = []
        
        # calculate pairwise cosine similarity, setting up for
        # eqn1 for Gao et al
        for indx_i, i in enumerate(first_latent_state):
            for indx_j, j in enumerate(second_latent_state):
                sim = (i.T @ j)/torch.norm(i)/torch.norm(j)
                all_sims.append(sim)
                if indx_i == indx_j:
                    agreeing_sims.append(sim)

        # tabulate the similaities, and use it to formulate 
        # the training objective (eqn 1 in gao et al)
        normalization = torch.sum(torch.exp(torch.stack(all_sims))/tau)
        agreeing_sims = torch.exp(torch.stack(agreeing_sims))/tau

        contrastive_obj = -torch.log(agreeing_sims/normalization)

        return {
            "latent": first_latent_state,
            "loss": torch.mean(contrastive_obj)
        }

# emb = NACCEmbedder(len(dataset.features))
# sample = torch.stack([dataset[i][0] for i in range(1,5)])
# mask = torch.stack([dataset[i][1] for i in range(1,5)])

# emb.train()
# sim = emb(sample, mask)
# sim
# sim
# sim
# torch.norm(
# first[0]
# second[0]
        
