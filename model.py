import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NACCDataset

# dataset = NACCDataset("./data/investigator_nacc57.csv", "./features/combined")
# len(dataset.features)

class NACCEmbedder(nn.Module):
    def __init__(self, input_dim, latent=128, nhead=4, nlayers=3, num_pretrain_classes=3):
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

        # the modeling head
        self.linear3 = nn.Linear(latent, num_pretrain_classes)

        # tanh
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    # tau is the temperature parameter to normalize the encodings
    def forward(self, x, mask, 
                x_pos=None, pos_mask=None,
                x_neg=None, neg_mask=None, pretrain_target=None, tau = 0.05):

        base = self.linear0(torch.unsqueeze(x, dim=2))

        # Because we don't have a [CLS] token that's constant
        # we perform embedding by averaging all the tokens together

        base_encoding = self.encoder(base.transpose(0,1), src_key_padding_mask=mask).transpose(0,1)
        base_latent_state = torch.mean(base_encoding, dim=1)
        base_latent_state = self.tanh(self.linear1(base_latent_state))
        base_latent_state = self.tanh(self.linear2(base_latent_state))

        # calculate predictions and normalized latents
        preds = self.softmax(self.linear3(base_latent_state))
        normalized_latents = (base_latent_state.T/torch.norm(base_latent_state, dim=1)).T

        # if we are pretraining, just compute the pretraining target and loss
        if pretrain_target != None:
            return {
                "latent": normalized_latents,
                "logits": preds,
                "loss": torch.mean(torch.log(preds)*pretrain_target)
            }

        if not self.training:
            return {
                "latent": normalized_latents,
                "logits": preds
            }

        pos = self.linear0(torch.unsqueeze(x_pos, dim=2))
        neg = self.linear0(torch.unsqueeze(x_neg, dim=2))

        # compute positive "entailment" and "contradiction" pairs

        pos_encoding = self.encoder(pos.transpose(0,1), src_key_padding_mask=pos_mask).transpose(0,1)
        pos_latent_state = torch.mean(pos_encoding, dim=1)
        pos_latent_state = self.tanh(self.linear1(pos_latent_state))
        pos_latent_state = self.tanh(self.linear2(pos_latent_state))

        neg_encoding = self.encoder(neg.transpose(0,1), src_key_padding_mask=neg_mask).transpose(0,1)
        neg_latent_state = torch.mean(neg_encoding, dim=1)
        neg_latent_state = self.tanh(self.linear1(neg_latent_state))
        neg_latent_state = self.tanh(self.linear2(neg_latent_state))

        # compute pairwise consine similarity
        all_errors = []

        # for each element in the batch, compute the loss
        col_loss = []

        for i in range(x.shape[0]):
            # compute hi, h+, and h-
            h_i = base_latent_state[i]
            h_p = pos_latent_state[i]

            # e^(sim(hi, hp)/tau)
            top = torch.exp(((h_i.T @ h_p)/torch.norm(h_i)/torch.norm(h_p))/tau)

            bottom = []

            # calculate each negative value
            for j in range(x.shape[0]):
                hj_p = pos_latent_state[j]
                hj_n = neg_latent_state[j]
                bottom.append(torch.exp((((h_i.T @ hj_p)/torch.norm(h_i)/torch.norm(hj_p)))/tau) + 
                              torch.exp((((h_i.T @ hj_n)/torch.norm(h_i)/torch.norm(hj_n)))/tau))

            col_loss.append(-torch.log(top/torch.sum(torch.stack(bottom))))

        return {
            "latent": normalized_latents,
            "pos": pos_latent_state,
            "neg": neg_latent_state,
            "logits": preds,
            "loss": torch.mean(torch.stack(col_loss))
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
        
