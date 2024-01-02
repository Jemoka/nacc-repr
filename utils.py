import torch
from torch.utils.data import DataLoader, RandomSampler

def sample_uniformity(model, dataset, num_samples=100):
    # get device
    device = next(model.parameters()).device

    # make N samples samples
    sampler = iter(DataLoader(dataset, batch_size=2, shuffle=True))
    samples = []

    # for each sample, calculcate uniformity objective
    for _ in range(num_samples):
        # get sample
        sample = next(sampler)
        inp = sample[0]
        mask = sample[1]

        # calculate embedding latents
        with torch.inference_mode():
            was_training = model.training
            model.eval()
            x,y = model(inp.to(device), mask.to(device))["latent"]
            if was_training:
                model.train()

        # calculate uniformity objective
        square_dist = torch.norm(x/(torch.norm(x))-y/torch.norm(y))**2
        samples.append(torch.log(torch.exp(-2*(square_dist))))

    return torch.mean(torch.stack(samples))

def sample_alignment(model, dataset, num_samples=100):
    # get device
    device = next(model.parameters()).device

    # make N samples samples
    sampler = iter(DataLoader(dataset, batch_size=1, shuffle=True))
    samples = []

    # for each sample, calculcate uniformity objective
    for _ in range(num_samples):
        # get sample
        sample = next(sampler)
        inp_1 = sample[0]
        mask_1 = sample[1]
        target = sample[-1]

        # and get its values
        inp_2 = sample[2]
        mask_2 = sample[3]

        # calculate embedding latents
        with torch.inference_mode():
            was_training = model.training
            model.eval()
            x,y = (model(inp_1.to(device), mask_1.to(device))["latent"],
                   model(inp_2.to(device), mask_2.to(device))["latent"])
            if was_training:
                model.train()

        # calculate uniformity objective
        square_dist = torch.norm((x/torch.norm(x)-y/torch.norm(y)).squeeze())**2
        samples.append(square_dist)

    return torch.mean(torch.stack(samples))



# sample[0].shape
# tmp_a = sample_alignment(emb, dataset)
# tmp_u = sample_uniformity(emb, dataset)

# tmp_a
# tmp_u


# tmp
# tmp




