import torch
import torch.nn as nn
import numpy as np
from torch.autograd.functional import jacobian
import tqdm as nbk
import torch.autograd.profiler as profiler
import functools
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from torch.func import jacrev, vmap
import time
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.networks import ScoreNet
from simulation_functions.gbaoab_functions import cotangent_projection, create_constraints,gBAOAB_integrator_retain, gBAOAB_integrator

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--gamma", type=float)
parser.add_argument("--h", type = float, help="Stepsize")
parser.add_argument("--root", type = str, help="Root directory for saving results")
parser.add_argument("--k", type = float, help="Kb T for ")
args = parser.parse_args()
import os



score_model = ScoreNet(80).cuda()

gs = create_constraints()

L_fn = cotangent_projection(gs)


#@title Define the loss function (double click to expand or collapse)
def loss(model, xp,eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """

  # x is the batch of simulated qps that we need to noise.
  gs = create_constraints()

  # projection matrix
  L_fn = cotangent_projection(gs)

  x = xp[:,:3]
  p = xp[:,3:]
  random_t = torch.round((torch.rand(1, device=x.device)**1.75)*(300))

  # Noising

  M = torch.eye(x.shape[1]).broadcast_to(x.shape[0], x.shape[1],x.shape[1]).cuda()

  h = args.h
  k = args.k
  gamma = args.gamma

  def force(x):
    return torch.zeros_like(x)

  sim_xp = gBAOAB_integrator(x,p,force, gs, h,M, gamma, k, int(random_t.item()),1,10**(-13))

  L , G = L_fn(sim_xp[:,:3]) # defining the projection matrix using only the position, not velocity
  # raise ValueError(L.shape,G.shape)

  score = model(sim_xp, random_t/300,L,G).cuda()


  # just defining this in order to get the jacobian
  def _model(sim_xp):
    velocity = model(sim_xp, random_t/300,L,G)
    position = torch.zeros_like(velocity)
    xp = torch.cat([velocity, position], dim = -1)
    return xp.sum(dim =0)

  jac = torch.autograd.functional.jacobian(_model, inputs=sim_xp, create_graph=True,strict=False).permute(1,0,2)
  # not sure if I should have both inputs here, but I do think it's right... Euclidean divergence
  jac_trace = jac.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
  loss = 1/2 * torch.linalg.norm(score, axis=(-2, -1))**2 + jac_trace

  return loss.mean(), score, sim_xp

my_dataset = torch.load('fire_qps.pth').float()


####### Main training loop #######
lr=1e-4 
batch_size =  512 
dataloader = DataLoader(my_dataset,batch_size=batch_size, shuffle=True)
n_epochs = 100
tqdm_epoch = nbk.trange(n_epochs)
optimizer = Adam(score_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

root_dir = args.root
i = 0
epoch_losses =[]
corrs = []
epoch_no = 0
for epoch in tqdm_epoch:
    t_dl =nbk.tqdm(dataloader)
    avg_loss = 0.
    num_items = 0
    for pw in t_dl:
        x = pw.cuda()
        i += 1
        loss_val,score,sim_xp = loss(score_model, x)
        t_dl.set_description(f"Loss = {loss_val.item()}")
        loss_val.backward()

        # Gradient accumulation so that we have multiple times in each gradient step
        if i % 10 == 0:
          optimizer.step()
          optimizer.zero_grad()

        avg_loss += loss_val.item() * x.shape[0]
        num_items += x.shape[0]

    epoch_losses.append(avg_loss / num_items)
    torch.save(score_model.state_dict(), os.path.join(args.root,f'model_ep{epoch}.pth'))
    torch.save(epoch_losses, os.path.join('proper-fire-training/epoch-losses-training-larger.pth'))
    tqdm_epoch.set_description(f"Average Loss: {avg_loss / num_items}")
    if epoch % 10 == 0:
        scheduler.step()