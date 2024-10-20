import torch
from torch import nn
from simulation_functions.gbaoab_functions import create_constraints, cotangent_projection, gBAOAB_integrator

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False).cuda()

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)



class ScoreNet(nn.Module):
  """A time-dependent score-based model."""
  def __init__(self, embed_dim):
    super().__init__()
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),nn.Linear(embed_dim, embed_dim))
    self.lin_embed = nn.Linear(embed_dim,embed_dim)
    self.lin_embed2 = nn.Linear(embed_dim,embed_dim)
    self.lin1 = nn.Linear(6,embed_dim)
    self.lin2 = nn.Linear(embed_dim,embed_dim)
    self.lin3 = nn.Linear(embed_dim,embed_dim)
    self.lin4 = nn.Linear(embed_dim,embed_dim)
    self.lin5 = nn.Linear(embed_dim,3)
    self.act = torch.nn.Sigmoid()

  #@torch.compile(mode="default")
  def forward(self,x,t,L,G):
      p = x[:,3:]
      # setting the fixed points of x
      embed = self.act(self.embed(t))
      h = self.lin1(x)
      h = h+ self.lin_embed(embed)
      h = self.act(self.lin2(h))
      h = self.act(self.lin3(h))+ self.lin_embed2(embed)
      h = self.act(self.lin4(h))
      h = self.lin5(h)

      # projection
      p = torch.squeeze(L@ torch.unsqueeze(h-p,-1),-1)
      return p


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

  h = 0.05
  k = 0.001
  gamma = 1

  def force(x):
    return torch.zeros_like(x)

  sim_xp = gBAOAB_integrator(x,p,force, gs, h,M, gamma, k, int(random_t.item()),1,10**(-13))
  t_o = int(random_t.item())
  L , G = L_fn(sim_xp[:,:3]) # defining the projection matrix using only the position, not velocity
  # raise ValueError(L.shape,G.shape)

  score = model(sim_xp, random_t/300,L,G).cuda()
