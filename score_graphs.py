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


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def G_(gs):
    '''
    each g in gs should act on batches, eg lambda x: (x[:,0] -  x[:,1])**2
    :param gs: a list of tensor functions
    :return: a function sending a tensor to the stacked matrix of the functions of that tensor
    '''
    def G_gs(tensor):
        x = tensor
        # raise ValueError(torch.stack([g(x) for g in gs], 1).shape)
        return torch.stack([g(x) for g in gs], 1)
    return G_gs


def J1(gs, x):
    '''Returns the Jacobian evaluated at x for a list gs of constraint functions'''
    jac_batched = jacobian(G(gs), x) # shape (fns, batch_size, batch_size, dims)

    r = jac_batched.permute(1, 3, 0, 2).diagonal(dim1=-2, dim2=-1).permute(2, 0, 1)
    return r


def J(gs,x):
  func = G_(gs)
  # raise ValueError(x, gs[0](x),gs[1](x), gs[2](x))
  # x in shape (Batch, Length)
  def _func_sum(x):
    return func(x).sum(dim=0)
  return jacobian(_func_sum, x, create_graph=False).permute(1,0,2)



def rattle_step(x, v1, h, M, gs, e):
    '''
    Defining a function to take a step in the position, velocity form.
    g should be a vector-valued function of constraints.
    :return: x_1, v_1
    '''

    # M1 =  torch.inverse(M) commenting this out since we use the identity

    G1 = G_(gs)


    DV = torch.zeros_like(x)
    batch_size = x.shape[0]
    DV_col = DV.reshape(batch_size,-1, 1)


    x_col = x.reshape(batch_size,-1, 1)
    v1_col = v1.reshape(batch_size,-1, 1)

    # doing Newton-Raphson iterations
    x2 = x_col + h * v1_col - 0.5*(h**2)* torch.bmm(M, DV_col)
    Q_col = x2
    Q = torch.squeeze(Q_col)
    J1 = J(gs, torch.squeeze(x_col))

    diff = torch.tensor([1.]).cuda()
    initial_q = Q_col
    initial_v = v1_col
    steps =0
    limit = ((Q_col**2).sum(dim=1).max() -1).abs()

    while not torch.allclose((Q_col**2).sum(dim=1), torch.tensor([1.]).cuda(), atol = limit):
        J2 = J(gs, torch.squeeze(Q))
        R = torch.bmm(torch.bmm(J2,M),J1.mT)
        dL = torch.bmm(torch.linalg.inv(R),G1(Q).unsqueeze(-1))
        diff = torch.bmm(torch.bmm(M,J1.mT), dL)
        Q= Q- diff.squeeze(-1)
        steps +=1

    # half step for velocity
    Q_col = Q.reshape(batch_size,-1,1)
    v1_half = (Q_col - x_col)/h
    x_col = Q_col
    J1 = J(gs, torch.squeeze(x_col))

    # getting the level
    J2 = J(gs, torch.squeeze(Q))
    P = torch.bmm(torch.bmm(J1, M),J1.mT)
    T = torch.bmm(J1, (2/h * v1_half - torch.bmm(M, DV_col)))

    #solving the linear system
    L = torch.linalg.solve(P,T)

    v1_col = v1_half - h/2 * DV_col - h/2 * torch.bmm(J2.mT,L)

    return torch.squeeze(x_col), torch.squeeze(v1_col)


def gBAOAB_step(q_init,p_init,F, gs, h,M, gamma, k, kr,e):
    # setting up variables
    M1 = M.cuda()
    batch_size = q_init.shape[0]
    R = torch.randn(batch_size, q_init.shape[1]).cuda()
    p = p_init
    q = q_init

    a2 = torch.exp(torch.tensor(-gamma*h))
    b2 = torch.sqrt(k*(1-a2**(2)))


    # doing the initial p-update
    J1 = J(gs,q)
    G = J1
    to_invert = torch.bmm(torch.bmm(G, M1), torch.transpose(G,-2,-1))

    t2 = torch.bmm(torch.inverse(to_invert), torch.bmm(G , M1))
    L1 = torch.eye(q_init.shape[1]).cuda() - torch.bmm(torch.transpose(G,-2,-1),t2)

    p = p - h/2 * torch.bmm(L1, F(q).unsqueeze(-1)).squeeze(-1)


    # doing the first RATTLE step
    for _ in range(kr):
      q, p = rattle_step(q, p, h/(2*kr), M, gs, e)


    # the second p-update - (O-step in BAOAB)
    J2 = J(gs,q)
    G = J2
    to_invert=torch.bmm(G, torch.bmm(M1,torch.transpose(G,-1,-2)))
    # raise ValueError(to_invert.diagonal(dim1=-2,dim2=-1))

    # raise ValueError(torch.bmm(G, M1).shape, torch.bmm(torch.transpose(G,-2,-1),torch.inverse(to_invert)).shape)
    L2 = torch.eye(q_init.shape[1]).cuda() - torch.bmm(torch.bmm(torch.transpose(G,-2,-1),torch.inverse(to_invert)), torch.bmm(G, M1))
    p = a2* p + b2* torch.bmm(torch.bmm(torch.bmm(M**(1/2),L2), M**(1/2)), R.unsqueeze(-1)).squeeze(-1)

    # doing the second RATTLE step
    for i in range(kr):
      q, p = rattle_step(q, p, h/(2*kr), M, gs, e)


    # the final p update
    J3= J(gs,q).cuda()
    G = J3

    qp = torch.cat([q,p], dim = 1)
    L3 = torch.eye(q_init.shape[1]).cuda() - torch.bmm(torch.bmm(torch.bmm(torch.transpose(G,-2,-1), torch.inverse(G@ M1@ torch.transpose(G,-2,-1))), G), M1)
    p = p -  h/2 * torch.bmm(L3, F(q).unsqueeze(-1)).squeeze(-1)

    return q, p


def gBAOAB_integrator(q_init,p_init,F, gs, h,M, gamma, k, steps,kr,e):
    q = q_init
    p = p_init

    for _ in range(steps):
      q, p = gBAOAB_step(q, p, F, gs, h,M, gamma, k, kr,e)
    return torch.cat([q, p], dim=1)



def gBAOAB_integrator_retain(q_init,p_init,F, gs, h,M, gamma, k, steps,kr,e):
    q = q_init
    p = p_init
    qs = []
    means = []

    def ac(x):
      # print(x[2], torch.arccos(x[2]))
      return (x[1])
    def get_mean(x):
      return torch.nanmean(torch.vmap(ac)(x))
    for _ in nbk.tqdm(range(steps)):
      q, p = gBAOAB_step(q, p, F, gs, h,M, gamma, k, kr,e)
      qs.append(q)
      means.append(get_mean(q).cpu())
      if len(means) % 500 == 0:
        plt.plot(np.convolve(means, np.ones(50)/50, mode='valid'))
        plt.show()
    return qs[-1000:],means

def cotangent_projection(gs):
    def proj(x):
        G = J(gs,x).cuda()

        M = torch.eye(G.size()[2]).cuda().broadcast_to(x.shape[0],G.size()[2],G.size()[2]).cuda()
        b1 = torch.bmm(G,M)
        b2 = G.mT
        to_invert = torch.bmm(b1,b2)
        L= torch.eye(G.size()[2]).cuda() - torch.bmm(torch.bmm(G.mT, torch.inverse(to_invert)) ,torch.bmm(G ,(M)))
        return L, G
    return proj

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

score_model = ScoreNet(30).cuda()

def create_constraints():
    constraint_fns = [lambda x : (x**2).sum(dim =1)-1]
    return constraint_fns
gs = create_constraints()

L_fn = cotangent_projection(gs)

# to ensure that we are noising enough - plots should look uniform for high t values
def graph_points(q,t, alpha=0.1):
  xs = q[:, 0]
  ys = q[:, 1]
  zs = q[:, 2]

  # Create a new figure
  fig = plt.figure()
  plt.title(f't = {t}')
  ax = fig.add_subplot(111, projection='3d')

  # Plot the data
  ax.scatter(xs.cpu(), ys.cpu(), zs.cpu(), alpha = alpha, s = 1)

  # Show the plot
  plt.show()



def make_velocity_graph(my_dataset, epoch, corrs):
    batch_size =  2 #@param {'type':'integer'}
    dataloader = DataLoader(my_dataset,batch_size=batch_size, shuffle=True)
    n_epochs = 20
    tqdm_epoch = nbk.trange(n_epochs)
    optimizer = Adam(score_model.parameters(), lr=lr)
    i = 0
    epoch_losses =[]
    count = 0
    
    sis = []
    sjs = []
    sks = []
    
    vis = []
    vjs = []
    vks = []
    
    for epoch in tqdm_epoch:
        t_dl =nbk.tqdm(dataloader)
        avg_loss = 0.
        num_items = 0
        for pw in t_dl:
            x = pw.cuda()
            i += 1
            loss_val,score, sim_xp = loss(score_model, x)

            sis.append(score[0][0].cpu().detach())
            sjs.append(score[0][1].cpu().detach())
            sks.append(score[0][2].cpu().detach())

            vis.append(sim_xp[0][3].cpu().detach())
            vjs.append(sim_xp[0][4].cpu().detach())
            vks.append(sim_xp[0][5].cpu().detach())
            count +=1
            if count % 5 == 0:
              print(count)
            if count >= 100:
              break

    # Assuming 'sis' and 'vis' are defined lists
    # Create a figure and three subplots arranged horizontally (1x3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Subplot 1
    axes[0].scatter([si.cpu().detach() for si in sis], vis)
    axes[0].set_xlabel('Score Function X')
    axes[0].set_ylabel('Velocity X')

    # Subplot 2 (repeat or modify as needed for different data)
    axes[1].scatter([si.cpu().detach() for si in sjs], vjs)
    axes[1].set_xlabel('Score Function Y')
    axes[1].set_ylabel('Velocity Y')

    # Subplot 3 (repeat or modify as needed for different data)
    axes[2].scatter([si.cpu().detach() for si in sks], vks)
    axes[2].set_xlabel('Score Function Z')
    axes[2].set_ylabel('Velocity Z')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.savefig('velocity_vs_score_'+str(epoch)+'.pdf')
    print('SAVED AT EPOCH',epoch)
    r1 = np.corrcoef(np.array([si.cpu().detach().item() for si in sks]), np.array(vks))[0,1]
    r2 = np.corrcoef(np.array([si.cpu().detach().item() for si in sjs]), np.array(vjs))[0,1]
    r3 = np.corrcoef(np.array([si.cpu().detach().item() for si in sis]), np.array(vis))[0,1]
    corrs.append((r1+r2+r3)/3.0)


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
  random_t = torch.round((torch.rand(1, device=x.device))*(300))

  # Noising

  M = torch.eye(x.shape[1]).broadcast_to(x.shape[0], x.shape[1],x.shape[1]).cuda()

  h = 0.05
  k = 0.001

  def force(x):
    return torch.zeros_like(x)

  sim_xp = gBAOAB_integrator(x,p,force, gs, h,M, 1, k, int(random_t.item()),1,10**(-13))
  t_o = int(random_t.item())
  if t_o % 100 == 0:
    graph_points(sim_xp, t_o)

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

my_dataset = torch.load('qps.pth')


## size of a mini-batch
## learning rate
lr=1e-3 #@param {'type':'number'}
batch_size =  512 #@param {'type':'integer'}


# score_model.load_state_dict(torch.load('/notebooks/model_checkpoint_ep5_step1100.pth'))
## learning rate
dataloader = DataLoader(my_dataset,batch_size=batch_size, shuffle=True)
n_epochs = 20
tqdm_epoch = nbk.trange(n_epochs)
optimizer = Adam(score_model.parameters(), lr=lr)
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
        if i % 5 == 0:
          optimizer.step()
          optimizer.zero_grad()
          # break
        avg_loss += loss_val.item() * x.shape[0]
        num_items += x.shape[0]
        if i%5 == 0:
          epoch_losses.append(avg_loss / num_items)
          # torch.save(score_model.state_dict(), f'model_checkpoint_ep{epoch}_step{i}.pth')
          torch.save(epoch_losses, 'epoch-losses-withskip-fixed.pth')
          make_velocity_graph(my_dataset,epoch_no, corrs)
          epoch_no +=1
          corrs_tensor = torch.tensor(corrs)
          torch.save(corrs_tensor, 'correlations-withskip-fixed.pth')

def graph_score(score_function, dim = 0, time = 0.1):
  # now apply fcolor to each
  phi = np.linspace(0, 2*np.pi, 100)
  theta = np.linspace(0, np.pi, 100)
  phi, theta = np.meshgrid(phi, theta)

  # The Cartesian coordinates of the unit sphere
  x = np.sin(theta) * np.cos(phi)
  y = np.sin(phi) * np.sin(theta)
  z = np.cos(theta)

  gs = create_constraints()
  L_fn = cotangent_projection(gs)

  fcolor = np.zeros(x.shape)
  for i in range(len(x)):
    for j in range(len(x[0])):
      x_ij = x[i][j]
      y_ij = y[i][j]
      z_ij = z[i][j]

      xyz = torch.tensor([x_ij,y_ij, z_ij,0.0,0.0,0.0]).unsqueeze(0).to(torch.float32).cuda()

      L , G = L_fn(xyz[:,:3]) # defining the projection matrix using only the position, not velocity
      sm = score_model(xyz, torch.tensor([0.1]).cuda(), L, G)

      fcolor[i][j] = sm[0,dim]

  return fcolor

def make_score_graph(score_model, dim = 0, t=0.1):

  fcolor = graph_score(score_model,dim,t)
  import matplotlib.pyplot as plt
  from matplotlib import cm, colors
  from mpl_toolkits.mplot3d import Axes3D
  import numpy as np
  from scipy.special import sph_harm
  from matplotlib.colors import Normalize


  # now apply fcolor to each
  phi = np.linspace(0, 2*np.pi, 100)
  theta = np.linspace(0, np.pi, 100)
  phi, theta = np.meshgrid(phi, theta)

  # The Cartesian coordinates of the unit sphere
  x = np.sin(theta) * np.cos(phi)
  y = np.sin(phi) * np.sin(theta)
  z = np.cos(theta)


  norm = Normalize(fcolor.min(), fcolor.max())
  fcolors= norm(fcolor)


  # Set the aspect ratio to 1 so our sphere looks spherical
  fig = plt.figure(figsize=(9,9))
  ax = fig.add_subplot(111, projection='3d')
  surface = ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.seismic(fcolors))
  ax.set_xticks([-1,-.75,-.5,-.25,0,.25,.5,.75,1])
  ax.set_zticks([-1,-.75,-.5,-.25,0,.25,.5,.75,1])
  plt.xlabel('x')
  plt.ylabel('y')
  colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.seismic), ax=ax)
    
gs = create_constraints()
L_fn = cotangent_projection(gs)

sis = []
sjs = []
sks = []

vis = []
vjs = []
vks = []
## size of a mini-batch
## learning rate
lr=1e-3 #@param {'type':'number'}


