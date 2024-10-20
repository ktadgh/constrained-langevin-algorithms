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

import os

import sympy as sp
x,y,z =sp.symbols('x,y,z', real=True)

f = sp.Matrix([[x**2 + y**2 + z**2 -1]])
j = f.jacobian([x,y,z])
to_invert = f.jacobian([x,y,z]).multiply(f.jacobian([x,y,z]).T)
projector1 = sp.eye(3) - f.jacobian([x,y,z]).T @ to_invert.inv() @ f.jacobian([x,y,z])

fun = sp.lambdify((x,y,z),projector1, "numpy") 

def torchfun(x):
  return fun(x[0],x[1],x[2])

def projector(xs):
  return np.apply_along_axis(torchfun, axis=1, arr=xs)

j = f.jacobian([x,y,z])

j_fun = sp.lambdify((x,y,z),j, "numpy") 

def j_torchfun(x):
  return fun(x[0],x[1],x[2])

def J(xs):
  return np.apply_along_axis(torchfun, axis=1, arr=xs)



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


# def J(gs,x):
#   func = G_(gs)
#   def _func_sum(x):
#     return func(x).sum(dim=0)
#   return jacobian(_func_sum, x, create_graph=False).permute(1,0,2)





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
    J1 = torch.tensor(J(torch.squeeze(x_col).detach().cpu())).cuda()

    diff = torch.tensor([1.]).cuda()
    steps =0

    for _ in range(4):
        J2 = torch.tensor(J(torch.squeeze(Q).detach().cpu())).cuda()
        R = torch.bmm(torch.bmm(J2,M),J1.mT)
        dL = torch.bmm(torch.linalg.inv(R),G1(Q).unsqueeze(-1))
        diff = torch.bmm(torch.bmm(M,J1.mT), dL)
        Q= Q- diff.squeeze(-1)
        steps +=1

    # half step for velocity
    Q_col = Q.reshape(batch_size,-1,1)
    v1_half = (Q_col - x_col)/h
    x_col = Q_col
    J1 = torch.tensor(J(torch.squeeze(x_col).detach().cpu())).cuda()

    # getting the level
    J2 = torch.tensor(J(torch.squeeze(Q).detach().cpu())).cuda()
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
    J1 = torch.tensor(J(torch.squeeze(q).detach().cpu())).cuda()
    G = J1
    to_invert = torch.bmm(torch.bmm(G, M1), torch.transpose(G,-2,-1))

    t2 = torch.bmm(torch.inverse(to_invert), torch.bmm(G , M1))
    L1 = torch.eye(q_init.shape[1]).cuda() - torch.bmm(torch.transpose(G,-2,-1),t2)

    p = p - h/2 * torch.bmm(L1, F(q).unsqueeze(-1)).squeeze(-1)


    # doing the first RATTLE step
    for _ in range(kr):
      q, p = rattle_step(q, p, h/(2*kr), M, gs, e)


    # the second p-update - (O-step in BAOAB)
    J2 = torch.tensor(J(torch.squeeze(q).detach().cpu())).cuda()
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
    J3= torch.tensor(J(torch.squeeze(q).detach().cpu())).cuda()
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
        G = torch.tensor(J(torch.squeeze(x).detach().cpu())).cuda()
        M = torch.eye(G.size()[2]).cuda().broadcast_to(x.shape[0],G.size()[2],G.size()[2]).cuda()
        b1 = torch.bmm(G,M)
        b2 = G.mT
        to_invert = torch.bmm(b1,b2)
        L= torch.eye(G.size()[2]).cuda() - torch.bmm(torch.bmm(G.mT, torch.inverse(to_invert)) ,torch.bmm(G ,(M)))
        return L, G
    return proj


def create_constraints():
    constraint_fns = [lambda x : (x**2).sum(dim =1)-1]
    return constraint_fns




###### REVERSAL FUNCTIONS ######
    
import sympy as sp
x,y,z =sp.symbols('x,y,z', real=True)

f = sp.Matrix([[x**2 + y**2 + z**2 -1]])
j = f.jacobian([x,y,z])
to_invert = f.jacobian([x,y,z]).multiply(f.jacobian([x,y,z]).T)
projector1 = sp.eye(3) - f.jacobian([x,y,z]).T @ to_invert.inv() @ f.jacobian([x,y,z])

pjs = []
for i in range(0,projector1.shape[0]):
  pjs.append(projector1[i,:].jacobian([x,y,z]).trace())
traces_matrix = sp.Matrix(pjs)

fun = sp.lambdify((x,y,z),traces_matrix, "numpy") 

def torchfun(x):
  return fun(x[0],x[1],x[2])

def divergence(xs):
  return np.apply_along_axis(torchfun, axis=1, arr=xs)


def reverse_gBAOAB_step(q_init,p_init,score, gs, h,M, gamma, k, kr,e):
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

    qp = torch.cat([q,p], dim = 1)
    qc = q.detach().cpu().numpy()
    d = torch.tensor(divergence(qc), device='cuda').squeeze().float()

    p = p + torch.bmm(L1, 2*gamma*d.unsqueeze(-1)).squeeze(-1)*(h/2)


    # doing the first RATTLE step - changing p-update to negative
    minusp = -p
    for _ in range(kr):
      q, p = rattle_step(q, minusp, h/(2*kr), M, gs, e)
      minusp = -p
        
    # the second p-update - (O-step in BAOAB), with 1 euler-maruyama step
    J2 = J(gs,q)
    G = J2
    to_invert=torch.bmm(G, torch.bmm(M1,torch.transpose(G,-1,-2)))
    L2 = torch.eye(q_init.shape[1]).cuda() - torch.bmm(torch.bmm(torch.transpose(G,-2,-1),torch.inverse(to_invert)), torch.bmm(G, M1))
    qp = torch.cat([q,p], dim = 1)
    R = torch.randn(batch_size, q_init.shape[1]).cuda()*torch.sqrt(torch.tensor([2*h])).cuda()
    p = p + torch.bmm(L2, gamma*p.unsqueeze(-1)).squeeze(-1)*h + torch.bmm(L2, 2 * gamma * score(qp).unsqueeze(-1)).squeeze(-1)*h + torch.bmm(L2,math.sqrt(k*2*gamma)*R.unsqueeze(-1)).squeeze(-1)

    # doing the second RATTLE step
    minusp = -p
    for i in range(kr):
      q, p = rattle_step(q, -p, h/(2*kr), M, gs, e)
      minusp = -p

    # the final p update
    J3= J(gs,q).cuda()
    G = J3
    qc = q.detach().cpu().numpy()
    d = torch.tensor(divergence(qc), device='cuda').squeeze().float()
    qp = torch.cat([q,p], dim = 1)
    L3 = torch.eye(q_init.shape[1]).cuda() - torch.bmm(torch.bmm(torch.bmm(torch.transpose(G,-2,-1), torch.inverse(G@ M1@ torch.transpose(G,-2,-1))), G), M1)
    p = p + torch.bmm(L3, 2*gamma*d.unsqueeze(-1)).squeeze(-1)*(h/2)

    return q, p


def reverse_gBAOAB_integrator(q_init,p_init,score_model, gs, h,M, gamma, k, steps,kr,e):
    # reverse force should just be the score function
    q = q_init
    p = p_init
    qs = []
    for i in nbk.tqdm(range(steps)):
      t = torch.tensor([300 - (i/steps)*300]).cuda()
      L_fn = cotangent_projection(gs)
      M = torch.eye(q.shape[1]).broadcast_to(q.shape[0], q.shape[1],q.shape[1]).cuda()

      def proj_score(x):
        L , G = L_fn(x[:,:3]) # defining the projection matrix using only the position, not velocity
        with torch.no_grad():
          x= score_model(x, t/300, L, G)
        return x.detach()
      
      q, p = reverse_gBAOAB_step(q, p, proj_score, gs, h,M, gamma, k, kr,e)

      qs.append(q)

    return qs
