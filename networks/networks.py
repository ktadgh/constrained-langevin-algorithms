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

