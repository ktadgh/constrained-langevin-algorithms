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


def J(gs,x):
  func = G_(gs)
  def _func_sum(x):
    return func(x).sum(dim=0)
  return jacobian(_func_sum, x, create_graph=False).permute(1,0,2)
