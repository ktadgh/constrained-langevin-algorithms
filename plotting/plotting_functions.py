import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import tqdm.notebook as nbk
from networks import loss

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
    n_epochs = 1
    tqdm_epoch = nbk.trange(n_epochs)
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

