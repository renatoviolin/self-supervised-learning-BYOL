# %%
try:
    get_ipython()
    NUM_WORKERS = 0
except:
    NUM_WORKERS = 8

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from model import BYOL, EMA
import dataset
from tqdm import tqdm
import glob
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
CKPT = 'byol.pt'
EPOCHS = 50


# %%
data = glob.glob('data/no_label/*.jpg')[:10000]
train_dataset = dataset.BYOLDataset(data)
train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)


# %%
byol_model = BYOL().to(DEVICE)
max_steps = len(train_dataloader)
ema = EMA(max_steps, tau=0.99)
optimizer = torch.optim.Adam(byol_model.parameters(), lr=1e-4)
ema = EMA(max_steps)


# %%
for epoch in range(EPOCHS):
    print(f'Epoch {epoch+1}')
    t_loss = []
    t_loader = tqdm(train_dataloader)
    for batch in t_loader:
        optimizer.zero_grad()
        img_1, img_2 = batch[0].to(DEVICE), batch[1].to(DEVICE)
        _, _, loss = byol_model(img_1, img_2)

        loss.backward()
        optimizer.step()
        ema(byol_model.online_network, byol_model.target_network)

        l = loss.detach().cpu().numpy().item()
        t_loss.append(l)
        t_loader.set_description(f'loss: {l:.4f}')

    print(f"train loss: {np.mean(t_loss):.4f}")

    if (epoch % 5) == 0:
        torch.save(byol_model.online_network, CKPT)

torch.save(byol_model.online_network, CKPT)
