import torch
from torch import nn
from torchvision import models
from copy import deepcopy
import torch.nn.functional as F
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EMA():
    def __init__(self, max_steps, tau=0.99):
        self.initial_tau = tau
        self.current_tau = tau
        self.max_steps = max_steps

    def update_tau(self, global_step):
        self.current_tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * global_step / self.max_steps) + 1) / 2
        print(self.current_tau)

    def __call__(self, online_net, target_net):
        for (name, online_p), (_, target_p) in zip(online_net.named_parameters(), target_net.named_parameters()):
            if 'weight' in name:
                target_p.data = self.current_tau * target_p.data + (1 - self.current_tau) * online_p.data


class MLP(nn.Module):
    def __init__(self, input_dim=512, hidden_size=256, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SiameseArm(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(pretrained=False)
        del m.fc
        self.encoder = torch.nn.Sequential(
            *m.children(),
            nn.Flatten()
        )
        self.projector = MLP(input_dim=512, hidden_size=4096)
        self.predictor = MLP(input_dim=256, hidden_size=4096)

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h

    def get_embeddings(self, x):
        with torch.no_grad():
            return self.encoder(x)


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class BYOL(nn.Module):
    def __init__(self):
        super().__init__()
        self.online_network = SiameseArm()
        self.target_network = deepcopy(self.online_network)

    def forward(self, img_1, img_2):
        # Image 1 to image 2 loss
        _, _, h1 = self.online_network(img_1)
        with torch.no_grad():
            _, z2, _ = self.target_network(img_2)
        loss_a = loss_fn(h1, z2)

        # Image 2 to image 1 loss
        _, _, h1 = self.online_network(img_2)
        with torch.no_grad():
            _, z2, _ = self.target_network(img_1)
        loss_b = loss_fn(h1, z2)

        total_loss = (loss_a + loss_b).mean()
        return loss_a, loss_b, total_loss


class ClassifierBYOL(nn.Module):
    def __init__(self, checkpoint):
        super(ClassifierBYOL, self).__init__()

        self.backbone = torch.load(checkpoint, map_location=DEVICE)
        # for p in self.backbone.parameters():
        #     p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.backbone(x)[0]
        return self.classifier(x)


class ClassifierScratch(nn.Module):
    def __init__(self):
        super(ClassifierScratch, self).__init__()

        _m = models.resnet18(pretrained=False)
        del _m.fc
        self.backbone = torch.nn.Sequential(
            *_m.children(),
            nn.Flatten()
        )

        # for p in self.backbone.parameters():
        #     p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


# %%
# inp = torch.rand(2, 3, 224, 224)
# batch_size = 32
# len_dataloader = 50
# max_steps = len_dataloader


# %%
# byol = BYOL()
# a, b, c = byol(inp, inp)

# optimizer = torch.optim.Adam(byol.parameters())
# c.backward()
# optimizer.step()


# #%%
# enc_model = deepcopy(byol.online_network)
# enc_model.eval()
# enc_model.get_emb(inp).shape


# #%%
# ema = EMA(max_steps)
# ema(byol.online_network, byol.target_network)

# print(next(iter(byol.online_network.projector.parameters()))[:10, 0].data)
# print(next(iter(byol.target_network.projector.parameters()))[:10, 0].data)
