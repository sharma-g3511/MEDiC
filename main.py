import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
from hyperparameters import n_epoch, batch_size, n_T, device, n_feat, lrate, save_model, samples_per_class, exp_classes



class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            return x1


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*[ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)])

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=8):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask))
        c = c * context_mask
        
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        
        up2 = self.up1(up1, down2)
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  
        "oneover_sqrta": oneover_sqrta, 
        "sqrt_beta_t": sqrt_beta_t,  
        "alphabar_t": alphabar_t, 
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,  
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w = 0.0):

        x_i = torch.randn(n_sample, *size).to(device)
        c_i = torch.arange(0,10).to(device)
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        context_mask = torch.zeros_like(c_i).to(device)

        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. 

        x_i_store = []
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
    
    def sample_custom(self, n_sample, c_i, size, device, guide_w = 0.0):
        x_i = torch.randn(n_sample, *size).to(device)
        if not isinstance(c_i, torch.Tensor):
            c_i = torch.tensor(c_i)
        c_i = c_i.to(device)
        context_mask = torch.zeros_like(c_i).to(device)

        x_i_store = []
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample] 
            eps2 = eps[n_sample:]
            print(eps.shape)
            print(eps1.shape)
            print(eps2.shape)
            eps = (1+guide_w)*eps1
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


tf = transforms.Compose([transforms.ToTensor()])

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        data = np.load(str(data_dir/'data.npy')).astype('float32')
        self.data = np.zeros((len(data), 1, 8, 8), dtype='float32')
        for i in range(len(data)):
            self.data[i,0] = data[i].reshape((8, 8))
        self.MAX = self.data.max()
        self.MIN = self.data.min()
        self.data = np.interp(self.data, (self.MIN, self.MAX), (0, 1)).astype('float32')
        self.labels = np.load(str(data_dir/'labels.npy'))
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.data[idx], self.labels[idx])



def train_ddpm(exp, data_dir, n_classes, model_save_dir, model_name='default'):
    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = MyDataset(data_dir, tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.002), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    
    pbar = tqdm(range(n_epoch))
    for ep in pbar:
        ddpm.train()

        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        loss_ema = None
        for x, c in dataloader:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            optim.step()
        pbar.set_description(f"loss: {loss_ema:.4f}")
    
    torch.save(ddpm.state_dict(), str(model_save_dir/f"{model_name}.pth"))
    
    
    
exps = ['emb_1']
prefix_dir = Path('new')
model_save_dirs = list(map(lambda exp: prefix_dir/Path('model')/exp, exps))
data_dirs = list(map(lambda exp: prefix_dir/Path('data')/exp, exps))
output_dirs = list(map(lambda exp: prefix_dir/Path('output')/exp, exps))


for exp, data_dir, n_classes, model_save_dir in zip(exps, data_dirs, exp_classes, model_save_dirs):
    print(f'Training for {exp}')
    train_ddpm(exp, data_dir, n_classes, model_save_dir)

######################################################################################################################################
#Inference

def generate_samples(ddpm, class_labels, MIN, MAX):
    n = len(class_labels)
    all_samples = []
    step = 100
    for i in tqdm(range(0, n, step)):
        labels = class_labels[i:i+step]
        samples = None
        with torch.no_grad():
            samples = ddpm.sample_custom(len(labels), labels, (1, 28, 28), device, guide_w=2)
        samples = np.interp(np.moveaxis(samples[0].cpu().numpy(), 1, -1), (0, 1), (MIN, MAX))
        flattened = np.zeros((len(samples), 28*28))
        for i in range(len(samples)):
            flattened[i] = samples[i].flatten()
        all_samples.append(flattened)
    return np.concatenate(all_samples)

def generate(exp, data_dir, n_classes, model_save_dir, output_dir, model_name='default'):
    output_dir = Path(output_dir)/model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_save_dir = Path(model_save_dir)
    
    dataset = MyDataset(data_dir, tf)
    MIN, MAX = dataset.MIN, dataset.MAX
    del dataset
    
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.002), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    
    ddpm.load_state_dict(torch.load(str(model_save_dir/f'{model_name}.pth')))
    
    num_samples = n_classes*samples_per_class
    class_labels = np.concatenate([np.full(samples_per_class, i) for i in range(n_classes)])
    samples = generate_samples(ddpm, class_labels, MIN, MAX)
    
    np.save(str(output_dir/'data.npy'), samples)
    np.save(str(output_dir/'labels.npy'), class_labels)
    
    
for exp, data_dir, n_classes, model_save_dir, output_dir in zip(exps, data_dirs, exp_classes, model_save_dirs, output_dirs):
    print(f'Generating for {exp}')
    generate(exp, data_dir, n_classes, model_save_dir, output_dir)