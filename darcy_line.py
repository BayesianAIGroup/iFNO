import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from neuralop.layers.fno_block import FNOBlocks
import argparse
import random
import os
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import time
from utils import *
from logger import get_logger
from vanilla_vae import *


class IFNO(nn.Module):
    def __init__(self, modes1, modes2, width, beta):
        super(IFNO, self).__init__()
        self.beta = beta
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = args.padding
        self.p1 = nn.Linear(3, self.width)
        self.p2 = nn.Linear(3, self.width)

        self.q1 = MLP(self.width, 3, self.width * 4)
        self.q2 = MLP(self.width, 3, self.width * 4)

        self.width = int(self.width / 2)
        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.ws = nn.ModuleList()

        for _ in range(2 * args.n_layers):
            self.convs.append(FNOBlocks(self.width, self.width, (self.modes1, self.modes2)))
            self.mlps.append(MLP(self.width, self.width, self.width))
            self.ws.append(nn.Conv2d(self.width, self.width, 1))

        self.vae_net = VanillaVAE(in_channels=1, latent_dim=args.rank)

    def VAE_train(self, x, y, return_mu=False):
        size = x.shape[0]
        kl_scale = args.kl
        recon_img, _, mu, log_var = self.vae_net.forward(x[:, :1, :, :])
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        myloss = LpLoss(size_average=False)
        unnorm_img = x_normalizer.decode(y.permute(0, 2, 3, 1).clone())
        unnorm_recon_img = x_normalizer.decode(
            torch.cat(
                (
                    recon_img.permute(0, 2, 3, 1).clone(),
                    y.permute(0, 2, 3, 1).clone()[:, :, :, 1:],
                ),
                axis=-1,
            )
        )
        mse_loss = myloss(unnorm_recon_img.reshape(size, -1), unnorm_img.reshape(size, -1))
        loss = kl_scale * kl_loss + mse_loss
        if return_mu:
            return loss, recon_img
        return loss

    def sp(self, x):
        sp = torch.nn.Softplus(beta=self.beta)
        return sp(x)

    def forward(self, x):
        nchannel = x.shape[-1]
        x = x.reshape(x.shape[0], s, s, nchannel)
        input_x = x
        x = self.p1(x)
        x = x.permute(0, 3, 1, 2)
        x_recon = self.q2(x).permute(0, 2, 3, 1)
        loss_recon = ((x_recon - input_x) ** 2).mean()

        u1 = x[:, :awidth, :, :]
        u2 = x[:, awidth:, :, :]

        for i in range(args.n_layers):
            u2_pad = F.pad(u2, [0, self.padding, 0, self.padding])
            x1 = self.mlps[2 * i](self.convs[2 * i](u2_pad))
            x2 = self.ws[2 * i](u2_pad)
            s2 = F.gelu(x1 + x2)
            s2 = s2[..., : (s2.size(-2) - self.padding), : (s2.size(-1) - self.padding)]

            v1 = u1 * self.sp(s2)
            v1_pad = F.pad(v1, [0, self.padding, 0, self.padding])
            x1 = self.mlps[2 * i + 1](self.convs[2 * i + 1](v1_pad))
            x2 = self.ws[2 * i + 1](v1_pad)
            s1 = F.gelu(x1 + x2)
            s1 = s1[..., : (s1.size(-2) - self.padding), : (s1.size(-1) - self.padding)]
            v2 = u2 * self.sp(s1)

            u1 = v1
            u2 = v2

        x = torch.cat((u1, u2), axis=1)
        y_pred = self.q1(x)
        y_pred = y_pred.permute(0, 2, 3, 1)

        return y_pred, loss_recon

    def backward(self, y):
        nchannels = y.shape[-1]
        y = y.reshape(y.shape[0], s, s, nchannels)
        input_y = y
        v = self.p2(y)
        v = v.permute(0, 3, 1, 2)

        y_recon = self.q1(v).permute(0, 2, 3, 1)
        loss_recon = ((y_recon - input_y) ** 2).mean()

        v1 = v[:, :awidth, :, :]
        v2 = v[:, awidth:, :, :]

        for i in range(args.n_layers - 1, -1, -1):
            v1_pad = F.pad(v1, [0, self.padding, 0, self.padding])
            x1 = self.mlps[2 * i + 1](self.convs[2 * i + 1](v1_pad))
            x2 = self.ws[2 * i + 1](v1_pad)
            s1 = F.gelu(x1 + x2)
            s1 = s1[..., : (s1.size(-2) - self.padding), : (s1.size(-1) - self.padding)]

            u2 = v2 * self.sp(s1) ** (-1)
            u2_pad = F.pad(u2, [0, self.padding, 0, self.padding])
            x1 = self.mlps[2 * i](self.convs[2 * i](u2_pad))
            x2 = self.ws[2 * i](u2_pad)
            s2 = F.gelu(x1 + x2)
            s2 = s2[..., : (s2.size(-2) - self.padding), : (s2.size(-1) - self.padding)]

            u1 = v1 * self.sp(s2) ** (-1)

            v1 = u1
            v2 = u2

        x = torch.cat((v1, v2), axis=1)
        x_preds = self.q2(x)

        x_preds = x_preds.permute(0, 2, 3, 1)

        return x_preds, loss_recon


dataset = "[Darcy-Flow-Line]"
parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, help="Rank of VAE", default=24)
parser.add_argument("--padding", type=int, help="padding num for FNO", default=20)
parser.add_argument("--nl", type=float, help="noise level", default=0.0)
parser.add_argument("--lr-VAE", type=float, help="learning rate for VAE Training", default=0.001)
parser.add_argument("--lr-IFNO", type=float, help="learning rate for IFNO Training", default=0.00025)
parser.add_argument("--lr-forward", type=float, help="learning rate for joint Training-forward", default=0.000005)
parser.add_argument("--lr-backward", type=float, help="learning rate for joint Training-backward", default=0.000005)
parser.add_argument("--kl", type=float, help="KL divergence weight in VAE loss", default=0.01)
parser.add_argument("--modes", type=int, help="number of modes in IFNO", default=16)
parser.add_argument("--epochs-VAE", type=int, help="epochs setting for VAE", default=2)
parser.add_argument("--epochs-IFNO", type=int, help="epochs setting for IFNO", default=2)
parser.add_argument("--epochs", type=int, help="epochs setting for joint training", default=2)
parser.add_argument("--n-train", type=int, help="num of train dataset", default=300)
parser.add_argument("--n-valid", type=int, help="num of valid dataset", default=100)
parser.add_argument("--n-test", type=int, help="num of test dataset", default=500)
parser.add_argument("--batchsize", type=int, help="num of batchsize for training", default=10)
parser.add_argument("--batchsize2", type=int, help="num of batchsize for vae training", default=100)
parser.add_argument("--batchsize3", type=int, help="num of batchsize for testing", default=20)
parser.add_argument("--valid", action="store_true", help="wheter or not do validation process")
parser.add_argument("--n-layers", type=int, default=4, metavar="N", help="data resolution (default: 4)")
parser.add_argument("--hidden", type=int, help="dimension of hidden layer in IFNO", default=64)
parser.add_argument("--beta", type=float, help="beta in softplus", default=2.0)
parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed (default: 0)")
parser.add_argument("--count", type=int, help="number of test", default=100)
args = parser.parse_args()
hyperparams = {
    "rank": args.rank,
    "noise-level": args.nl,
    "KL-weight": args.kl,
    "modes": args.modes,
    "hidden-dimension": args.hidden,
    "padding": args.padding,
    "beta": args.beta,
    "lr-VAE": args.lr_VAE,
    "lr-IFNO": args.lr_IFNO,
    "lr-forward": args.lr_forward,
    "lr-backward": args.lr_backward,
    "ntrain": args.n_train,
    "nvalid": args.n_valid,
    "ntest": args.n_test,
    "bz": args.batchsize,
    "bz2": args.batchsize2,
    "bz3": args.batchsize3,
    "valid": args.valid,
    "seed": args.seed,
}
exp_name = f"dataset_{dataset}/nl{args.nl}/s{args.seed}"

log_path = os.path.join("logger_grid_greedy", exp_name)

if not os.path.exists(log_path):
    os.makedirs(log_path)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

logger = None
logger = get_logger(os.path.join(log_path, "exp-" + str(args.count) + ".log"))
logger.info("======Settings======")
logger.info(f"  dataset:  {dataset}\n")
for key, value in hyperparams.items():
    logger.info(f"  {key}:  {value}\n")


epochs_VAE = args.epochs_VAE
epochs_IFNO = args.epochs_IFNO
ntrain = args.n_train
ntest = args.n_test
s = 64
modes = args.modes
width = args.hidden
awidth = int(width / 2)
batch_size = args.batchsize
batch_size2 = args.batchsize2
batch_size3 = args.batchsize3
epochs = args.epochs
beta = args.beta


y_no_noise = np.load("./data/DF_line_u.npy")
x_no_noise = np.load("./data/DF_line_k.npy")
logger.info("    data shape, f mean, u mean")
logger.info("   " + str(x_no_noise.shape) + " " + str(x_no_noise.mean()) + " " + str(y_no_noise.mean()))

if args.nl == 0.0:
    logger.info("   load no noise data")
    y = np.load("./data/DF_line_u.npy")
    x = np.load("./data/DF_line_k.npy")
    extra_f = np.load("./data/DF_line_k.npy")[:ntrain, :, :]
elif args.nl == 0.1:
    logger.info("   load 0.1 noise data")
    y = np.load("./data/DF_line_u_01.npy")
    x = np.load("./data/DF_line_k_01.npy")
    extra_f = np.load("./data/DF_line_k_01.npy")[:ntrain, :, :]
elif args.nl == 0.2:
    logger.info("   load 0.2 noise data")
    y = np.load("./data/DF_line_u_02.npy")
    x = np.load("./data/DF_line_k_02.npy")
    extra_f = np.load("./data/DF_line_k_02.npy")[:ntrain, :, :]


xtr = x[:ntrain, :, :]
ytr = y[:ntrain, :, :]
if args.valid:
    xte = xtr[-args.n_valid :, :, :]
    yte = ytr[-args.n_valid :, :, :]
    xtr = xtr[: ntrain - args.n_valid, :, :]
    ytr = ytr[: ntrain - args.n_valid, :, :]
    xte_no_noise = x_no_noise[ntrain - args.n_valid : ntrain, :, :]
    yte_no_noise = y_no_noise[ntrain - args.n_valid : ntrain, :, :]
    ntrain -= args.n_valid
    ntest = args.n_valid
else:
    xte = x[-ntest:, :, :]
    yte = y[-ntest:, :, :]
    xte_no_noise = x_no_noise[-ntest:, :, :]
    yte_no_noise = y_no_noise[-ntest:, :, :]


xtr2 = np.concatenate(
    (
        extra_f,
        extra_f.transpose(0, 2, 1),
        np.flip(extra_f, (0, 2)),
        np.flip(extra_f.transpose(0, 2, 1), (0, 2)),
    ),
    axis=0,
)


xtr = torch.tensor(xtr, dtype=torch.float32)
ytr = torch.tensor(ytr, dtype=torch.float32)
xtr2 = torch.tensor(xtr2, dtype=torch.float32)
xte = torch.tensor(xte, dtype=torch.float32)
yte = torch.tensor(yte, dtype=torch.float32)
xte_no_noise = torch.tensor(xte_no_noise, dtype=torch.float32)
yte_no_noise = torch.tensor(yte_no_noise, dtype=torch.float32)


xtr = xtr.to("cuda")
ytr = ytr.to("cuda")
xtr2 = xtr2.to("cuda")
xte = xte.to("cuda")
yte = yte.to("cuda")
xte_no_noise = xte_no_noise.to("cuda")
yte_no_noise = yte_no_noise.to("cuda")
x = np.linspace(0, 1, 64)
y = np.linspace(0, 1, 64)

X, Y = np.meshgrid(x, y)
X = X.T
Y = Y.T
grid = torch.tensor(
    np.concatenate((X[:, :, None], Y[:, :, None]), axis=-1),
    dtype=torch.float32,
    device="cuda",
)

xtr = torch.cat((xtr[:, :, :, None], grid[None, :, :, :].repeat(xtr.shape[0], 1, 1, 1)), axis=-1)
xtr2 = torch.cat((xtr2[:, :, :, None], grid[None, :, :, :].repeat(xtr2.shape[0], 1, 1, 1)), axis=-1)
xte = torch.cat((xte[:, :, :, None], grid[None, :, :, :].repeat(xte.shape[0], 1, 1, 1)), axis=-1)
xte_no_noise = torch.cat(
    (
        xte_no_noise[:, :, :, None],
        grid[None, :, :, :].repeat(xte_no_noise.shape[0], 1, 1, 1),
    ),
    axis=-1,
)
ytr = torch.cat((ytr[:, :, :, None], grid[None, :, :, :].repeat(ytr.shape[0], 1, 1, 1)), axis=-1)
yte = torch.cat((yte[:, :, :, None], grid[None, :, :, :].repeat(yte.shape[0], 1, 1, 1)), axis=-1)
yte_no_noise = torch.cat(
    (
        yte_no_noise[:, :, :, None],
        grid[None, :, :, :].repeat(yte_no_noise.shape[0], 1, 1, 1),
    ),
    axis=-1,
)

x_normalizer = UnitGaussianNormalizer(xtr)
xtr = x_normalizer.encode(xtr)
xtr2 = x_normalizer.encode(xtr2)
xte = x_normalizer.encode(xte)
xte_no_noise = x_normalizer.encode(xte_no_noise)
y_normalizer = UnitGaussianNormalizer(ytr)
ytr = y_normalizer.encode(ytr)
yte = y_normalizer.encode(yte)
yte_no_noise = y_normalizer.encode(yte_no_noise)
train_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtr2), batch_size=batch_size3, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(xte, yte, xte_no_noise, yte_no_noise),
    batch_size=batch_size2,
    shuffle=False,
)
myloss = LpLoss(size_average=False)
x_normalizer.cuda()
y_normalizer.cuda()


start_time = time.time()


def train():
    model = IFNO(modes, modes, width, beta).cuda()
    logger.info(f"  number of parameters:  {count_model_params(model)}\n")

    def pretrain_VAE():
        logger.info(f"  start pretraining")
        logger.info(f"  start VAE pretraining")
        min_err = 1.0
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_VAE)
        sc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=10, verbose=True)
        for ep in tqdm(range(epochs_VAE)):
            model.train()
            train_l2 = 0
            for (x,) in train_loader2:
                x = x.cuda()
                optimizer.zero_grad()
                loss = model.VAE_train(x.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2))
                loss.backward(retain_graph=False)
                clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                train_l2 += loss.item()
            sc.step(train_l2)

            if ep % 10 == 0:
                model.eval()
                test_l2 = 0.0
                with torch.no_grad():
                    for x, y, x_no_noise, y_no_noise in test_loader:
                        x, y, x_no_noise, y_no_noise = (
                            x.cuda(),
                            y.cuda(),
                            x_no_noise.cuda(),
                            y_no_noise.cuda(),
                        )
                        batchS = x.shape[0]
                        recon_img, _, _, _ = model.vae_net.forward2(x_no_noise.permute(0, 3, 1, 2)[:, :1, :, :])
                        unnorm_img = x_normalizer.decode(x_no_noise)
                        unnorm_recon_img = x_normalizer.decode(
                            torch.cat(
                                (
                                    recon_img.permute(0, 2, 3, 1).clone(),
                                    x_no_noise[:, :, :, 1:],
                                ),
                                axis=-1,
                            )
                        )
                        diff = unnorm_recon_img[:, :, :, 0].reshape(batchS, -1) - unnorm_img[:, :, :, 0].reshape(
                            batchS, -1
                        )
                        test_l2 += (
                            ((diff**2).sum(1) / ((unnorm_img[:, :, :, 0].reshape(batchS, -1) ** 2).sum(1))) ** 0.5
                        ).sum()
                test_l2 /= ntest
                train_l2 /= ntrain
                if test_l2 < min_err:
                    min_err = test_l2
                logger.info(
                    f"  Epoch [{ep + 1}/{epochs_VAE}], Training Loss: {train_l2:.4f}, Test Loss: {test_l2:.4f},  Min. Test Error: {min_err:.4f}"
                )

    def pretrain_IFNO():
        logger.info(f"  start IFNO pretraining")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_IFNO)
        min_err_forward = 1.0
        min_err_backward = 1.0
        pretrain_preds = None
        pretrain_gt = None

        for ep in tqdm(range(epochs_IFNO)):
            model.train()
            train_l2_forward = 0
            train_l2_backward = 0
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                batchS = x.shape[0]
                pred_y, loss_recon = model(x)
                pred_y = pred_y.reshape(batchS, s, s, 3)
                pred_y = y_normalizer.decode(pred_y.clone())
                y_true = y_normalizer.decode(y.clone())
                loss_forward = (
                    myloss(pred_y[:, :, :, 0], y_true[:, :, :, 0])
                    + myloss(pred_y[:, :, :, 1:], y_true[:, :, :, 1:]) / (100 * x.shape[0] ** 2)
                    + loss_recon
                )
                optimizer.zero_grad()
                loss_forward.backward(retain_graph=False)
                clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

                pred_x, loss_recon = model.backward(y)
                pred_x = x_normalizer.decode(pred_x.reshape(batchS, s, s, 3).clone())
                x_true = x_normalizer.decode(x.reshape(batchS, s, s, 3).clone())
                loss_backward = (
                    myloss(pred_x[:, :, :, 0], x_true[:, :, :, 0])
                    + myloss(pred_x[:, :, :, 1:], x_true[:, :, :, 1:]) / (100 * x.shape[0] ** 2)
                    + loss_recon
                )
                optimizer.zero_grad()
                loss_backward.backward(retain_graph=False)
                clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                train_l2_forward += loss_forward.item()
                train_l2_backward += loss_backward.item()

            if ep % 10 == 0:
                model.eval()
                test_l2_forward = 0.0
                test_l2_backward = 0.0
                test_grid = 0.0
                with torch.no_grad():
                    for x, y, x_no_noise, y_no_noise in test_loader:
                        x, y, x_no_noise, y_no_noise = (
                            x.cuda(),
                            y.cuda(),
                            x_no_noise.cuda(),
                            y_no_noise.cuda(),
                        )
                        batchS = x.shape[0]
                        pred_y, _ = model(x)
                        pred_y_copy = pred_y.clone()
                        pred_y = pred_y.reshape(batchS, s, s, 3)
                        pred_y = y_normalizer.decode(pred_y.clone())
                        y_true = y_normalizer.decode(y_no_noise.reshape(batchS, s, s, 3).clone())

                        pred_x, _ = model.backward(y.reshape(batchS, s, s, 3))
                        pred_x = x_normalizer.decode(pred_x.reshape(batchS, s, s, 3).clone())
                        x_true = x_normalizer.decode(x_no_noise.reshape(batchS, s, s, 3).clone())

                        diff_forward = pred_y[:, :, :, 0].reshape(batchS, -1) - y_true[:, :, :, 0].reshape(batchS, -1)
                        diff_backward = pred_x[:, :, :, 0].reshape(batchS, -1) - x_true[:, :, :, 0].reshape(batchS, -1)
                        diff_forward_grid = pred_y[:, :, :, 1:].reshape(batchS, -1) - y_true[:, :, :, 1:].reshape(
                            batchS, -1
                        )
                        diff_backward_grid = pred_x[:, :, :, 1:].reshape(batchS, -1) - x_true[:, :, :, 1:].reshape(
                            batchS, -1
                        )

                        test_l2_forward += (
                            ((diff_forward**2).sum(1) / ((y_true[:, :, :, 0].reshape(batchS, -1) ** 2).sum(1))) ** 0.5
                        ).sum()
                        test_l2_backward += (
                            ((diff_backward**2).sum(1) / ((x_true[:, :, :, 0].reshape(batchS, -1) ** 2).sum(1))) ** 0.5
                        ).sum()

                        test_grid += (
                            ((diff_forward_grid**2).sum(1) / ((y_true[:, :, :, 1:].reshape(batchS, -1) ** 2).sum(1)))
                            ** 0.5
                        ).sum()
                        test_grid += (
                            ((diff_backward_grid**2).sum(1) / ((x_true[:, :, :, 1:].reshape(batchS, -1) ** 2).sum(1)))
                            ** 0.5
                        ).sum()

                        pretrain_preds = pred_x[:, :, :, 0]
                        pretrain_gt = x_true[:, :, :, 0]

                train_l2_forward /= ntrain
                train_l2_backward /= ntrain
                test_l2_forward /= ntest
                test_l2_backward /= ntest
                test_grid /= 2 * ntest

                if test_l2_forward < min_err_forward:
                    min_err_forward = test_l2_forward
                if test_l2_backward < min_err_backward:
                    min_err_backward = test_l2_backward
                logger.info(
                    f"  Epoch [{ep + 1}/{epochs_IFNO}], Training Loss Forward: {train_l2_forward:.4f}, Training Loss Backward: {train_l2_backward:.4f},Test Loss Forward: {test_l2_forward:.4f}, Test Loss Backward: {test_l2_backward:.4f},Test Grid: {test_grid:.4f}, Min. Test Forward Error: {min_err_forward:.4f}, Min. Test Backward Error: {min_err_backward:.4f}"
                )

        return pretrain_preds.detach().cpu().numpy(), pretrain_gt.detach().cpu().numpy()

    def joint_train():
        optimizer1 = torch.optim.AdamW(model.parameters(), lr=args.lr_forward)
        optimizer2 = torch.optim.AdamW(model.parameters(), lr=args.lr_backward)
        min_err_forward = 1.0
        min_err_backward = 1.0
        for ep in tqdm(range(epochs)):
            model.train()
            model.vae_net.train()
            train_l2_forward = 0
            train_l2_backward = 0
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                batchS = x.shape[0]
                pred_y, _ = model(x)
                pred_y = pred_y.reshape(batchS, s, s, 3)
                pred_y = y_normalizer.decode(pred_y.clone())
                y_true = y_normalizer.decode(y.clone())
                loss_forward = myloss(pred_y.view(batch_size, -1), y_true.view(batchS, -1))
                optimizer1.zero_grad()

                loss_forward.backward(retain_graph=False)
                clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer1.step()

                x_true = x_normalizer.decode(x.reshape(batchS, s, s, 3).clone())
                pred_x, _ = model.backward(y)
                loss_backward, _ = model.VAE_train(pred_x.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2), return_mu=True)
                loss_grid = myloss(pred_x[:, :, :, 1:], x_true[:, :, :, 1:]) / (100 * x.shape[0] ** 2)
                loss_backward += loss_grid
                optimizer2.zero_grad()
                loss_backward.backward(retain_graph=False)
                clip_grad_norm_(model.parameters(), max_norm=10.0)

                optimizer2.step()

                train_l2_forward += loss_forward.item()
                train_l2_backward += loss_backward.item()

            if ep % 10 == 0:
                cur_time = time.time()
                model.eval()
                model.vae_net.eval()
                test_l2_forward = 0.0
                test_l2_backward = 0.0
                with torch.no_grad():
                    for x, y, x_no_noise, y_no_noise in test_loader:
                        x, y, x_no_noise, y_no_noise = (
                            x.cuda(),
                            y.cuda(),
                            x_no_noise.cuda(),
                            y_no_noise.cuda(),
                        )
                        batchS = x.shape[0]
                        pred_y, _ = model(x)
                        pred_y = pred_y.reshape(batchS, s, s, 3)
                        pred_y = y_normalizer.decode(pred_y.clone())
                        y_true = y_normalizer.decode(y_no_noise.reshape(batchS, s, s, 3).clone())

                        pred_x, _ = model.backward(y.reshape(batchS, s, s, 3))
                        pred_x, _, _, _ = model.vae_net.forward2(pred_x[:, :, :, 0].reshape(batchS, 1, 64, 64))
                        pred_x = x_normalizer.decode(
                            torch.cat(
                                (
                                    pred_x.reshape(batchS, s, s, 1).clone(),
                                    x[:, :, :, 1:],
                                ),
                                axis=-1,
                            )
                        )
                        x_true = x_normalizer.decode(x_no_noise.reshape(batchS, s, s, 3).clone())
                        diff_forward = pred_y[:, :, :, 0].reshape(batchS, -1) - y_true[:, :, :, 0].reshape(
                            batchS, s, s
                        ).reshape(batchS, -1)
                        diff_backward = pred_x[:, :, :, 0].reshape(batchS, -1) - x_true[:, :, :, 0].reshape(batchS, -1)
                        test_l2_forward += (
                            ((diff_forward**2).sum(1) / ((y_true[:, :, :, 0].reshape(batchS, -1) ** 2).sum(1))) ** 0.5
                        ).sum()
                        test_l2_backward += (
                            ((diff_backward**2).sum(1) / ((x_true[:, :, :, 0].reshape(batchS, -1) ** 2).sum(1))) ** 0.5
                        ).sum()

                train_l2_forward /= ntrain
                train_l2_backward /= ntrain
                test_l2_forward /= ntest
                test_l2_backward /= ntest

                if test_l2_forward < min_err_forward:
                    min_err_forward = test_l2_forward
                if test_l2_backward < min_err_backward:
                    min_err_backward = test_l2_backward
                logger.info(
                    f"  Time [{cur_time-start_time}], Epoch [{ep + 1}/{epochs}], Training Loss Forward: {train_l2_forward:.4f}, Training Loss Backward: {train_l2_backward:.4f}, Test Loss Forward: {test_l2_forward:.4f}, Test Loss Backward: {test_l2_backward:.4f}, Min. Test Forward Error: {min_err_forward:.4f}, Min. Test Backward Error: {min_err_backward:.4f}"
                )

    preds, gt = pretrain_IFNO()
    pretrain_VAE()
    joint_train()
  

for i in range(1):
    train()
