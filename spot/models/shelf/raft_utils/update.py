import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class AlphaHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class SepConvGRU2(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128, memory_dim=128, enable_gma=False):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

        # for sensory memory
        self.convz1_m = nn.Conv2d(memory_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1_m = nn.Conv2d(memory_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1_m = nn.Conv2d(memory_dim, hidden_dim, (1, 5), padding=(0, 2))
        nn.init.zeros_(self.convz1_m.weight.data)
        nn.init.zeros_(self.convz1_m.bias.data)
        nn.init.zeros_(self.convr1_m.weight.data)
        nn.init.zeros_(self.convr1_m.bias.data)
        nn.init.zeros_(self.convq1_m.weight.data)
        nn.init.zeros_(self.convq1_m.bias.data)

        self.convz2_m = nn.Conv2d(memory_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2_m = nn.Conv2d(memory_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2_m = nn.Conv2d(memory_dim, hidden_dim, (5, 1), padding=(2, 0))
        nn.init.zeros_(self.convz2_m.weight.data)
        nn.init.zeros_(self.convz2_m.bias.data)
        nn.init.zeros_(self.convr2_m.weight.data)
        nn.init.zeros_(self.convr2_m.bias.data)
        nn.init.zeros_(self.convq2_m.weight.data)
        nn.init.zeros_(self.convq2_m.bias.data)
        self.enable_gma = enable_gma
        if enable_gma:
            # for gma
            self.convz1_gma = nn.Conv2d(hidden_dim, hidden_dim, (1, 5), padding=(0, 2))
            self.convr1_gma = nn.Conv2d(hidden_dim, hidden_dim, (1, 5), padding=(0, 2))
            self.convq1_gma = nn.Conv2d(hidden_dim, hidden_dim, (1, 5), padding=(0, 2))
            nn.init.zeros_(self.convz1_gma.weight.data)
            nn.init.zeros_(self.convz1_gma.bias.data)
            nn.init.zeros_(self.convr1_gma.weight.data)
            nn.init.zeros_(self.convr1_gma.bias.data)
            nn.init.zeros_(self.convq1_gma.weight.data)
            nn.init.zeros_(self.convq1_gma.bias.data)

            self.convz2_gma = nn.Conv2d(hidden_dim, hidden_dim, (5, 1), padding=(2, 0))
            self.convr2_gma = nn.Conv2d(hidden_dim, hidden_dim, (5, 1), padding=(2, 0))
            self.convq2_gma = nn.Conv2d(hidden_dim, hidden_dim, (5, 1), padding=(2, 0))
            nn.init.zeros_(self.convz2_gma.weight.data)
            nn.init.zeros_(self.convz2_gma.bias.data)
            nn.init.zeros_(self.convr2_gma.weight.data)
            nn.init.zeros_(self.convr2_gma.bias.data)
            nn.init.zeros_(self.convq2_gma.weight.data)
            nn.init.zeros_(self.convq2_gma.bias.data)

    def forward(self, h, x, s_m, mot_global=None):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        if self.enable_gma:
            z = torch.sigmoid(self.convz1(hx) + self.convz1_m(s_m) + self.convz1_gma(mot_global))
            r = torch.sigmoid(self.convr1(hx) + self.convr1_m(s_m) + self.convr1_gma(mot_global))
            q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)) + self.convq1_m(s_m) + self.convq1_gma(mot_global))
            h = (1 - z) * h + z * q

            # vertical
            hx = torch.cat([h, x], dim=1)
            z = torch.sigmoid(self.convz2(hx) + self.convz2_m(s_m) + self.convz2_gma(mot_global))
            r = torch.sigmoid(self.convr2(hx) + self.convr2_m(s_m) + self.convr2_gma(mot_global))
            q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)) + self.convq2_m(s_m) + self.convq2_gma(mot_global))
        else:
            z = torch.sigmoid(self.convz1(hx) + self.convz1_m(s_m))
            r = torch.sigmoid(self.convr1(hx) + self.convr1_m(s_m))
            q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)) + self.convq1_m(s_m))
            h = (1 - z) * h + z * q

            # vertical
            hx = torch.cat([h, x], dim=1)
            z = torch.sigmoid(self.convz2(hx) + self.convz2_m(s_m))
            r = torch.sigmoid(self.convr2(hx) + self.convr2_m(s_m))
            q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)) + self.convq2_m(s_m))
        h = (1 - z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, refine_alpha, corr_levels=4, corr_radius=4):
        super().__init__()
        in_dim = 2 + (3 if refine_alpha else 0)
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.refine_alpha = refine_alpha
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(in_dim, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - in_dim, 3, padding=1)

    def forward(self, flow, alpha, corr):
        if self.refine_alpha:
            flow = torch.cat([flow, alpha, torch.zeros_like(flow)], dim=1)
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        feat = F.relu(self.convf1(flow))
        feat = F.relu(self.convf2(feat))
        feat = torch.cat([cor, feat], dim=1)
        feat = F.relu(self.conv(feat))
        return torch.cat([feat, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, patch_size=8, refine_alpha=False):
        super().__init__()
        self.refine_alpha = refine_alpha
        self.encoder = BasicMotionEncoder(refine_alpha)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)

        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, patch_size * patch_size * 9, 1, padding=0)
        )

        if refine_alpha:
            self.alpha_head = AlphaHead(hidden_dim, hidden_dim=256)
            self.alpha_mask = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, patch_size * patch_size * 9, 1, padding=0)
            )

    def forward(self, net, inp, corr, flow, alpha):
        mot = self.encoder(flow, alpha, corr)
        inp = torch.cat([inp, mot], dim=1)
        net = self.gru(net, inp)

        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)

        delta_alpha, mask_alpha = None, None
        if self.refine_alpha:
            delta_alpha = self.alpha_head(net)
            mask_alpha = .25 * self.alpha_mask(net)

        return net, mask, delta_flow, mask_alpha, delta_alpha


class BasicUpdateBlock2(nn.Module):
    def __init__(self, args=None, hidden_dim=128, patch_size=8, refine_alpha=False):
        super().__init__()
        self.refine_alpha = refine_alpha
        self.encoder = BasicMotionEncoder(refine_alpha)
        self.gru = SepConvGRU2(hidden_dim=hidden_dim, input_dim=128 + hidden_dim, memory_dim=hidden_dim)
                               

        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, patch_size * patch_size * 9, 1, padding=0)
        )

        if refine_alpha:
            self.alpha_head = AlphaHead(hidden_dim, hidden_dim=256)
            self.alpha_mask = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, patch_size * patch_size * 9, 1, padding=0)
            )

    def forward(self, net, inp, corr, flow, alpha, s_m, attention=None):
        mot = self.encoder(flow, alpha, corr)
        mot_global = None
        inp = torch.cat([inp, mot], dim=1)
        net = self.gru(net, inp, s_m, mot_global)

        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)

        delta_alpha, mask_alpha = None, None
        if self.refine_alpha:
            delta_alpha = self.alpha_head(net)
            mask_alpha = .25 * self.alpha_mask(net)

        return net, mask, delta_flow, mask_alpha, delta_alpha, mot
