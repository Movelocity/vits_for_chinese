import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, embed_dim=192):
        super().__init__()
        # it needs to be removed from future version.
        filter_channels = in_channels
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.cond = nn.Linear(embed_dim, filter_channels)

    def forward(self, x, x_mask, w=None, embed=None, training=False, noise_scale=1.0):
        x = self.pre(torch.detach(x))
        x += self.cond(torch.detach(embed)).unsqueeze(-1)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if training:
            flows = self.flows
            assert w is not None

            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q

            logdet_tot_q = 0
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, embed=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) +F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2))* x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, embed=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2))* x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b] 训练时直接返回Loss(batched)
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, embed=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw  # 预测时返回时长预测值


class DurationPredictor(nn.Module):
    # 和StochasticDurationPredictor二选一
    # 论文里随机时长预测比这个确定时长预测MOS高0.04
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, embed_dim):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        self.cond = nn.Linear(embed_dim, in_channels)

    def forward(self, x, x_mask, w, embed, training=False, noise_scale=1):
        # w, noise scale 参数仅作为占位，实际上不使用
        x = torch.detach(x)
        x += self.cond(torch.detach(embed)).unsqueeze(-1)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        logw_pred = x * x_mask
        if training:  
            logw_calc = torch.log(w + 1e-6) * x_mask
            # torch.sum(..., [1,2])使得损失值l_length保留了一个batch维度
            return torch.sum((logw_pred-logw_calc)**2, [1, 2]) # 训练时直接返回Loss(batched)
        else:  
            return logw_pred  # 预测时返回时长预测值


class TextEncoder(nn.Module):
    """
    文本侧的编码器
    forward() 返回的最后一个是 mask, 因为处理的是 sequential data, 长短不一, 短的需要加mask。
    """
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels,
        n_heads, n_layers, kernel_size, p_dropout
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(hidden_channels, filter_channels, n_heads,
            n_layers, kernel_size, p_dropout)
        # 输出通道 out_channels*2, 拆分出一半均值和一半方差对数
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        
    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    # 来自 WaveGlow
    # flow-based model. 数据可以双向流动，只有一个方向需要学习，另一个方向可以推算出来
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
        n_layers, n_flows=4, embed_dim=192
    ):
        super().__init__()

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size, dilation_rate,
                    n_layers, embed_dim=embed_dim, mean_only=True)
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, embed, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, embed=embed, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, embed=embed, reverse=reverse)
        return x


class WaveEncoder(nn.Module):
    """
    音频侧的编码器。核心是WaveNet
    forward() 返回的最后一个是 mask, 因为处理的是 sequential data, 长短不一, 短的需要加mask。
    """
    def __init__(self, in_channels, out_channels, hidden_channels,
        kernel_size, dilation_rate, n_layers, embed_dim=192
    ):
        super(WaveEncoder, self).__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, embed_dim=embed_dim)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, embed):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, max_length=x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, embed=embed)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(logs)*torch.exp(logs)) * x_mask  # VAE reparametrization
        return z, m, logs, x_mask

"""
upsample_rates = [8, 8, 2, 2]
upsample_kernel_sizes = [16, 16, 4, 4]
upsample_initial_channel = 512

ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4)
sum(resblocks)
ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4)
sum(resblocks)
ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
sum(resblocks)
ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
"""
class WaveGenerator(torch.nn.Module):
    def __init__(
        self, initial_channel, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, 
        upsample_initial_channel, upsample_kernel_sizes, embed_dim=192
    ):
        super(WaveGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        self.MRFs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel//(2**i)
            out_ch = in_ch//2
            self.ups.append(weight_norm(
                ConvTranspose1d(in_ch, out_ch, kernel_size=k, stride=u, padding=(k-u)//2)
            ))
            self.MRFs.append(modules.MultiReceptiveField(out_ch, resblock_kernel_sizes, resblock_dilation_sizes))
        self.ups.apply(init_weights)

        self.conv_post = Conv1d(out_ch, 1, 7, 1, padding=3, bias=False)
        self.cond = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, upsample_initial_channel),
            nn.Tanh()
        )

    def forward(self, x, embed):
        x =  self.conv_pre(x) + self.cond(embed).unsqueeze(-1)  # 让 Linear 的输出类似于 Conv1d
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            x = self.MRFs[i](x)
        x = F.leaky_relu(x)
        x = self.conv_post(x)  # 输出1通道语音信号
        x = torch.tanh(x)      # tanh 激活后输出有正负
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1),
                   padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1),
                   padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1),
                   padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1),
                   padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1,
                   padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    """
    来自HifiGAN MultiScaleDiscriminator的第一个子模块
    """
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),  # 分组卷积
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        # 把从第二维到最后一维之间的数据平坦化，相当于fla = torch.nn.Flattern(); x = fla(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    来自HifiGAN
    """
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + \
            [DiscriminatorP(i, use_spectral_norm=use_spectral_norm)
             for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

from text import symbols

class VITS_Model(nn.Module):
    def __init__(
        self, 
        n_vocab=len(symbols),
        spec_channels=513,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
        upsample_initial_channel=512,
        upsample_rates=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        embed_dim=192,
        use_sdp=False,
        **kwargs
    ):
        super().__init__()
        self.enc_p = TextEncoder(
            n_vocab, inter_channels, hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size, p_dropout)

        self.dec = WaveGenerator(
            initial_channel=inter_channels,  
            resblock_kernel_sizes=resblock_kernel_sizes, 
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates, 
            upsample_initial_channel=upsample_initial_channel, 
            upsample_kernel_sizes=upsample_kernel_sizes, 
            embed_dim=embed_dim)

        self.enc_q = WaveEncoder(
            in_channels=spec_channels, 
            out_channels=inter_channels, 
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16, 
            embed_dim=embed_dim)

        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, embed_dim=embed_dim)

        if use_sdp:
            self.dp = StochasticDurationPredictor(in_channels=hidden_channels, filter_channels=192, 
                kernel_size=3, p_dropout=0.5, n_flows=4, embed_dim=embed_dim)
        else:
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, embed_dim=embed_dim)

    def forward(self, x, x_lengths, y, y_lengths, embed):
        pass

    @torch.no_grad()
    def infer(self, x, x_lengths, embed, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # predict alignments
        logw = self.dp(x, x_mask, embed=embed, training=False, noise_scale=noise_scale_w)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)

        # 沿着维度1,2求和，剩下第0维度(batch), clamp(w, 1)使得w最小值为1
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale  # VAE sampling
        z = self.flow(z_p, y_mask, embed=embed, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], embed=embed)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, embed=g_src)
        z_p = self.flow(z, y_mask, embed=g_src)
        z_hat = self.flow(z_p, y_mask, embed=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, embed=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)

# class SynthesizerTrn(nn.Module):
#     """
#     合成器，可以用于训练和推理
#     改配置文件不如改代码方便, 于是设置init参数默认值
#     """
#     def __init__(
#         self, 
#         n_vocab=len(symbols),
#         spec_channels=513,
#         segment_size=32,
#         inter_channels=192,
#         hidden_channels=192,
#         filter_channels=768,
#         n_heads=2,
#         n_layers=6,
#         kernel_size=3,
#         p_dropout=0.1,
#         resblock_kernel_sizes=[3, 7, 11],
#         resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
#         upsample_initial_channel=512,
#         upsample_rates=[8, 8, 2, 2],
#         upsample_kernel_sizes=[16, 16, 4, 4],
#         embed_dim=192,
#         use_sdp=False,
#         **kwargs):

#         super().__init__()
#         self.segment_size = segment_size  # 训练专用
#         self.use_sdp = use_sdp

#         self.enc_p = TextEncoder(
#             n_vocab, inter_channels, hidden_channels, filter_channels,
#             n_heads, n_layers, kernel_size, p_dropout)

#         self.dec = WaveGenerator(
#             initial_channel=inter_channels,  
#             resblock_kernel_sizes=resblock_kernel_sizes, 
#             resblock_dilation_sizes=resblock_dilation_sizes,
#             upsample_rates=upsample_rates, 
#             upsample_initial_channel=upsample_initial_channel, 
#             upsample_kernel_sizes=upsample_kernel_sizes, 
#             embed_dim=embed_dim)

#         self.enc_q = WaveEncoder(
#             in_channels=spec_channels, 
#             out_channels=inter_channels, 
#             hidden_channels=hidden_channels,
#             kernel_size=5,
#             dilation_rate=1, 
#             dilation_rate=16, 
#             embed_dim=embed_dim)

#         self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, embed_dim=embed_dim)

#         if use_sdp:
#             self.dp = StochasticDurationPredictor(in_channels=hidden_channels, filter_channels=192, 
#                 kernel_size=3, p_dropout=0.5, n_flows=4, embed_dim=embed_dim)
#         else:
#             self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, embed_dim=embed_dim)

#     def forward(self, x, x_lengths, y, y_lengths, embed):
#         # x: 文本编码；y: 语音频谱
#         x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
#         z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, embed=embed)
#         z_p = self.flow(z, y_mask, embed=embed)  # 具体形状有待调试

#         with torch.no_grad():
#             # negative cross-entropy between speech prior (s_p) and word prior (z_p)
#             # -sum(x * log(y) + (1 - x) * log(1 - y))
#             # [b, d, t] meaning sp times squre root of e
#             s_p_sq_r = torch.exp(-2 * logs_p)  # (e^ln(s_p))^(-2) = 1 / s_p^2
#             neg_cent1 = torch.sum(-0.5*math.log(2*math.pi) -logs_p, [1], keepdim=True)  # [b, 1, t_s]
#             # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
#             neg_cent2 = torch.matmul(-0.5*(z_p**2).transpose(1, 2), s_p_sq_r)
#             # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
#             neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
#             neg_cent4 = torch.sum(-0.5*(m_p**2) * s_p_sq_r,[1], keepdim=True)  # [b, 1, t_s]
#             neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

#             attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
#             attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

#         w = attn.sum(2)
#         # calculate duration loss
#         if self.use_sdp:
#             l_length = self.dp(x, x_mask, w, embed=embed)
#             l_length = l_length / torch.sum(x_mask)
#         else:
#             logw_ = torch.log(w + 1e-6) * x_mask
#             logw = self.dp(x, x_mask, embed=embed)
#             # torch.sum(..., [1,2])使得损失值l_length保留了一个batch维度，为什么？
#             l_length = torch.sum((logw-logw_)**2, [1, 2]) / torch.sum(x_mask)

#         # expand prior
#         m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
#         logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

#         z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
#         # 批次里每条语音只取一个片段用于计算loss，这里指定片段的偏移量，便于从ground truth里截取同样的片段
#         o = self.dec(z_slice, embed=embed)
#         return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

#     @torch.no_grad()
#     def infer(self, x, x_lengths, embed, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
#         x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

#         # speaker embedding
#         # if embed is None:
#         # embed = torch.randn(x.shape[0], self.embed_dim)
#         # g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]

#         # predict alignments
#         logw = self.dp(x, x_mask, embed=embed, reverse=True, noise_scale=noise_scale_w)
#         w = torch.exp(logw) * x_mask * length_scale
#         w_ceil = torch.ceil(w)

#         # 沿着维度1,2求和，剩下第0维度(batch), clamp(w, 1)使得w最小值为1
#         y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        
#         y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
#         attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
#         attn = commons.generate_path(w_ceil, attn_mask)

#         m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
#         logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']

#         z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale  # VAE sampling
#         z = self.flow(z_p, y_mask, embed=embed, reverse=True)
#         o = self.dec((z * y_mask)[:, :, :max_len], embed=embed)
#         return o, attn, y_mask, (z, z_p, m_p, logs_p)

#     def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
#         g_src = self.emb_g(sid_src).unsqueeze(-1)
#         g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
#         z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, embed=g_src)
#         z_p = self.flow(z, y_mask, embed=g_src)
#         z_hat = self.flow(z_p, y_mask, embed=g_tgt, reverse=True)
#         o_hat = self.dec(z_hat * y_mask, embed=g_tgt)
#         return o_hat, y_mask, (z, z_p, z_hat)
