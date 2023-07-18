from pyts.image import RecurrencePlot, GramianAngularField, MarkovTransitionField
import numpy as np
import torch
import torch.nn as nn
import pywt


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)) # batch, channel, length
        x = x.permute(0, 2, 1) # batch, length, channel
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        #print("moving", moving_mean.size())
        res = x - moving_mean
        return res, moving_mean, x

class Image_coding(nn.Module):
    def __init__(self, decompose):
        super().__init__()
        self.decompose = decompose
        self.time_decom = series_decomp(25)
        self.scale = np.arange(1, 176, 1)

    def tonumpy(self, x):
        to_num = np.array(x)
        return to_num

    def continuous_wavelet(self, x, len, size):
        x = np.squeeze(x, axis=2)
        ucr_train_cwt = np.zeros((1, size, size))
        for i in range(len):
            cwtmatr, freqs_rate = pywt.cwt(x[i], scales=self.scale, wavelet='morl')
            cwtmatr = np.expand_dims(cwtmatr, axis=0)
            if i == 0:
                ucr_train_cwt = cwtmatr
            else:
                ucr_train_cwt = np.concatenate((ucr_train_cwt, cwtmatr), axis=0)
        return ucr_train_cwt

    def Recurrence_Plot(self, x, y, z):
        rp = RecurrencePlot(dimension=2, time_delay=1)
        return rp.transform(x), rp.transform(y), rp.transform(z)

    def Gramian_sum(self, x, y, z):
        gas = GramianAngularField()
        return gas.transform(x), gas.transform(y), gas.transform(z)

    def Gramian_diff(self, x, y, z):
        gad = GramianAngularField(method='difference')
        return gad.transform(x), gad.transform(y), gad.transform(z)

    def Markovtran(self, x, y, z):
        mk = MarkovTransitionField()
        return mk.transform(x), mk.transform(y), mk.transform(z)

    def forward(self, x):
        xt = torch.FloatTensor(x)
        len = xt.size(0)
        size = xt.size(1)
        x_cwv = self.continuous_wavelet(x, len, size)
        # x_cwv = torch.FloatTensor(x_cwv)
        # print(x_cwv.size())
        x_res, x_tr, xt = self.time_decom(xt)
        xt = xt.squeeze(2)
        x_res = x_res.squeeze(2)
        x_tr = x_tr.squeeze(2)

        x_rp, x_rerp, x_trrp = self.Recurrence_Plot(xt, x_res, x_tr)
        x_gas, x_regas, x_trgas = self.Gramian_sum(xt, x_res, x_tr)
        x_gad, x_regad, x_trgad = self.Gramian_diff(xt, x_res, x_tr)
        x_mk, x_remk, x_trmk = self.Markovtran(xt, x_res, x_tr)
        return x_rp, x_rerp, x_trrp, x_gas, x_regas, x_trgas, x_gad, x_regad, x_trgad, x_mk, x_remk, x_trmk, x_cwv