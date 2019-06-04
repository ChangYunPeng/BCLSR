import math

import torch
import torch.nn as nn
import torch.nn.modules.upsampling as upsampling
from torch.nn.modules import Upsample
from torch.nn.parameter import Parameter
from spectral import SpectralNorm
import numpy as np
from model_ab.bilstm.BiConvLSTM import BiConvLSTM


class depthwise_conv(nn.Module):
    def __init__(self, in_dim, out_dim, wn):
        super(depthwise_conv, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = wn(nn.Conv2d(in_dim, out_dim, (3, 3), (1, 1), (1, 1), groups=4))
        self.conv2 = wn(nn.Conv2d(out_dim, out_dim, (1, 1), (1, 1), (0, 0), groups=1))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class dilated_conv(nn.Module):
    def __init__(self, in_dim, out_dim, wn):
        super(dilated_conv, self).__init__()
        self.relu = nn.LeakyReLU()
        self.conv3 = wn(nn.Conv2d(in_dim, out_dim, (3, 3), (1, 1), (1, 1), dilation=1))
        self.conv5 = wn(nn.Conv2d(out_dim, out_dim, (3, 3), (1, 1), (1, 1), dilation=1))
        # self.conv7 = wn(nn.Conv2d(out_dim, out_dim, (3, 3), (1, 1), (3, 3), dilation=3))
        # self.conv9 = wn(nn.Conv2d(out_dim, out_dim, (3, 3), (1, 1), (4, 4), dilation=4))
        # self.conv2 = wn(nn.Conv2d(out_dim, out_dim, (1, 1), (1, 1), (0, 0), groups=1))
    
    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x3)
        # x7 = self.conv7(x)
        # x9 = self.conv9(x)
        x = torch.cat([x3,x5], 1)
        # x = self.relu(x)
        return x

class Block(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1, res_par = False, depthwise_tag = False):
        super(Block, self).__init__()
        self.res_par_tag = res_par
        if res_par:
            self.res_scale = nn.Parameter(torch.ones([1,n_feats,1,1]),requires_grad=True)
            # self.res_scale = nn.Parameter(torch.ones(1),requires_grad=True)
        else :
            self.res_scale = res_scale

        body = []
        expand = 6
        linear = 0.8
        
        body.append(
            wn(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))
            
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) 
        res = x + res
        return res


class DenseBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1, res_par = False, depthwise_tag = False):
        super(DenseBlock, self).__init__()
        self.res_par_tag = res_par
        if res_par:
            self.res_scale = nn.Parameter(torch.ones([1,n_feats,1,1]),requires_grad=True)
        else :
            self.res_scale = res_scale

        body = []
        dense_connection_num = 4
        all_channels = n_feats*2
        dense_cell_channel = 16

        dense_in_body = []
        dense_in_body.append( nn.Conv2d( n_feats, all_channels , 1 , padding=0 ) ) 
        dense_in_body.append( act ) 
        self.dense_in_body = nn.Sequential(*dense_in_body)

        self.dense_body = []

        # multi -- cell
        cell0 = []
        cell0.append( nn.Conv2d(all_channels + dense_cell_channel*0, dense_cell_channel , 3 , padding=3//2 ) )
        cell0.append(act)
        self.cell0 = nn.Sequential(*cell0)

        cell1 = []
        cell1.append( nn.Conv2d(all_channels + dense_cell_channel*1, dense_cell_channel , 3 , padding=3//2 ) )
        # cell1.append( nn.Conv2d(, dense_cell_channel , 3 , padding=3//2 ) )
        cell1.append(act)
        self.cell1 = nn.Sequential(*cell1)

        cell2 = []
        cell2.append( nn.Conv2d(all_channels + dense_cell_channel*2, dense_cell_channel , 3 , padding=3//2 ) )
        cell2.append(act)
        self.cell2 = nn.Sequential(*cell2)

        cell3 = []
        cell3.append( nn.Conv2d(all_channels + dense_cell_channel*3, dense_cell_channel , 3 , padding=3//2 ) )
        cell3.append(act)
        self.cell3 = nn.Sequential(*cell3)

        # for idx in range(dense_connection_num):
        #     cell = []
        #     cell.append( nn.Conv2d(all_channels + dense_cell_channel*idx, dense_cell_channel , 3 , padding=3//2 ) )
        #     cell.append(act)
        #     cell = nn.Sequential(*cell)
        #     self.dense_body.append(cell.cuda())
        
        dense_out_body = []
        # dense_out_body.append( nn.Conv2d( all_channels + dense_cell_channel*dense_connection_num , all_channels , 3 , padding=3//2 ) ) 
        dense_out_body.append( nn.Conv2d( all_channels + dense_cell_channel*dense_connection_num , n_feats , 1 , padding=0 ) ) 
        dense_out_body.append( act ) 
        
        self.dense_out_body = nn.Sequential(*dense_out_body)

        # body.append(
        #     wn(nn.Conv2d(n_feats, n_feats*expand, 3, padding=3//2)))
        # body.append(act)
            
        # self.body = nn.Sequential(*body)

    def forward(self, x):

        x_in = self.dense_in_body(x)
        out  = self.cell0(x_in)
        x_in = torch.cat( [x_in, out ] , 1 )
        out  = self.cell1(x_in)
        x_in = torch.cat( [x_in, out ] , 1 )
        out  = self.cell2(x_in)
        x_in = torch.cat( [x_in, out ] , 1 )
        out  = self.cell3(x_in)
        x_in = torch.cat( [x_in, out ] , 1 )

        res = self.dense_out_body(x_in) 
        x_out = x + res * self.res_scale
        return x_out

class DenseMODEL(nn.Module):
    def __init__(self, args, nor_type = 'sp', img_size=256,recursive_num=4,fusion_strategy = 0, upsample_strategy = 0 ,res_par_tag = False, depthwise_tag = False, att_tag = True):
        super(DenseMODEL, self).__init__()
        # hyper-params
        self.args = args
        self.res_par_tag = res_par_tag
        self.lstm_tags = True

        self.fusion_strategy = fusion_strategy
        self.upsample_strategy = upsample_strategy


        self.recursive_num = recursive_num
        self.img_size = img_size

        if res_par_tag:
            self.res_scale = nn.Parameter(torch.ones(1))
        else :
            self.res_scale = 1

        scale = args.upscale_factor
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        # n_feats = 64
        kernel_size = 3
        act = nn.ReLU(True)
        
        if nor_type == 'sp':
            wn = SpectralNorm
        elif nor_type == 'wn':
            wn = lambda x: torch.nn.utils.weight_norm(x)
        elif nor_type == 'none':
            wn = lambda x: x
        
        if args.n_colors == 3:
            self.rgb_mean = nn.Parameter(torch.FloatTensor([args.r_mean, args.g_mean, args.b_mean])).view([1, 3, 1, 1]).cuda()
        else:
            self.rgb_mean = nn.Parameter(torch.FloatTensor([args.r_mean])).view([1, 1, 1, 1]).cuda()

        # define head module
        head = []
        # head.append(wn(nn.Conv2d(args.n_colors, n_feats, 3, padding=3//2)))
        head.append(wn(nn.Conv2d(args.n_colors, n_feats, 3, padding=3//2)))

        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(
                Block(n_feats, kernel_size, act=act, res_scale=args.res_scale, wn=wn, res_par=res_par_tag, depthwise_tag=depthwise_tag))

        # define LSTM
        height = self.img_size//scale
        width = self.img_size//scale
        self.converge = BiConvLSTM(input_size=(height, width),
                 input_dim=n_feats,
                 hidden_dim=[ n_feats//2, n_feats//2],
                 kernel_size=(1, 1),
                 num_layers=2,
                 bias=True,
                 return_all_layers=False, return_fl = False)



        # define tail module
        tail = []
        out_feats = scale*scale*args.n_colors
        if self.fusion_strategy == 2:
            tail.append(
                wn(nn.Conv2d(n_feats*self.recursive_num, out_feats, 1)))
        elif self.fusion_strategy == 0:
            tail.append(
                wn(nn.Conv2d(self.recursive_num*n_feats//2, out_feats, 1)))
        else:
            tail.append(
                wn(nn.Conv2d(n_feats, out_feats, 1)))

        tail.append(nn.PixelShuffle(scale))
        self.tail = nn.Sequential(*tail)



        # make object members
        self.head = nn.Sequential(*head)
        # self.body = nn.Sequential(*body)

        self.body1 = DenseBlock(n_feats, kernel_size, act=act, res_scale=args.res_scale, wn=wn, res_par=res_par_tag, depthwise_tag=depthwise_tag).cuda()
        # self.body1 = Block(n_feats, kernel_size, act=act, res_scale=args.res_scale, wn=wn, res_par=res_par_tag, depthwise_tag=depthwise_tag).cuda()
        # self.body2 = DenseBlock(n_feats, kernel_size, act=act, res_scale=args.res_scale, wn=wn, res_par=res_par_tag, depthwise_tag=depthwise_tag).cuda()
        # self.body3 = DenseBlock(n_feats, kernel_size, act=act, res_scale=args.res_scale, wn=wn, res_par=res_par_tag, depthwise_tag=depthwise_tag).cuda()
        # self.body4 = DenseBlock(n_feats, kernel_size, act=act, res_scale=args.res_scale, wn=wn, res_par=res_par_tag, depthwise_tag=depthwise_tag).cuda()

        # self.skip = nn.Sequential(*skip)
        skip = []
        skip.append(
            wn(nn.Conv2d(args.n_colors, out_feats, 5, padding=5//2))
        )
        skip.append(nn.PixelShuffle(scale))

        if self.upsample_strategy == 0:
            self.skip = Upsample(scale_factor=scale)
        elif self.upsample_strategy == 1:
            self.skip = nn.Sequential(*skip)

        self.att_tag = att_tag

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = (x - self.rgb_mean*255)/127.5
        
        s = self.skip(x)
        # s = 
        x = self.head(x)
        x_head = x

        x_list = []
        for idx in range(self.recursive_num):
            x = self.body1(x)
            x_list.append( torch.unsqueeze(x,1) )

        # x = self.body1(x)
        # x_list.append( torch.unsqueeze(x,1) )
        # x = self.body2(x)
        # x_list.append( torch.unsqueeze(x,1) )
        # x = self.body3(x)
        # x_list.append( torch.unsqueeze(x,1) )
        # x = self.body4(x)
        # x_list.append( torch.unsqueeze(x,1) )
        
        
        if self.fusion_strategy == 0:
            x = torch.cat(x_list,1)

            lstm_out = self.converge(x)
            tmp_list = []
            for idx in range(lstm_out.shape[1]):
                tmp = lstm_out[:,idx,:,:,:]
                tmp_list.append(tmp)
            # lstm_out = torch.cat( (lstm_out[:,0,:,:,:],lstm_out[:,-1,:,:,:]), dim=1 )
            lstm_out =  torch.cat(tmp_list,1)
            x = lstm_out
        elif self.fusion_strategy == 1:
            x_temp = 0
            for x_iter in x_list:
                x_temp+=torch.squeeze(x_iter,1)
            x = x_temp
        elif self.fusion_strategy == 2:
            # x_sq_list = []
            # for var in x_list:
            #     import pdb; pdb.set_trace()
            #     x_sq_list.append(torch.squeeze(var,1))
            x_sq_list = [ torch.squeeze(var,1) for var in x_list]
            x = torch.cat(x_sq_list, 1)
            # import pdb; pdb.set_trace()
            # x = x.view()
        # x = lstm_out.view(lstm_out.shape[0],-1,lstm_out.shape[3],lstm_out.shape[4])
        # torch.cat( (lstm_out[:,0,:,:,:],lstm_out[:,-1,:,:,:]), dim=1 )  
        
        # import pdb; pdb.set_trace()
        # x = self.body(x)
       
       
        x = self.tail(x)
        # self.res_scale
        if self.res_par_tag:
            x = s + x*self.res_scale
        else:
            x += s
        x = x*127.5 + self.rgb_mean*255
        return x



if __name__ == "__main__":
    # test_model = DenseMODEL( )
    print('test')