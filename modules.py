import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class Generator(nn.Module):
    def __init__(self,z_dim,label_dim,img_shape):
        super(Generator,self).__init__()
        self.img_shape= img_shape
        self.n_classes = label_dim
        self.input_x = nn.Sequential(
            # input is Z, going into a convolution
            # 4倍 1*1->4*4
            nn.ConvTranspose2d(z_dim, 64*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(),
        )
        self.input_y = nn.Sequential(
            # input is Z, going into a convolution
            # 4倍 1*1->4*4
            nn.ConvTranspose2d( label_dim, 64*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(),
        )
        self.concat = nn.Sequential(
            # 2倍 4*4->8*8
            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False),    
            nn.BatchNorm2d(64*4),
            nn.ReLU(),
            # 2倍 8*8->??*??
            nn.ConvTranspose2d( 64*4, 64*2, 4, 2, 2, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(),
 
            # 倍 ??*??->28*28
            nn.ConvTranspose2d( 64*2, 1, 4, 2, 1, bias=False),
            #nn.Sigmoid()
            nn.Tanh()
 
        )
    def forward(self, z, labels):
        # img:[batch_size,img_size,img_size], z:[batch_size,latent_dim]
        x = self.input_x(z.reshape((z.size(0),z.size(1),1,1)))
        y = self.input_y(labels.reshape((labels.size(0),labels.size(1),1,1)))
        img = torch.cat([x, y] , dim=1)
        img = self.concat(img)
        return img


class Encoder(nn.Module):
    def __init__(self,z_dim,label_dim,img_shape):
        super(Encoder,self).__init__()
        self.img_shape= img_shape

        self.input_x = nn.Sequential(
            # 32*32 -> 16*16
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.input_y = nn.Sequential(
            # 32*32 -> 16*16
            nn.Conv2d(label_dim, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.concat = nn.Sequential( 
            # 16*16 -> 8*8          
            nn.Conv2d(64*2 , 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2),          
            # 8*8 -> 4*4
            nn.Conv2d(64*4, 64*8, 4, 2, 2, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2),
            # 4*4 -> 1*1r
            nn.Conv2d(64 * 8, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8, z_dim),
            nn.BatchNorm1d(z_dim, 0.8),
            nn.Tanh(),
        )
    def forward(self, img, labels):
        # img:[batch_size,img_size,img_size], z:[batch_size,latent_dim]
        x = self.input_x(img)

        y = self.input_y(labels)
        z = torch.cat([x, y] , dim=1)
        z = self.concat(z)
        z = self.fc(z.view((img.size(0),-1)))
        # z = self.fc(z.flatten(start_dim=1))
        return z

    
class Discriminator(nn.Module):
    def __init__(self, latent_dim,label_dim,img_shape):
        super(Discriminator, self).__init__()

        joint_shape = latent_dim + np.prod(img_shape) + label_dim
        self.label_emb = nn.Embedding(label_dim, label_dim)

        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(joint_shape, 512)),
            #nn.ELU(),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(512, 512)),
            nn.Dropout(0.4),
            #nn.ELU(),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(512, 512)),
            nn.Dropout(0.4),
            #nn.ELU(),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(512, 1)),
            nn.Sigmoid() 
        )

        # self.input_x = nn.Sequential(
            
        #     nn.Conv2d(1, 64, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        # )
        # self.input_y = nn.Sequential(
            
        #     nn.Conv2d(label_dim, 64, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        
        # self.concat = nn.Sequential(
            
        #     nn.Conv2d(64*2 , 64*4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(64 * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
            
 
        #     nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(64 * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
 
        #     nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
        #     nn.Sigmoid()
 
        # )


    def forward(self, img, z, labels):
        # img:[batch_size,img_size,img_size], z:[batch_size,latent_dim]
        joint = torch.cat((img.view(img.size(0),-1),z),dim=1)
        dis_input = torch.cat((joint,self.label_emb(labels)),dim=1)
        validity = self.model(dis_input)
        return validity

