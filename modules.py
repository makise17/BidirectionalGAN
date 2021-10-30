import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class Generator(nn.Module):
    def __init__(self,latent_dim,img_shape,n_classes):
        super(Generator,self).__init__()
        self.img_shape= img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim+n_classes, 128, normalize=True),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self, z, labels):
        # img:[batch_size,img_size,img_size], z:[batch_size,latent_dim]
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return (img,z,labels)


class Encoder(nn.Module):
    def __init__(self,latent_dim,img_shape,n_classes):
        super(Encoder,self).__init__()
        self.img_shape= img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape))+n_classes,1024),
            *block(1024, 512, normalize=True),
            *block(512, 256, normalize=True),
            *block(256, 128, normalize=True),
            *block(128, latent_dim, normalize=True),
            nn.Tanh()
        )

    def forward(self, img, labels):
        # img:[batch_size,img_size,img_size], z:[batch_size,latent_dim]
        emb = self.label_emb(labels)
        enc_input = torch.cat((emb, img), -1)
        z = self.model(enc_input)
        img = img.view(img.size(0), *self.img_shape)
        return (img,z, labels)

    
class Discriminator(nn.Module):
    def __init__(self, latent_dim,img_shape, n_classes):
        super(Discriminator, self).__init__()

        joint_shape = latent_dim + np.prod(img_shape) + n_classes
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(joint_shape, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, z, labels):
        # img:[batch_size,img_size,img_size], z:[batch_size,latent_dim]
        joint = torch.cat((img.view(img.size(0),-1),z),dim=1)
        dis_input = torch.cat((joint,self.label_emb(labels)),dim=1)
        validity = self.model(dis_input)
        return validity

