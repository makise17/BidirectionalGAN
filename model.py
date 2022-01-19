import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.optim import lr_scheduler

from modules import Generator,Encoder,Discriminator

import itertools
import matplotlib.pyplot as plt

class BiCoDCGAN(nn.Module):
    def __init__(self,config):
        super(BiCoDCGAN,self).__init__()

        self._work_type = config.work_type
        self._epochs = config.epochs
        self._batch_size = config.batch_size

        self._encoder_lr = config.encoder_lr
        self._encoder_alpha = config.encoder_alpha
        self._encoder_phi = config.encoder_phi
        self._encoder_rho = config.encoder_rho
        
        self._generator_lr = config.generator_lr
        self._discriminator_lr = config.discriminator_lr
        self._latent_dim = config.latent_dim
        self._weight_decay = config.weight_decay

        self._img_shape = (config.input_size,config.input_size)
        self._img_save_path = config.image_save_path
        self._model_save_path = config.model_save_path
        self._device = config.device

        # conditional
        self._n_classes = config.n_classes

        if self._work_type == 'train':
            # Loss function
            self._adversarial_criterion = torch.nn.MSELoss()

            # Initialize generator, encoder and discriminator
            # add n_classes
            self._G = Generator(self._latent_dim,self._n_classes,self._img_shape).to(self._device)
            self._E = Encoder(self._latent_dim,self._n_classes,self._img_shape).to(self._device)
            self._D = Discriminator(self._latent_dim,self._n_classes,self._img_shape).to(self._device)

            self._G.apply(self.weights_init)
            self._E.apply(self.weights_init)
            self._D.apply(self.weights_init)
            #self._D.apply(self.discriminator_weights_init)

            # self._G_optimizer = torch.optim.Adam(list(self._G.parameters())+list(self._E.parameters()),
            #                                     lr=self._gen_enc_lr,betas=(0.5,0.999))
            # self._G_optimizer = torch.optim.Adam(list(self._G.parameters())+list(self._E.parameters()),
            #                                     lr=self._gen_enc_lr,betas=(0.5,0.999),weight_decay=self._weight_decay)
            self._D_optimizer = torch.optim.Adam(self._D.parameters(),lr=self._discriminator_lr,betas=(0.5,0.999))
            self._G_optimizer = torch.optim.Adam(self._G.parameters(),
                                                                    lr=self._generator_lr,betas=(0.5,0.999))
            self._E_optimizer = torch.optim.Adam(self._E.parameters(),
                                                                    lr=self._encoder_lr,betas=(0.5,0.999))
            self._G_scheduler = lr_scheduler.ExponentialLR(self._G_optimizer, gamma= 0.99) 
            self._D_scheduler = lr_scheduler.ExponentialLR(self._D_optimizer, gamma= 0.99) 

    # def EG_loss(self,DG, DE, eps=1e-6):
    #     loss = torch.log(DG + eps) + torch.log(1 - DE + eps)
    #     return -torch.mean(loss)

    def E_loss(self,EC, labels):
        pass
        #loss = torch.log(DG + eps) + torch.log(1 - DE + eps)
        #return -torch.mean(loss)

    def G_loss(self,DG, DE, eps=1e-6):
        loss = torch.log(DG + eps) + torch.log(1 - DE + eps)
        return -torch.mean(loss)

    def D_loss(self,DG, DE, eps=1e-6):
        loss = torch.log(DE + eps) + torch.log(1 - DG + eps)
        return -torch.mean(loss)
        
    def train(self,train_loader):
        #Tensor = torch.cuda.FloatTensor if self._device == 'cuda' else torch.FloatTensor
        n_total_steps = len(train_loader)

        # if self._device == 'cuda':
        #     onehot_before_cod = torch.LongTensor([i for i in range(10)]).cuda() #0123456789
        # else:
        #     onehot_before_cod = torch.LongTensor([i for i in range(10)])
        # onehot = nn.functional.one_hot(onehot_before_cod, num_classes=10)
        # onehot = onehot.reshape(10,10,1,1).float()

        #fill = onehot.repeat(1,1,28,28)
        
        n_show = 10
        fixed_z = torch.randn(n_show, self._latent_dim)
        #fixed_z = 2 * torch.rand(n_show, self._latent_dim) - 1
        fixed_z = fixed_z.to(self._device)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        for epoch in range(self._epochs):
            self._D.train()
            self._E.train()
            self._G.train()

            # unpack images, labels
            for i, (images, labels) in enumerate(train_loader):
                # Adversarial ground truths
                #valid = Variable(Tensor(images.size(0), 1).fill_(1), requires_grad=False)
                #fake = Variable(Tensor(images.size(0), 1).fill_(0), requires_grad=False)
                label_real = torch.full((images.size(0),),1.0).to(self._device)
                label_fake = torch.full((images.size(0),),0.0).to(self._device)

                # Configure input
                images = images.to(self._device)
                labels = labels.to(self._device)
                onehot = nn.functional.one_hot(labels, num_classes=10).to(torch.float32)
                #one-hot
                #c_real = fill[labels].to(self._device)

                # ---------------------
                # Train Discriminator
                # ---------------------
                z = torch.randn(images.size(0), self._latent_dim)
                #z = 2 * torch.rand(images.size(0), self._latent_dim) - 1
                z = z.to(self._device)
                Gz =self._G(z, onehot)
                Ex, Ec = self._E(images)
                predict_encoder = self._D(images, Ex, torch.nn.functional.softmax(Ec))
                #gen_img = gen_img.reshape(-1,np.prod(self._img_shape)).to(self._device)
                predict_generator = self._D(Gz, z, onehot)
                
                loss_D_real = criterion(predict_encoder.view(-1),label_real)
                loss_D_fake = criterion(predict_generator.view(-1),label_fake)
                loss_D = loss_D_real + loss_D_fake
                #loss_D = self.D_loss(predict_generator, predict_encoder)                 
                
                self._D_optimizer.zero_grad()
                self._G_optimizer.zero_grad()
                self._E_optimizer.zero_grad()
                loss_D.backward()
                self._D_optimizer.step()
                #self._D_scheduler.step() 

                # ---------------------
                # Train Encoder
                # ---------------------
                Ex, Ec = self._E(images)
                gamma = min(self._encoder_alpha*np.exp(self._encoder_rho*epoch),self._encoder_phi)
                loss_E = torch.nn.functional.cross_entropy(Ec, labels)#self.E_loss(Ec, labels) 
                # loss_E = gamma * self.E_loss(Ec, labels) 
                # self._D_optimizer.zero_grad()
                # self._G_optimizer.zero_grad()
                # self._E_optimizer.zero_grad()
                # loss_E.backward()
                # self._E_optimizer.step()
                # ---------------------
                # Train Generator
                # ---------------------    
                # Sample noise as generator input
                #z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0],self._latent_dim))))
                # z_ is encoded latent vector
                # labels is character label
                #Ex, Ec = self._E(images)
                predict_encoder = self._D(images, Ex, torch.nn.functional.softmax(Ec))
                z = torch.randn(images.size(0), self._latent_dim)
                #z = 2 * torch.rand(images.size(0), self._latent_dim) - 1
                z = z.to(self._device)
                # labels
                Gz = self._G(z, onehot)
                predict_generator = self._D(Gz, z, onehot)
                loss_G_real = criterion(predict_encoder.view(-1),label_fake)
                loss_G_fake = criterion(predict_generator.view(-1),label_real)
                loss_G = loss_G_real + loss_G_fake
                #loss_G = self.G_loss(predict_generator, predict_encoder)
                loss_EG = gamma * loss_E + loss_G
                self._G_optimizer.zero_grad()
                self._D_optimizer.zero_grad()
                self._E_optimizer.zero_grad()
                loss_EG.backward()
                self._E_optimizer.step()    
                self._G_optimizer.step()         
                #self._G_scheduler.step()
                
                if i %100 == 0:
                    print (f'Epoch [{epoch+1}/{self._epochs}], Step [{i+1}/{n_total_steps}]')
                    #print (f'Generator Loss: {loss_EG.item():.4f} Discriminator Loss: {loss_D.item():.4f}')
                    print (f'Generator Loss: {loss_E.item():.4f} Encoder Loss: {loss_G.item():.4f} Discriminator Loss: {loss_D.item():.4f}')

 
                if i % 400 ==0:
                    vutils.save_image(Gz.cpu().data[:64, ], f'{self._img_save_path}/E{epoch+1}_Iteration{i}_fake.png')
                    vutils.save_image(images.cpu().data[:64, ], f'{self._img_save_path}/E{epoch+1}_Iteration{i}_real.png')
                    print('image saved')
                    print('')
            if epoch % 100==0:
                torch.save(self._G.state_dict(), f'{self._model_save_path}/netG_{epoch+1}epoch.pth')
                torch.save(self._E.state_dict(), f'{self._model_save_path}/netE_{epoch+1}epoch.pth')
                torch.save(self._D.state_dict(), f'{self._model_save_path}/netD_{epoch+1}epoch.pth')
            if (epoch + 1) % 1 == 0:
                n_show = 10
                self._D.eval()
                self._E.eval()
                self._G.eval()

                with torch.no_grad():
                    #generate images from same class as real ones
                    real = images[:n_show]
                    c = torch.zeros(n_show, 10, dtype=torch.float32).to(self._device)
                    c[torch.arange(n_show), labels[:n_show]] = 1#onehot
                    #c_real = fill[labels[:n_show]].to(self._device)
                    c = c.to(self._device) 
                    gener = self._G(fixed_z, c).reshape(n_show, 28, 28).cpu().numpy()
                    recon = self._G(self._E(real)[0], c).reshape(n_show, 28, 28).cpu().numpy()#.reshape(n_show, 28, 28).cpu().numpy()
                    real = real.reshape(n_show, 28, 28).cpu().numpy()

                    fig, ax = plt.subplots(3, n_show, figsize=(15,5))
                    fig.subplots_adjust(wspace=0.05, hspace=0)
                    plt.rcParams.update({'font.size': 20})
                    fig.suptitle('Epoch {}'.format(epoch+1))
                    fig.text(0.04, 0.75, 'G(z, c)', ha='left')
                    fig.text(0.04, 0.5, 'x', ha='left')
                    fig.text(0.04, 0.25, 'G(E(x), c)', ha='left')

                    for i in range(n_show):
                        ax[0, i].imshow(gener[i], cmap='gray')
                        ax[0, i].axis('off')
                        ax[1, i].imshow(real[i], cmap='gray')
                        ax[1, i].axis('off')
                        ax[2, i].imshow(recon[i], cmap='gray')
                        ax[2, i].axis('off')
                    plt.savefig(f'{self._img_save_path}/summary_E{epoch+1}.png')
                    #plt.show()
        
        torch.save(self._G.state_dict(), f'{self._model_save_path}/netG_last.pth')
        torch.save(self._E.state_dict(), f'{self._model_save_path}/netE_last.pth')
        torch.save(self._D.state_dict(), f'{self._model_save_path}/netD_last.pth')



    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        # if classname.find('BatchNorm') != -1:
        #     m.weight.data.normal_(1.0, 0.02)
        #     m.bias.data.fill_(0)
        # elif classname.find('Linear') != -1:
        #     m.bias.data.fill_(0)

    def discriminator_weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.5)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.bias.data.fill_(0)

class BiGAN(nn.Module):
    def __init__(self,config):
        super(BiGAN,self).__init__()

        self._work_type = config.work_type
        self._epochs = config.epochs
        self._batch_size = config.batch_size

        self._gen_enc_lr = config.gen_enc_lr
        self._discriminator_lr = config.discriminator_lr
        self._latent_dim = config.latent_dim
        self._weight_decay = config.weight_decay

        self._img_shape = (config.input_size,config.input_size)
        self._img_save_path = config.image_save_path
        self._model_save_path = config.model_save_path
        self._device = config.device

        if self._work_type == 'train':
            # Loss function
            self._adversarial_criterion = torch.nn.MSELoss()

            # Initialize generator, encoder and discriminator
            self._G = Generator(self._latent_dim,self._img_shape).to(self._device)
            self._E = Encoder(self._latent_dim,self._img_shape).to(self._device)
            self._D = Discriminator(self._latent_dim,self._img_shape).to(self._device)

            self._G.apply(self.weights_init)
            self._E.apply(self.weights_init)
            self._D.apply(self.discriminator_weights_init)

            #2021/12/10
            self._G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(self._G.parameters(), self._E.parameters())),
                                                lr=self._gen_enc_lr,betas=(0.5,0.999),weight_decay=self._weight_decay)
            # self._G_optimizer = torch.optim.Adam([{'params' : self._G.parameters()},{'params' : self._E.parameters()}],
            #                                     lr=self._generator_lr,betas=(0.5,0.999),weight_decay=self._weight_decay)
            self._D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._D.parameters()),lr=self._discriminator_lr,betas=(0.5,0.999))
            
            self._G_scheduler = lr_scheduler.ExponentialLR(self._G_optimizer, gamma= 0.99) 
            self._D_scheduler = lr_scheduler.ExponentialLR(self._D_optimizer, gamma= 0.99) 

    def train(self,train_loader):
        Tensor = torch.cuda.FloatTensor if self._device == 'cuda' else torch.FloatTensor
        n_total_steps = len(train_loader)
        for epoch in range(self._epochs):
            self._G_scheduler.step()
            self._D_scheduler.step()

            for i, (images, _) in enumerate(train_loader):
                # Adversarial ground truths
                valid = Variable(Tensor(images.size(0), 1).fill_(1), requires_grad=False)
                fake = Variable(Tensor(images.size(0), 1).fill_(0), requires_grad=False)

                
                # ---------------------
                # Train Encoder
                # ---------------------
                
                # Configure input
                images = images.reshape(-1,np.prod(self._img_shape)).to(self._device)

                # z_ is encoded latent vector
                (original_img,z_)= self._E(images)
                predict_encoder = self._D(original_img,z_)
  

                # ---------------------
                # Train Generator
                # ---------------------
                
                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0],self._latent_dim))))
                (gen_img,z)=self._G(z)
                predict_generator = self._D(gen_img,z)
                                                                                                               
                G_loss = (self._adversarial_criterion(predict_generator,valid)+self._adversarial_criterion(predict_encoder,fake)) *0.5   

                self._G_optimizer.zero_grad()
                G_loss.backward()
                self._G_optimizer.step()         

                # ---------------------
                # Train Discriminator
                # ---------------------

                z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0],self._latent_dim))))
                (gen_img,z)=self._G(z)
                (original_img,z_)= self._E(images)
                predict_encoder = self._D(original_img,z_)
                predict_generator = self._D(gen_img,z)

                D_loss = (self._adversarial_criterion(predict_encoder,valid)+self._adversarial_criterion(predict_generator,fake)) *0.5                
                
                self._D_optimizer.zero_grad()
                D_loss.backward()
                self._D_optimizer.step()

                

                
                if i % 100 == 0:
                    print (f'Epoch [{epoch+1}/{self._epochs}], Step [{i+1}/{n_total_steps}]')
                    print (f'Generator Loss: {G_loss.item():.4f} Discriminator Loss: {D_loss.item():.4f}')
 
                if i % 400 ==0:
                    vutils.save_image(gen_img.unsqueeze(1).cpu().data[:64, ], f'{self._img_save_path}/E{epoch}_Iteration{i}_fake.png')
                    vutils.save_image(original_img.unsqueeze(1).cpu().data[:64, ], f'{self._img_save_path}/E{epoch}_Iteration{i}_real.png')
                    print('image saved')
                    print('')
            if epoch % 100==0:
                torch.save(self._G.state_dict(), f'{self._model_save_path}/netG_{epoch}epoch.pth')
                torch.save(self._E.state_dict(), f'{self._model_save_path}/netE_{epoch}epoch.pth')
                torch.save(self._D.state_dict(), f'{self._model_save_path}/netD_{epoch}epoch.pth')





    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.bias.data.fill_(0)

    def discriminator_weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.5)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.bias.data.fill_(0)



