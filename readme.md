# Bidirectional GAN - Pytorch implementation
This is the pytorch implementation of Bidirectional GAN(BiGAN). Unlike ordinary GANs which are focused on generating data, the Bidirectional GAN is focused on creating an embedding of the original data. The model has an additional "Encoder" Structure from the original GAN which helps to encode original data. The descriminator will then discriminate the joint distribution of the latent vector and original data. The [paper](https://arxiv.org/abs/1605.09782) theoretically proves that the latent embedding will become an inverse mapping of the original input data when trained properly. 

# Implementation Notes
The model structure were referenced from this great [github1](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py) [github2](https://github.com/jaeho3690/BidirectionalGAN) repo. However, I made some minor adjustments as the implemented model struck on mode collapse.

### Requirements
```
Main dependencies (with python=3.6)  
pytorch = 1.6.0  
torchvision = 0.7.0  
```

### Installation
```
pip install requirements.txt
```

### ETL文字 Database
[etlcdb](http://etlcdb.db.aist.go.jp/?lang=ja)


### Run
```
python main.py
```

### Results
**Epoch 1**  
![Image1]()    
 