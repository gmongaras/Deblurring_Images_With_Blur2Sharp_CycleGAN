import torch
from torch import DoubleStorage, nn
import torch.nn.functional as F



# Set the device and the default tensor type
device = torch.device("cuda")
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.autograd.set_detect_anomaly(True)




class Generator(nn.Module):
    # Initializes the structures needed to build the generator. The generator
    # has the following components:
    #  Encoder (with 3 convolution and 3 ReLU layers)
    #  R2sidual Block (with 3 convolution, 3 batch norm, and 2 ReLU layers)
    #  Decoder (with 3 deconvolution and 3 ReLU layers)
    def __init__(self):
        # The Encoder is composed of:
        #  Convolution layer (3 input filters, 64 output filters, kernel size of 7, stride of 1, padding of 3)
        #  ReLU layer
        #  Convolution layer (64 input filters, 128 output filters, kernel size of 3, stride of 2, padding of 1)
        #  ReLU layer
        #  Convolution layer (128 input filters, 256 output filters, kernel size of 3, stride of 2, padding of 1)
        #  ReLU layer
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            # Input is shape [3, 256, 256]
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            # Output is shape [64, 256, 256]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Output is shape [128, 128, 128]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Output is shape [256, 64, 64]
        ).cuda()

        # Residual blocks which are used to build the transformer. Each residual
        # block has the following structure:
        #  Convolution layer (256 input filters, 256 output filters, kernel size of 3, stride of 1, padding of 1)
        #  BatchNorm Layer
        #  ReLU layer
        #  Convolution layer (256 input filters, 256 output filters, kernel size of 3, stride of 1, padding of 1)
        #  BatchNorm Layer
        #  ReLU layer
        #  Convolution layer (256 input filters, 256 output filters, kernel size of 3, stride of 1, padding of 1)
        #  BatchNorm Layer
        self.resBlock = nn.Sequential(
            # Input is shape [256, 64, 64]
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            #nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            # Output is shape [256, 64, 64]
        ).cuda()

        # The Decoder is composed of:
        #  Convolution layer (256 input filters, 128 output filters, kernel size of 2, stride of 2, padding of 0)
        #  ReLU layer
        #  Convolution layer (128 input filters, 64 output filters, kernel size of 2, stride of 2, padding of 0)
        #  ReLU layer
        #  Convolution layer (64 input filters, 3 output filters, kernel size of 1, stride of 1, padding of 0)
        #  ReLU layer
        self.decoder = nn.Sequential(
            # Input is shape [256, 64, 64]
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            # Output is shape [128, 128, 128]
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            # Output is shape [64, 256, 256]
            nn.ConvTranspose2d(64, 3, kernel_size=1, stride=1),
            nn.ReLU(),
            # Output is shape [3, 256, 256]
        ).cuda()
    

    # The forward pass has the following components:
    #  Encoder (with 3 convolution and 3 ReLU layers)
    #  Transformer (with 12 residual blocks)
    #  Decoder (with 3 deconvolution and 3 ReLU layers)
    def forward(self, image):
        # Encoder
        x = self.encoder(image)

        # Transformer (12 residual blocks)
        x_prev = x.clone()
        for i in range(0, 12):
            x = self.resBlock(x) + x_prev
            x = nn.ReLU()(x)
            x_prev = x
        x = x_prev
        
        # Decoder
        x = self.decoder(x)

        return x


class Discriminator(nn.Module):
    # Initializes the discriminator structure. The discriminator
    # has the following structure respectively:
    #  Convolution layer (3 input filters, 64 output filters, kernel size of 4, stride of 2, padding of 1)
    #  Leaky ReLU with a slope of 0.2
    #  Convolution layer (64 input filters, 128 output filters, kernel size of 4, stride of 2, padding of 1)
    #  Leaky ReLU with a slope of 0.2
    #  Convolution layer (128 input filters, 256 output filters, kernel size of 4, stride of 2, padding of 1)
    #  Leaky ReLU with a slope of 0.2
    #  Convolution layer (256 input filters, 5 output filters, kernel size of 4, stride of 2, padding of 1)
    #  Leaky ReLU with a slope of 0.2
    #  Convolution layer (5 input filters, 1 output filters, kernel size of 3, stride of 1, padding of 1)
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input is shape [3, 256, 256]
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Input is shape [64, 128, 128]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Input is shape [128, 64, 64]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Input is shape [256, 32, 32]
            nn.Conv2d(256, 5, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Input is shape [5, 16, 16]
            nn.Conv2d(5, 1, kernel_size=3, stride=1, padding=1),
            # Output is shape [1, 16, 16]
        ).cuda()
    
    # Returns the forward pass for this network
    # Input:
    #  tensor of shape [3, 256, 256]
    # Output:
    #  tensor of shape [1, 16, 16]
    def forward(self, image):
        return self.disc(image)




"""
# Calculates the total loss function for discriminator
# (what the discriminator tries to minimize)
# Inputs:
#  fake_img - Output tensor from discriminator when the discriminator is given a fake image
#  real_img - Output tensor from discriminator when the discriminator is given a real image
# Outputs:
#  The loss for the discriminator
def DiscAdvLoss(fake_img, real_img):
    fake_img.detach()
    return torch.sum((fake_img**2) + ((real_img-torch.ones_like(real_img))**2))


# Loss function for generator (what the generator tries to minimize)
# Inputs:
#  Output tensor from discriminator when the discriminator is given a fake image
#    where a fake image is from the generator
# Outputs:
#  The loss for the generator
def GenAdvLoss(input_image):
    return torch.sum((input_image-torch.ones_like(input_image))**2)


# The cycle loss for the GAN
# Inputs:
#  Gen_XY - The generated output mapping the input image to an output image
#  Gen_YX - The generated output mapping Gen_XY to an output image which attempts
#           to look like the original input image into Gen_XY
#  X - The input to Gen_XY
#  Y - The output of Gen_XY and the input to Gen_YX
# Outputs:
#  The cycle loss
def CycleLoss(Gen_XY, Gen_YX, X, Y):
    return torch.sum((Gen_YX - X) + (Gen_XY - Y))


# The total loss for the generator
# Inputs:
#   realA - The image which generator A (the first generator) is trying to output.
#   realB - The image which generator B (the second generator) is trying to output.
#   fakeA - The first generator's output
#   fakeB - The second generator's output (the generator which gets fed fakeA)
#   discA - The output of discriminator A (the first discriminator)
#   discB - The output of discriminator B (the second discriminator)
def TotalGenLoss(realA, realB, fakeA, fakeB, discA, discB, lambda_cycle):
    adv_lossXY = GenAdvLoss(discA)
    adv_lossYX = GenAdvLoss(discB)
    cycle_loss = CycleLoss(fakeA, fakeB, realA, realB)


    # Return the total generator loss
    return (adv_lossXY+adv_lossYX) + lambda_cycle*cycle_loss
"""
# The loss for the discriminator when given a real image. The goal is
# to have the discriminator predict the image as real as opposed to fake,
# so the discriminator wants to predict these images as close to 1 as possible.
# Input:
#  disc - The discriminator object to calculate the loss.
#  realImg - The real image disc will make a prediction upon
# Output:
#  The adversarial loss for the discriminator
def advRealDiscLoss(disc, realImg):
    return torch.sum((disc(realImg)-1)**2)



# The loss for the discrminator when given a fake image. THe goal is to
# have the discriminator predict the image as fake as opposed to read,
# so the discriminator wants to predict these images as close to 0 as possible.
# Input:
#  disc - The discriminator object to calculate the loss.
#  fakeImg - The fake image disc will make a prediction upon
# Output:
#  The adversarial loss for the discriminator
def advFakeDiscLoss(disc, fakeImg):
    return torch.sum(disc(fakeImg)**2)



# The total loss for a discriminator for both fake and real images.
# Input:
#  disc - The discriminator object to calculate the loss.
#  fakeImg - The fake image disc will make a prediction upon
#  realImg - The real image disc will make a prediction upon
# Output:
#  The total loss for a discriminator
def totalDiscLoss(disc, fakeImg, realImg):
    fakeImg.detach()
    realImg.detach()
    return advFakeDiscLoss(disc, fakeImg)+advRealDiscLoss(disc, realImg)




# The loss for the generator. This value is calculated by wanting the given
# image to be as close to 1 as possible since 1 represents a real image
# and the genreator wants to fool the disriminator in thinking the image is real.
# Input:
#  discPredFake - The prediction from the discriminator given a fake image
# Output:
#  The total loss for a generator
def advGenLoss(discPredFake):
    return torch.sum((discPredFake-1)**2)



# The cycle loss is the loss that compares fake images to original images.
# The goal is to get the fake images as close as possible to the real images.
# so the difference between the original image and the fake image should be minimized.
# Input:
#  fakeImgY - The output from Gen_XY which is a fake image that should
#             resemble Y
#  fakeImgX - The output from Gen_YX which is a fake image that should
#             resemble X
#  Y - The real image that Gen_XY should predict
#  X - The real image that Gen_YX should predict
# Output:
#  The cycleLoss
def cycleLoss(fakeImgY, fakeImgX, Y, X):
    return torch.sum((fakeImgY-Y)**2+(fakeImgX-X)**2)



class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()

        # Create the generators and discriminators
        self.Gen_XY = Generator().to('cuda') # Generator from original image to fake new image
        self.Gen_YX = Generator().to('cuda') # Generator from image fake image to fake original image
        self.Disc_X = Discriminator().to('cuda') # Discriminator for fake original images
        self.Disc_Y = Discriminator().to('cuda') # Discriminator for fake new images

        # Create the optimizers
        self.Gen_opt = torch.optim.Adam(list(self.Gen_XY.parameters()) + list(self.Gen_YX.parameters()))
        self.Disc_X_opt = torch.optim.Adam(self.Disc_X.parameters())
        self.Disc_Y_opt = torch.optim.Adam(self.Disc_Y.parameters())
    
    def forward(self, X, Y):
        # Forward pass through the generator and discriminators
        self.Gen_XY_for = self.Gen_XY.forward(X)
        self.Disc_Y_for_fake = self.Disc_Y.forward(self.Gen_XY_for)
        self.Disc_Y_for_real = self.Disc_Y.forward(Y)
        self.Gen_YX_for = self.Gen_YX.forward(self.Gen_XY_for)
        self.Disc_X_for_fake = self.Disc_X.forward(self.Gen_YX_for)
        self.Disc_X_for_real = self.Disc_X.forward(X)

        return X


    def backward(self, X, Y):
        ####   Discriminator Loss    ####
        # Backward pass through discriminator Y which looks at image produced by
        # Gen_XY
        self.Disc_Y.zero_grad() # Zero out the gradient before backpropagation
        Disc_Y_Loss = totalDiscLoss(self.Disc_Y, self.Gen_XY_for, Y)
        Disc_Y_Loss.backward(retain_graph=True) # Update gradients
        print(Disc_Y_Loss)

        # Backward pass through discriminator X which looks at images produced
        # by Gen_YX
        self.Disc_X.zero_grad() # Zero out the gradient before backpropagation
        Disc_X_Loss = totalDiscLoss(self.Disc_X, self.Gen_YX_for, X)
        Disc_X_Loss.backward(retain_graph=True) # Update gradients
        print(Disc_X_Loss)
        



        ####    Generator Loss    ####
        # Update the generator generating fake new images (X->Y)
        self.Gen_XY.zero_grad() # Zero out the gradient before backpropagation
        Gen_XY_Loss = advGenLoss(self.Disc_Y_for_fake)
        Gen_XY_Loss.retain_grad()
        Gen_XY_Loss.backward(retain_graph=True) # Update gradients
        print(Gen_XY_Loss)

        # Update the generator generating fake original images (Y->X)
        self.Gen_YX.zero_grad() # Zero out the gradient before backpropagation
        Gen_YX_Loss = advGenLoss(self.Disc_X_for_fake)
        Gen_YX_Loss.retain_grad()
        Gen_YX_Loss.backward(retain_graph=True) # Update gradients
        print(Gen_YX_Loss)




        ####    CycleLoss    ####
        # Get the cycle loss to predict how close the predicted images are to
        # the real images.
        cycle_loss = cycleLoss(self.Gen_XY_for, self.Gen_YX_for, Y, X)
        cycle_loss.retain_grad()
        cycle_loss.backward(retain_graph=True) # Update gradients
        print(cycle_loss)




        ####    Update the weights and biases    ####
        self.Disc_Y_opt.step()
        self.Disc_X_opt.step()
        self.Gen_opt.step()





"""
Gen = Generator()
Disc = Discriminator()
realA = torch.rand(2, 3, 256, 256)
realB = torch.rand(2, 3, 256, 256)

fakeA = Gen.forward(torch.tensor(realA))
discA = Disc.forward(fakeA)
fakeB = Gen.forward(torch.tensor(fakeA))
discB = Disc.forward(fakeB)

print(TotalGenLoss(realA, realB, fakeA, fakeB, discA, discB, 1))
"""

X1 = torch.rand(2, 3, 256, 256, requires_grad=True)
Y = torch.rand(2, 3, 256, 256, requires_grad=True)

Cycle = CycleGAN()
for i in range(0, 1000):
    X = Cycle.forward(X1, Y)
    Cycle.backward(X, Y)