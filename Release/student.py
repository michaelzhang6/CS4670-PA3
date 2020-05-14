# Kaggle Team Name: Classic Winston and Cece Mess Around
# Michael Zhang maz62 Daniel Parangi dmp267
# Please place any imports here.
# BEGIN IMPORTS

import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchsummary import summary

# END IMPORTS

#########################################################
# BASELINE MODEL
#########################################################


class AnimalBaselineNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalBaselineNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        self.conv1 = nn.Conv2d(3, 6, 3, 2, 1)
        self.conv2 = nn.Conv2d(6, 12, 3, 2, 1)
        self.conv3 = nn.Conv2d(12, 24, 3, 2, 1)
        self.fc = nn.Linear(1536, 128)
        self.cls = nn.Linear(128, 16)
        self.relu = nn.ReLU()
        # TODO-BLOCK-END

    def forward(self, x):
        batch_size = x.size(0)
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(batch_size, 1536)
        x = self.relu(self.fc(x))
        x = self.cls(x)
        # TODO-BLOCK-END
        return x


def model_train(net, inputs, labels, criterion, optimizer):
    """
    Will be used to train baseline and student models.

    Inputs:
        net        network used to train
        inputs     (torch Tensor) batch of input images to be passed
                   through network
        labels     (torch Tensor) ground truth labels for each image
                   in inputs
        criterion  loss function
        optimizer  optimizer for network, used in backward pass

    Returns:
        running_loss    (float) loss from this batch of images
        num_correct     (torch Tensor, size 1) number of inputs
                        in this batch predicted correctly
        total_images    (float or int) total number of images in this batch

    Hint: Don't forget to zero out the gradient of the network before the backward pass. We do this before
    each backward pass as PyTorch accumulates the gradients on subsequent backward passes. This is useful
    in certain applications but not for our network.
    """
    # TODO: Foward pass
    # TODO-BLOCK-BEGIN
    predictions = net(inputs)
    predicted_labels = torch.argmax(predictions, dim=1)
    total_images = labels.size(0)
    num_correct = torch.eq(predicted_labels, labels.view(-1)).sum().item()
    # TODO-BLOCK-END

    # TODO: Backward pass
    # TODO-BLOCK-BEGIN
    optimizer.zero_grad()
    running_loss = criterion(predictions, labels.view(-1))
    running_loss.backward()
    optimizer.step()
    # TODO-BLOCK-END

    return running_loss, num_correct, total_images

#########################################################
# DATA AUGMENTATION
#########################################################


class Shift(object):
    """
  Shifts input image by random x amount between [-max_shift, max_shift]
    and separate random y amount between [-max_shift, max_shift]. A positive
    shift in the x- and y- direction corresponds to shifting the image right
    and downwards, respectively.

    Inputs:
        max_shift  float; maximum magnitude amount to shift image in x and y directions.
    """

    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W image as torch Tensor, shifted by random x
                          and random y amount, each amount between [-max_shift, max_shift].
                          Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape
        # TODO: Shift image
        # TODO-BLOCK-BEGIN
        image = image.transpose(1, 2, 0)
        x = random.randint(-self.max_shift, self.max_shift)
        y = random.randint(-self.max_shift, self.max_shift)
        M = np.float32([[1, 0, x], [0, 1, y]])
        image = cv2.warpAffine(image, M, (W, H)).transpose(2, 0, 1)

        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__


class Contrast(object):
    """
    Randomly adjusts the contrast of an image. Uniformly select a contrast factor from
    [min_contrast, max_contrast]. Setting the contrast to 0 should set the intensity of all pixels to the
    mean intensity of the original image while a contrast of 1 returns the original image.

    Inputs:
        min_contrast    non-negative float; minimum magnitude to set contrast
        max_contrast    non-negative float; maximum magnitude to set contrast

    Returns:
        image        3 x H x W torch Tensor of image, with random contrast
                     adjustment
    """

    def __init__(self, min_contrast=0.3, max_contrast=1.0):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W torch Tensor of image, with random contrast
                          adjustment
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Change image contrast
        # TODO-BLOCK-BEGIN
        alpha = random.uniform(self.min_contrast, self.max_contrast)
        for channel in range(3):
            mean = np.mean(image[channel])
            image[channel] = alpha * image[channel] + (1-alpha) * mean
            for row in range(H):
                for col in range(W):
                    if image[channel][row][col] > 1:
                        image[channel][row][col] = 1
                    if image[channel][row][col] < 0:
                        image[channel][row][col] = 0

        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__


class Rotate(object):
    """
    Rotates input image by random angle within [-max_angle, max_angle]. Positive angle corresponds to
    counter-clockwise rotation

    Inputs:
        max_angle  maximum magnitude of angle rotation, in degrees


    """

    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            rotated_image   image as torch Tensor; rotated by random angle
                            between [-max_angle, max_angle].
                            Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Rotate image
        # TODO-BLOCK-BEGIN
        angle = random.randrange(-self.max_angle, self.max_angle)
        image = image.transpose(1, 2, 0)
        M = cv2.getRotationMatrix2D((W//2, H//2), angle, 1)
        image = cv2.warpAffine(image, M, (W, H)).transpose(2, 0, 1)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__


class HorizontalFlip(object):
    """
    Randomly flips image horizontally.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be randomly rotated
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped horizontally with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        if random.random() < self.p:
            image = image.transpose(1, 2, 0)
            image = cv2.flip(image, 1).transpose(2, 0, 1)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

#########################################################
# STUDENT MODEL
#########################################################


def get_student_settings(net):
    """
    Return transform, batch size, epochs, criterion and
    optimizer to be used for training.
    """
    dataset_means = [123./255., 116./255.,  97./255.]
    dataset_stds = [54./255.,  53./255.,  52./255.]

    # TODO: Create data transform pipeline for your model
    # transforms.ToPILImage() must be first, followed by transforms.ToTensor()
    # TODO-BLOCK-BEGIN
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(
    ), transforms.RandomChoice([Shift(), Contrast(), Rotate(), HorizontalFlip()]), ])
    # transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),])
    # TODO-BLOCK-END

    # TODO: Settings for dataloader and training. These settings
    # will be useful for training your model.
    # TODO-BLOCK-BEGIN
    batch_size = 32
    # TODO-BLOCK-END

    # TODO: epochs, criterion and optimizer
    # TODO-BLOCK-BEGIN
    epochs = 4
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    # TODO-BLOCK-END

    return transform, batch_size, epochs, criterion, optimizer


class ResBlock(nn.Module):
    def __init__(self, cin, cout, halve):
        super(ResBlock, self).__init__()
        self.halve = halve
        self.conv2D_1 = nn.Conv2d(cin, cout, (3, 3), 1, 1)
        self.conv2D_2 = nn.Conv2d(cout, cout, (3, 3), 1, 1)
        self.conv2D_3 = nn.Conv2d(cout, cout, (3, 3), 1, 1)

        self.bn_1 = nn.BatchNorm2d(cout)
        self.bn_2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        # Cin x H x W
        y = self.conv2D_1(x)
        # Cout x H x W
        x = F.relu(self.bn_1(self.conv2D_2(y)))
        # Cout x H/2 x W/2
        x = F.relu(self.bn_2(self.conv2D_3(x))) + y
        if self.halve:
            x = F.max_pool2d(x, 2, 2)
        # Cout x H/2 x W/2
        return x


class AnimalStudentNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalStudentNet, self).__init__()
        bfc = 10
        self.conv2D_1 = nn.Conv2d(3, bfc, (3, 3), 1, 1)
        self.bn_1 = nn.BatchNorm2d(bfc)
        self.dp = nn.Dropout(p=0.4)

        self.rb_1 = ResBlock(bfc, bfc*2, True)
        self.rb_2 = ResBlock(bfc*2, bfc*4, True)
        self.rb_3 = ResBlock(bfc*4, bfc*8, True)
        self.rb_4 = ResBlock(bfc*8, bfc*16, True)

        self.fc_1 = nn.Linear(bfc*16*16, bfc*16)
        self.fc_2 = nn.Linear(bfc*16, num_classes)

    def forward(self, x):
        y = x.contiguous().view(-1, 3, 64, 64).float()
        # 3 x 64 x 64
        x = F.relu(self.bn_1(self.conv2D_1(y)))
        # 10 x 64 x 64
        x = self.rb_1(x)
        x = self.dp(x)
        # 20 x 32 x 32
        x = self.rb_2(x)
        x = self.dp(x)
        # 40 x 16 x 16
        x = self.rb_3(x)
        x = self.dp(x)
        # 80 x 8 x 8
        x = self.rb_4(x)
        x = self.dp(x)
        # 160 x 4 x 4
        (_, C, H, W) = x.size()
        x = x.view(-1, C * H * W)
        x = F.relu(self.fc_1(x))
        x = torch.sigmoid(self.fc_2(x))
        return x

#########################################################
# ADVERSARIAL IMAGES
#########################################################


def get_adversarial(img, output, label, net, criterion, epsilon):
    """
    Generates adversarial image by adding a small epsilon
    to each pixel, following the sign of the gradient.

    Inputs:
        img        (torch Tensor) image propagated through network
        output     (torch Tensor) output from forward pass of image
                   through network
        label      (torch Tensor) true label of img
        net        image classification model
        criterion  loss function to be used
        epsilon    (float) perturbation value for each pixel

    Outputs:
        perturbed_img   (torch Tensor, same dimensions as img)
                        adversarial image, clamped such that all values
                        are between [0,1]
                        (Clamp: all values < 0 set to 0, all > 1 set to 1)
        noise           (torch Tensor, same dimensions as img)
                        matrix of noise that was added element-wise to image
                        (i.e. difference between adversarial and original image)

    Hint: After the backward pass, the gradient for a parameter p of the network can be accessed using p.grad
    """

    # TODO: Define forward pass
    # TODO-BLOCK-BEGIN
    loss = criterion(output, label)
    loss.backward()

    signs = torch.sign(img.grad)
    noise = signs * epsilon

    perturbed_image = torch.clamp(img + noise, 0, 1)
    # TODO-BLOCK-END

    return perturbed_image, noise
