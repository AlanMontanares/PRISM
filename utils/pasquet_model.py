import torch
import torch.nn as nn


class AvgPool2dSamePadding(nn.Module):
    def __init__(self, kernel_size, stride):
        super(AvgPool2dSamePadding, self).__init__()
        self.kernel_size  = kernel_size
        self.stride  = stride

    def forward(self, input):
        input = nn.functional.pad(input, (0, 1, 0, 1), mode="replicate")
        return nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)(input)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, nbS1, nbS2, with_kernel_5=True):
        super(InceptionBlock, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels, nbS1, kernel_size=1, stride=1, padding="same")
        self.conv1_3 = nn.Conv2d(in_channels, nbS1, kernel_size=1, stride=1, padding="same")

        self.conv2_1 = nn.Conv2d(in_channels, nbS2, kernel_size=1, stride=1, padding="same")
        self.conv2_2 = nn.Conv2d(nbS1, nbS2, kernel_size=3, stride=1, padding="same")

        if with_kernel_5:
            self.conv1_2 = nn.Conv2d(in_channels, nbS1, kernel_size=1, stride=1, padding="same")
            self.conv2_3 = nn.Conv2d(nbS1, nbS2, kernel_size=5, stride=1, padding="same")

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)#AvgPool2dSamePadding(kernel_size=2, stride=1)
        
        self.prelu1 = nn.PReLU(num_parameters=nbS1, init=0.0)
        self.prelu2 = nn.PReLU(num_parameters=nbS2, init=0.0)

        self.with_kernel_5 = with_kernel_5

    def forward(self,x):
        s1_1 = self.prelu1(self.conv1_1(x))
        s1_3 = self.prelu1(self.conv1_3(x))

        s2_1 = self.prelu2(self.conv2_1(x))
        s2_2 = self.prelu2(self.conv2_2(s1_1))
        
        s2_4 = self.avgpool(s1_3)
        
        if self.with_kernel_5:
            s1_2 = self.prelu1(self.conv1_2(x))
            s2_3 = self.prelu2(self.conv2_3(s1_2))
            
            return torch.cat((s2_1, s2_2, s2_3, s2_4), dim=1)
        
        else:
            return torch.cat((s2_1, s2_2, s2_4), dim=1)



class Pasquet_backbone(nn.Module):
    def __init__(self, in_channels):
        super(Pasquet_backbone, self).__init__()

        self.convs = nn.Sequential(
                                nn.Conv2d(in_channels, 64, kernel_size=5, padding="same"),
                                nn.PReLU(num_parameters=64, init=0.0),
                                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                InceptionBlock(64,nbS1=48, nbS2=64, with_kernel_5=True),
                                InceptionBlock(240,nbS1=64, nbS2=92, with_kernel_5=True),
                                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                InceptionBlock(340,nbS1=92, nbS2=128, with_kernel_5=True),
                                InceptionBlock(476,nbS1=92, nbS2=128, with_kernel_5=True),
                                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                InceptionBlock(476,nbS1=92, nbS2=128, with_kernel_5=False),
                                nn.Flatten()
                                )


    def forward(self, x):

        original_shape = x.shape
        new_shape = original_shape[:-4] + (-1,)

        leading = torch.prod(
            torch.tensor(original_shape[:-3])
        ).item()  # Batch*Transforms*Levels

        x = x.reshape(
            leading, original_shape[-3], original_shape[-2], original_shape[-1]
        )

        x = self.convs(x)
        x = x.reshape(*new_shape)

        return x