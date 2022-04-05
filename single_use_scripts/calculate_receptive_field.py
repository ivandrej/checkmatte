import sys

from torch.nn.modules.conv import Conv2d

sys.path.append("..")
from model.smaller_mobilenetv3 import MobileNetV3LargeEncoder

class ReceptiveFieldCalculator:

    def __init__(self):
        self.r = 1  # receptive field
        self.sp = 1  # product of strides

    def f(self, module):
        if type(module) == Conv2d:
            assert (module.kernel_size[0] == module.kernel_size[1])

            kernel_size = module.kernel_size[0]
            stride = module.stride[0]
            dilation = module.dilation

            self.r = self.r + (kernel_size - 1) * self.sp
            self.sp *= stride

            if kernel_size > 1:
                print(f"Kernel size: {kernel_size}, stride: {stride}, "
                      f"channels: {module.in_channels}->{module.out_channels}")
                print("Receptive field: ", self.r)
                # print("Product of strides: ", self.sp)

    def calc(self, module):
        module.apply(self.f)

encoder = MobileNetV3LargeEncoder().eval().cuda()
ReceptiveFieldCalculator().calc(encoder)
