from CIFAR10_LeNet import LeNet as lenet_cifar
from MNIST_LeNet import LeNet as lenet_mnist
from TrainingUtils import train, get_device

__all__ = ["lenet_cifar", "lenet_mnist", "train", "get_device"]
