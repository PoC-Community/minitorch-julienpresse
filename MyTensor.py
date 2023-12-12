import sys
import numpy as np  # Add this line
import torch as t
import torchvision as tv
from tests import MyCrossBackward

class MyTensor:
    def __init__(self, data: list | np.ndarray | int | float, dtype: np.dtype = None, requires_grad: bool = False, crossOpBack: MyCrossBackward = None) -> None:
        if isinstance(data, list):
            data = np.array(data)

        if isinstance(data, np.ndarray):
            self.__data = data
        elif isinstance(data, (int, float)):
            self.__data = np.array([data])
        else:
            raise ValueError("Invalid data type provided")

        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad = None
        self.backward_op = crossOpBack

    def add_backward(base, *tensors):
        left, right = tensors
        if left.requires_grad:
            left.backward_op(base.grad, left.grad)

        if right.requires_grad:
            right.backward_op(base.grad, right.grad)

    def add(self, other):
        result_data = self.__data + other.__data
        result_tensor = MyTensor(result_data, requires_grad=self.requires_grad)

        if self.requires_grad or other.requires_grad:
            backward_op = MyCrossBackward(self, other, backward_function=self.add_backward, name="AddBackward")
            result_tensor.backward_op = backward_op
            result_tensor.grad = result_tensor.grad and result_tensor.grad + 1 or 1

        return result_tensor

    @property
    def shape(self):
        return self.__data.shape

    @property
    def strides(self):
        return self.__data.strides

    @property
    def data(self):
        return self.__data

    @property
    def size(self):
        return self.__data.size

    def __str__(self) -> str:
        return f"MyTensor({self.__data}, requires_grad={self.requires_grad})"

    def __getitem__(self, key):
        return self.__data[key]

    def __setitem__(self, key, value):
        self.__data[key] = value

tensor_a = MyTensor([3, 4, 5], requires_grad=True)
tensor_b = MyTensor([1, 2, 3], requires_grad=True)

# Perform the add operation
tensor_c = tensor_a.add(tensor_b)

# Check the result and gradients
print( tensor_c)
