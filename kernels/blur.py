from torch import ones, tensor, Tensor
from torch import float32

class Blur:
    # Mean kernel
    mean_kernel = ones(3, 3) / 9.0  # Kích thước kernel 3x3
    mean_kernel = mean_kernel.type(float32)
    
    # Gaussian
    gaussian_kernel = tensor([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]], dtype = float32)
    gaussian_kernel = gaussian_kernel / 16  # Chia cho tổng của các phần tử trong kernel để chuẩn hóa

    # Kernel trung bình có trọng số
    weighted_mean_kernel = tensor([[0, 1, 0],
                                   [1, 4, 1],
                                   [0, 1, 0]], dtype = float32)
    weighted_mean_kernel = weighted_mean_kernel / 8  # Chia cho tổng các trọng số để chuẩn hóa

    @staticmethod
    def custom_mean_kernel(kernel_size: int = 3) -> Tensor:
        """
        Tạo một mean kernel (dùng để làm mờ ảnh).\n
        :kernel_size: Kích thước một cạnh kernel
        :return: Kernel làm mờ ảnh
        """
        kernel = ones(kernel_size, kernel_size) / (kernel_size ** 2)
        kernel = kernel.type(float32)
        return kernel
