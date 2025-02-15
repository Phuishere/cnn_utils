from torch import float32
from torch import Tensor, tensor, ones

class Sharpen:
    # Sharpen
    sharpen_kernel_v1 = tensor([[ 0, -1,  0],
                                [-1,  5, -1],
                                [ 0, -1,  0]])
    sharpen_kernel_v1 = sharpen_kernel_v1.type(float32)

    sharpen_kernel_v2 = tensor([[ 1,  1,  1],
                                [ 1, -7,  1],
                                [ 1,  1,  1]])
    sharpen_kernel_v2 = sharpen_kernel_v2.type(float32)

    @staticmethod
    def custom_sharpen(kernel_size: int = 3, centre_value: float = 5) -> Tensor:
        """
        Lưu ý: cần thêm nguồn liên quan đến việc đặt giá trị 0 cho các giá trị ngoại vi.\n
        :kernel_size: Kích thước kernel, là số lẻ
        :value: Giá trị custom cho giá trị trung tâm cho tensor
        :return: Trả về một kernel sharpen 3x3. Các giá trị ngoại vi sẽ được gán là 0
        """
        # Kernel size là số lẻ
        if kernel_size % 2 == 0:
            raise Exception("kernel_size phải là số lẻ!")
        
        # Các giá trị xung quanh trung tâm trái dấu để tạo sự tương phản
        if centre_value > 0:
            surround_value = -1
        else:
            surround_value = 1
        
        # Chuyển về dạng tensor
        centre_value = tensor(centre_value, dtype = float32)
        surround_value = tensor(surround_value, dtype = float32)
        
        # Tạo một ma trận đơn vị và thay giá trị ở giữa thành giá trị được đưa vào
        count = int((kernel_size - 3) / 2)
        kernel = ones(kernel_size, kernel_size)
        centre_coor = int((kernel_size - 1) / 2)
        kernel[centre_coor, centre_coor] = centre_value

        # Tạo kernel sharpen với kích thước đã cho
        for x in range(count):
            for y in range(count - x):
                kernel[x, y] = 0
                kernel[x, kernel_size - 1 - y] = 0
                kernel[kernel_size - 1 - x, y] = 0
                kernel[kernel_size - 1 - x, kernel_size - 1 - y] = 0
        kernel = kernel.type(float32)

        return kernel