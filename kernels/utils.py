from torch import Tensor, float32, tensor
from torch.nn import Conv2d
from torch import max as t_max
from torch import min as t_min
from matplotlib.pyplot import subplot, imshow, axis, title, show

def forward(image: Tensor, filter: Tensor) -> Tensor:
    """
    Thực hiện phép tích chập với một kernel có sẵn.\n
    :param image: Ảnh có shape (N, C, H, W)
    :param filter: Kernel (filter) dùng để xử lý ảnh, có 2 chiều (H, W)
    :return: Tensor ảnh đã được xử lý
    """
    # Đảm bảo filter là Tensor và điều chỉnh shape của nó
    if not isinstance(filter, Tensor):
        filter = tensor(filter)
    
    # Chuyển filter thành shape 4D, như (1, 1, 3, 3)
    filter = filter.view(1, 1, filter.shape[0], filter.shape[1])
    # Lặp lại filter cho mỗi kênh đầu ra (không lặp lại theo kênh đầu vào)
    filter = filter.repeat(image.shape[1], 1, 1, 1)
    filter = filter.type(float32)

    # Tạo convolution layer với groups = số kênh, để mỗi kênh xử lý độc lập
    kernel = Conv2d(
        in_channels=image.shape[1],
        out_channels=image.shape[1],
        kernel_size=3,
        padding="same",
        groups=image.shape[1],  # Mỗi kênh riêng biệt
        bias=False
    )
    kernel.weight.data = filter
    kernel.weight.requires_grad_(False)

    image = image.type(float32)
    image = kernel(image)
    return image

def normalize_tensor(tensor_img: Tensor) -> Tensor:
    """
    Chuẩn hóa ảnh về khoảng [0, 1].
    :param tensor_img: Tensor ảnh cần chuẩn hóa.
    :return: Tensor ảnh đã chuẩn hóa
    """
    tensor_img = (tensor_img - t_min(tensor_img))/(t_max(tensor_img) - t_min(tensor_img))
    return tensor_img

def tensor4plt(tensor_img: Tensor) -> Tensor:
    """
    Chuyển Tensor từ cấu trúc thường dùng trong Torch sang cấu trúc dùng cho Matplotlib.\n
    :param tensor_img: Tensor ảnh với kích thước (1, C, H, W).
    :return: Tensor ảnh với kích thước (H, W, C)
    """
    # Xử lý ảnh
    tensor_img = tensor_img.squeeze(0) # Loại bỏ chiều N (thường dùng cho batch size)
    tensor_img = tensor_img.permute(1, 2, 0) # Chuyển từ (C, H, W) thành (H, W, C) cho plt
    return tensor_img

def plot_tensors(column: int = 3, **tensor_imgs) -> None:
    """
    Plot một tensor ảnh bằng matplotlib.pyplot. Đã dùng qua hàm tensor4plt.\n
    :param img_per_row: Số ảnh được plot trên một hàng
    :param **tensor_dict: Tên của ảnh = Tensor ảnh với kích thước (N, C, H, W)
    :return: None
    """
    if len(tensor_imgs) == 1:
        for img_title, tensor_img in tensor_imgs:
            # Xử lý ảnh cho matplotlib
            tensor_img = tensor4plt(tensor_img)

            # Chuẩn hóa ảnh về khoảng [0, 1]
            out = (tensor_img - t_min(tensor_img))/(t_max(tensor_img) - t_min(tensor_img))

            print(out.shape)
            imshow(out.numpy())
            axis("off")
            show()
    else:
        # Xử lý thông tin subplot
        row = len(tensor_imgs) // column + 1
        
        # Plot các ảnh với subplot
        count = 1 # số thứ tự của ảnh (từ 1 đến len(tensor_imgs))
        for img_title, tensor_img in tensor_imgs.items():
            # Xử lý ảnh cho matplotlib
            tensor_img = tensor4plt(tensor_img)

            # Chuẩn hóa ảnh về khoảng [0, 1]
            out = normalize_tensor(tensor_img)

            # Plot
            subplot(row, column, count)
            imshow(out)
            axis("off")
            title(img_title)
            count += 1
        show()