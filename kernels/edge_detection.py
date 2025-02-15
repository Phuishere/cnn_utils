from torch import float32
from torch import tensor

class Edge_detection:
    # Sobel kernel
    sobel_x = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]
    sobel_x = tensor(sobel_x).type(float32)

    sobel_y = [[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]]
    sobel_y = tensor(sobel_y).type(float32)

    # Prewitt
    prewitt_x = [[-1, 0, 1],
                 [-1, 0, 1],
                 [-1, 0, 1]]
    prewitt_x = tensor(prewitt_x).type(float32)

    prewitt_y = [[-1, -1, -1],
                 [ 0,  0,  0],
                 [ 1,  1,  1]]
    prewitt_y = tensor(prewitt_y).type(float32)

    # Laplacian
    laplacian_L8 = [[-1, -1, -1],
                 [-1,  8, -1],
                 [-1, -1, -1]]
    laplacian_L8 = tensor(laplacian_L8).type(float32)

    laplacian_L4 = [[0,  1, 0],
                   [1, -4, 1],
                   [0,  1, 0]]
    laplacian_L4 = tensor(laplacian_L4).type(float32)