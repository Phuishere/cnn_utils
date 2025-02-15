# cnn_utils

## Introduction:  
- Repo for CNN learning and some personal, useful utils for Pytorch and Matplotlib.
- Will add some other utils like dirs reading (os lib), training process plotting, metrics plotting, etc.
  
## Structure of the Repo:  
<pre>
cnn_utils/  
│  
├── kernels/                 # Thư mục chứa các kernels cơ bản  
│   ├── __init__.py          # File __init__ cho module  
│   ├── blur.py              # Các kernel làm mờ ảnh  
│   ├── edge_detection.py    # Các kernel giúp nhận diện cạnh của hình ảnh  
│   ├── sharpen.py           # Các kernel làm sắc nét ảnh  
│   ├── utils.py             # Các utils cho ảnh (forward, tensor4plt, normalize_tensor, plot_tensors)  
│  
├── main.ipynb               # File main  
│  
├── .gitignore               # File để bỏ qua các file không cần theo dõi trong Git  
└── README.md                # Thông tin về dự án  
</pre>
