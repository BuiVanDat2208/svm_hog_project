# Xử lý tải và chuẩn bị dữ liệu
# MNIST có các ứng dụng chính như:
    # Nhận dạng chữ số viết tay
    # Kiểm tra các thuật toán học máy: Bộ dữ liệu này chứa 60,000 ảnh huấn luyện và 10,000 ảnh kiểm tra, mỗi ảnh là một chữ số viết tay từ 0 đến 9, kích thước 28x28 pixel với độ xám
    # Giáo dục và nghiên cứu: Bộ dữ liệu này được sử dụng rộng rãi trong nghiên cứu và học tập về học máy, đặc biệt là trong các khóa học, sách giáo khoa và các bài báo nghiên cứu.

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist_data(limit=10000):
    """
    Tải dữ liệu MNIST và chia tập train/test.
    :param limit: Số lượng mẫu dữ liệu cần tải.
    :return: X_train, X_test, y_train, y_test
    """
    print("Loading MNIST dataset...")

    # 'mnist_784': Đây là tên bộ dữ liệu trên OpenML. Bộ dữ liệu MNIST có 784 đặc trưng (mỗi đặc trưng đại diện cho một pixel trong ảnh 28x28).
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)  # Trả về NumPy arrays

    X = mnist.data.astype(np.float32)[:limit]  # Chuyển dữ liệu về kiểu float32
    # .astype(np.float32): Dữ liệu của MNIST được lưu trữ dưới dạng số nguyên, nhưng dòng này chuyển các giá trị này thành kiểu dữ liệu float32 (số thực 32 bit).

    y = mnist.target.astype(int)[:limit]       # Nhãn dưới dạng số nguyên
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
    # test_size=0.2: 20% của bộ dữ liệu sẽ được sử dụng để kiểm tra, phần còn lại (80%) sẽ dùng để huấn luyện mô hình.
    # random_state=42: Đây là hạt giống (seed) cho phép đảm bảo rằng việc chia dữ liệu sẽ có thể tái lặp lại, giúp kết quả kiểm tra có thể lặp lại được khi thực hiện lại mã.

