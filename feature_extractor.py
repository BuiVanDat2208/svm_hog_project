# Trích xuất đặc trưng HOG
# HOG (Histogram of Oriented Gradients) là một phương pháp đặc trưng hình ảnh được sử dụng phổ biến trong nhận dạng đối tượng, đặc biệt trong nhận diện khuôn mặt và phát hiện đối tượng.
# Gradient: Gradient là sự thay đổi độ sáng của một pixel trong hình ảnh. Thường sử dụng các bộ lọc Sobel để tính toán độ gradient tại mỗi pixel trong hình ảnh.

# Quy trình tính toán HOG:
# Tính gradient: Tính toán gradient trong hình ảnh.
# Chia hình ảnh thành các cell: Chia hình ảnh thành các ô nhỏ (cell), mỗi cell có một histogram của các hướng gradient.
# Tính toán histogram cho mỗi cell: Tạo một histogram các hướng gradient cho mỗi cell.
# Chia các cell thành block và chuẩn hóa: Chia các cell thành các block và chuẩn hóa các đặc trưng của block.
# Kết hợp đặc trưng: Kết hợp các đặc trưng đã chuẩn hóa từ tất cả các block để tạo thành đặc trưng HOG tổng thể của hình ảnh.

import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features(images):
    """
    Trích xuất đặc trưng HOG từ một tập hợp ảnh.
    :param images: Tập hợp ảnh đầu vào.
    :return: Mảng đặc trưng HOG cho từng ảnh.
    """
    hog_features = []
    for img in images:
        # Chuyển ảnh thành ảnh grayscale nếu nó chưa phải grayscale
        if len(img.shape) > 2:  # Nếu ảnh có nhiều kênh màu (RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Đảm bảo ảnh có kích thước phù hợp
        img_resized = cv2.resize(img, (64, 64))  # Thay đổi kích thước tùy ý

        # Trích xuất đặc trưng HOG

        # img_resized: Ảnh đã được thay đổi kích thước và chuyển thành grayscale.

        # orientations=9: Chia các hướng gradient thành 9 khoảng, tương đương với 9 hướng khác nhau mà ta tính toán các gradient.
        
        # pixels_per_cell=(8, 8): Mỗi "cell" (ô nhỏ) trong ảnh được chia thành một khối 8x8 pixel, nghĩa là mỗi cell sẽ chứa các thông tin gradient trong phạm vi này.
        
        # cells_per_block=(2, 2): Mỗi "block" (khối) gồm có 2x2 cells. Việc chia ảnh thành các block và tính toán đặc trưng trên mỗi block giúp tăng tính ổn định của đặc trưng.
        
        # visualize=False: Tham số này cho biết có cần trả về hình ảnh minh họa của các đặc trưng HOG hay không
        fd = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

        # Lưu đặc trưng HOG của ảnh vào mảng
        hog_features.append(fd)
    
    return np.array(hog_features)
