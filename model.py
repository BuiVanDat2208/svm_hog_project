# Định nghĩa và huấn luyện mô hình SVM
# SVM (Support Vector Machine) là một thuật toán học máy dùng để phân loại dữ liệu. 
from sklearn.svm import SVC

def train_svm(X_train, y_train):
# X_train: Dữ liệu huấn luyện, chứa các đặc trưng HOG đã được trích xuất từ các ảnh.
# y_train: Nhãn tương ứng của các ảnh, trong trường hợp này là các chữ số (0-9) từ bộ dữ liệu như MNIST.
    """
    Huấn luyện mô hình SVM.
    :param X_train: Dữ liệu huấn luyện (đặc trưng HOG).
    :param y_train: Nhãn tương ứng.
    :return: Mô hình SVM đã được huấn luyện.
    """
    print("Đang huấn luyện mô hình SVM...")
    svm = SVC(kernel='linear', random_state=42)

    # random_state=42: Thiết lập hạt giống (seed) cho quá trình sinh số ngẫu nhiên, giúp đảm bảo tính tái lập được của mô hình.

    svm.fit(X_train, y_train)
    # svm.fit(X_train, y_train): Đây là bước huấn luyện mô hình SVM. Hàm fit() sẽ sử dụng dữ liệu huấn luyện (X_train) và nhãn (y_train) để tìm ra siêu phẳng tối ưu phân chia các lớp (các chữ số từ 0-9 trong trường hợp này).

    return svm
