#  # Tính toán các chỉ số đánh giá
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# def evaluate_model(y_true, y_pred):
#     """
#     Tính toán các chỉ số đánh giá: Accuracy, Precision, Recall, F1 Score.
#     :param y_true: Nhãn thực tế.
#     :param y_pred: Nhãn dự đoán.
#     :return: Từ điển chứa các chỉ số.
#     """
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='weighted')
#     recall = recall_score(y_true, y_pred, average='weighted')
#     f1 = f1_score(y_true, y_pred, average='weighted')

#     return {
#         "Accuracy": accuracy,
#         "Precision": precision,
#         "Recall": recall,
#         "F1 Score": f1
#     }


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_true, y_pred):
    """
    Tính toán các chỉ số đánh giá: Accuracy, Precision, Recall, F1 Score.
    :param y_true: Nhãn thực tế.
    :param y_pred: Nhãn dự đoán.
    :return: Từ điển chứa các chỉ số.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Tạo từ điển chứa các chỉ số đánh giá
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    # Vẽ biểu đồ các chỉ số đánh giá
    plot_metrics(metrics)

    return metrics

def plot_metrics(metrics):
    """
    Vẽ biểu đồ cột cho các chỉ số đánh giá mô hình.
    :param metrics: Từ điển chứa các chỉ số đánh giá (Accuracy, Precision, Recall, F1 Score).
    :return: None
    """
    # Các chỉ số và giá trị tương ứng
    labels = list(metrics.keys())
    values = list(metrics.values())

    # Tạo biểu đồ cột
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['blue', 'green', 'orange', 'red'])
    
    # Thêm tiêu đề và nhãn
    plt.title('Model Evaluation Metrics')
    plt.ylim(0, 1)  # Giới hạn trục Y từ 0 đến 1
    plt.xlabel('Metrics')
    plt.ylabel('Scores')

    # Hiển thị biểu đồ
    plt.show()

# Ví dụ sử dụng evaluate_model và vẽ biểu đồ
# Giả sử bạn đã có các dự đoán từ mô hình và nhãn thực tế
# y_true = [dữ liệu thực tế]
# y_pred = [dữ liệu dự đoán]

# Call hàm evaluate_model với y_true và y_pred
# metrics = evaluate_model(y_true, y_pred)
