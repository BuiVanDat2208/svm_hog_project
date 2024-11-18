# Tệp chính để chạy chương trình

import sys

def main():
    sys.stdout.reconfigure(encoding='utf-8')  # Đặt encoding UTF-8
    from data_loader import load_mnist_data
    from feature_extractor import extract_hog_features
    from model import train_svm
    from metrics import evaluate_model

    # 1. Tải dữ liệu
    X_train, X_test, y_train, y_test = load_mnist_data(limit=10000)

    # 2. Trích xuất đặc trưng HOG
    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)

    # 3. Huấn luyện mô hình
    model = train_svm(X_train_hog, y_train)

    # 4. Dự đoán trên tập test
    print("Đang dự đoán trên tập kiểm tra...")
    y_pred = model.predict(X_test_hog)

    # 5. Đánh giá mô hình
    metrics = evaluate_model(y_test, y_pred)
    print("\nKết quả đánh giá mô hình:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()



