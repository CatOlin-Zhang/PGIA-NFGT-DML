import numpy as np
from data_preprocessing import load_data, preprocess_data, log_transform, inverse_log_transform
from model_training import MLPModel, batch_gradient_descent, evaluate_model
from prediction import predict_and_submit

def main():
    # 加载数据
    train_path = 'train.csv'
    test_path = 'test.csv'
    train_data, test_data = load_data(train_path, test_path)

    # 为训练数据和测试数据分别添加新特征
    train_data['TotalSF'] = train_data['TotalBsmtSF'] + train_data['1stFlrSF'] + train_data['2ndFlrSF']
    test_data['TotalSF'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']

    # 数据预处理
    preprocessor, numeric_features, categorical_features = preprocess_data(train_data)
    X_train_processed = preprocessor.fit_transform(train_data)
    X_test_processed = preprocessor.transform(test_data)

    # 将稀疏矩阵转换为密集矩阵
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # 对目标变量进行对数变换
    y_train_log = log_transform(train_data['SalePrice'].values)

    # 创建MLP模型
    input_size = X_train_processed.shape[1]
    hidden_sizes = [64, 32]  # 两层隐藏层，每层分别有64和32个神经元
    output_size = 1
    model = MLPModel(input_size, hidden_sizes, output_size)

    # 调试信息：打印输入数据的形状
    print(f'Training data shape: {X_train_processed.shape}')
    print(f'Test data shape: {X_test_processed.shape}')

    # 提取特征名称
    feature_names_numeric = preprocessor.named_transformers_['num'].named_steps['scaler'].get_feature_names_out(numeric_features)
    feature_names_categorical = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = list(feature_names_numeric) + list(feature_names_categorical)

    print(f'Feature names length: {len(feature_names)}')
    print(f'First few feature names: {feature_names[:10]}')

    # 训练模型
    model = batch_gradient_descent(model, X_train_processed, y_train_log, learning_rate=0.0001, epochs=1000, batch_size=16)

    # 评估模型
    val_size = int(0.2 * len(y_train_log))
    X_val_processed = X_train_processed[-val_size:]
    y_val_log = y_train_log[-val_size:]
    evaluate_model(model, X_val_processed, y_val_log)

    # 预测并生成提交文件
    submission_path = 'submission.csv'
    predict_and_submit(model, test_data, submission_path, preprocessor)

if __name__ == "__main__":
    main()



