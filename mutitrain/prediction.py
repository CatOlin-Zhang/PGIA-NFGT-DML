import pandas as pd
import torch

from data_preprocessing import inverse_log_transform


def predict_and_submit(model, test_data, submission_path, preprocessor):
    test_data['TotalSF'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']
    X_test = test_data.drop(['Id'], axis=1)
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # Convert test data to tensor
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

    test_predictions_log = model.forward(X_test_tensor).flatten().detach().numpy()
    test_predictions = inverse_log_transform(test_predictions_log)
    submission_df = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_predictions})
    submission_df.to_csv(submission_path, index=False)



