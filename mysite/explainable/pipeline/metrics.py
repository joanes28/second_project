from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from torch import nn
class performance_evaluation():
    """
    Perform comprehensive performance evaluation for a given machine learning model.

    Parameters:
    - model: Trained machine learning model.
    - X_test: Test features.
    - y_test: Test labels.

    Returns:
    - None (prints evaluation metrics).
    """
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test 
        self.y_test = y_test
        self.preds = self.model.predict(X_test)
        try:            
            self.pred_probs = self.model.predict_proba(X_test)
        except:
            self.pred_probs = None
    def classification_metrics(self):
        print("\nClassification Metrics:")
        accuracy = accuracy_score(self.y_test, self.preds)
        roc_auc = roc_auc_score(self.y_test, self.preds)
        f1 = f1_score(self.y_test, self.preds)
        cmatrix = confusion_matrix(self.y_test, self.preds)
        TP = cmatrix[1, 1]
        FP = cmatrix[0, 1]
        FN = cmatrix[1, 0]
        TN = cmatrix[0, 0]
        fpr = FP / (FP + TN)
        fnr = FN / (TP + FN)
        tpr = TP / (TP + FN)
        tnr = TN / (FP + TN)

        if type(self.pred_probs) is np.ndarray:
            loss = nn.CrossEntropyLoss()

            self.y_test_tensor = torch.LongTensor(self.y_test.values)  # Convert to PyTorch tensor
            pred_proba_tensor = torch.Tensor(self.pred_probs)
            cross_entropy_loss = loss(pred_proba_tensor, self.y_test_tensor)
        else:
            cross_entropy_loss =  None

        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"True Positive Rate: {tpr:.4f}")
        print(f"True Negative Rate: {tnr:.4f}")
        print(f"False Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
        print(f"Cross entropy loss: {cross_entropy_loss:.4f}")
        print("\nConfusion Matrix:")
        print(cmatrix)

        results = {"Accuracy":accuracy, 
                "AUC": roc_auc,
                "F1 Score": f1,
                "TP":tpr,
                "TN":tnr,
                "FP":fpr,
                "FN":fnr,
                "cross Entropy":cross_entropy_loss.item(),
                "Confusion Matrix":cmatrix }
        return results

    def regression_metrics(self):
        print("\nRegression Metrics:")
        mae = mean_absolute_error(self.y_test, self.preds)
        mse = mean_squared_error(self.y_test, self.preds)
        rmse = np.sqrt(mse)
        mbd = np.mean(self.y_test) - np.mean(self.preds)
        r2 = r2_score(self.y_test, self.preds)

        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Mean Bias Deviation: {mbd:.4f}")
        print(f"R-Squared: {r2:.4f}")

        results = {"MAE":mae, 
                "MSE":  mse,
                "RMSE": rmse,
                "MBD":mbd,
                "R2":r2}
        return results
