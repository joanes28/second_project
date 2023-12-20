import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
class model_bias_and_variance_analysis():
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def model_bias_and_variance_analysis_classification(self):
        # Measure the bias in the dataset
        X_test = pd.DataFrame(self.X_test)
        feature_null_values_list = []
        for feature_name in self.X_test.columns:
            feature_null_values = np.count_nonzero(self.X_test[feature_name].isnull())
            feature_null_percentage = feature_null_values / len(self.X_test)
            feature_null_values_list.append(feature_null_percentage)
        # Evaluate the average absolute performance difference among the subsets of the testing after splitting based on the unique values of categorical features.
        category_bias_list = []
        for category in self.X_test.columns[self.X_test.dtypes == 'float64']:
            category_values = self.X_test[category].unique()
            category_bias = 0
            #print(category_values)
            for category_value in category_values:
                print(category_value)
                category_subset = self.X_test[self.X_test[category] == category_value]
                perturbed_predictions = self.model.predict(category_subset)
                perturbed_accuracy = accuracy_score(perturbed_predictions, self.y_test[:len(perturbed_predictions)])  # Adjusted here
                category_bias += abs(perturbed_accuracy - accuracy_score(self.y_test, self.model.predict(self.X_test))) / len(category_values)
                category_bias_list.append(category_bias)


        # Measure the variance in the dataset
        feature_perturbation_list = []
        for feature_name in self.X_test.select_dtypes(include='number').columns:
            feature_values = self.X_test[feature_name].values
            sorted_feature_values = sorted(feature_values)
            feature_perturbation = 0
            for i in range(len(feature_values)):
                perturbed_feature_values = feature_values.copy()
                perturbed_feature_values[i] = sorted_feature_values[(i + 1) // 2]
                perturbed_X_test = self.X_test.copy()
                perturbed_X_test.loc[:, feature_name] = perturbed_feature_values
                perturbed_predictions = self.model.predict(perturbed_X_test)
                perturbed_accuracy = accuracy_score(perturbed_predictions, self.y_test[:len(perturbed_predictions)])  # Adjusted here
                feature_perturbation += abs(perturbed_accuracy - accuracy_score(self.y_test, self.model.predict(self.X_test)))
            feature_perturbation_list.append(feature_perturbation)

        print("Feature Null Values:", feature_null_values_list)
        print("Category Bias List:", category_bias_list)
        print("Feature Perturbation List:", feature_perturbation_list)
        return {"feature_description":X_test.columns,"FeatureNull":feature_null_values_list, "CategoryBias": category_bias_list, "FeaturePerturbation":feature_perturbation_list}
    def model_bias_and_variance_analysis_regression(self):
        # Measure the bias in the dataset
        X_test = pd.DataFrame(self.X_test)
        feature_null_values_list = []
        for feature_name in self.X_test.columns:
            feature_null_values = np.count_nonzero(self.X_test[feature_name].isnull())
            feature_null_percentage = feature_null_values / len(self.X_test)
            feature_null_values_list.append(feature_null_percentage)

        # Evaluate the average absolute performance difference among the subsets of the testing after splitting based on the unique values of numerical features.
        numerical_bias_list = []
        for numerical_feature in self.X_test.select_dtypes(include='number').columns:
            feature_values = self.X_test[numerical_feature].values
            sorted_feature_values = sorted(feature_values)
            numerical_bias = 0
            for i in range(len(feature_values)):

                perturbed_feature_values = feature_values.copy()
                perturbed_feature_values[i] = sorted_feature_values[(i + 1) // 2]
                perturbed_X_test = self.X_test.copy()
                perturbed_X_test.loc[:, numerical_feature] = perturbed_feature_values
                perturbed_predictions = self.model.predict(perturbed_X_test)
                perturbed_mae = mean_squared_error(perturbed_predictions, self.y_test[:len(perturbed_predictions)])  # Adjusted here
                numerical_bias += abs(perturbed_mae - mean_squared_error(self.y_test, self.model.predict(self.X_test)))
            numerical_bias_list.append(numerical_bias)
        handiena = np.max(numerical_bias_list)
        numerical_bias_list /= handiena
        numerical_bias_list = list(numerical_bias_list)

        # Measure the variance in the dataset
        feature_perturbation_list = []
        for feature_name in self.X_test.select_dtypes(include='number').columns:
            feature_values = self.X_test[feature_name].values
            sorted_feature_values = sorted(feature_values)
            feature_perturbation = 0
            for i in range(len(feature_values)):
                perturbed_feature_values = feature_values.copy()
                perturbed_feature_values[i] = sorted_feature_values[(i + 1) // 2]
                perturbed_X_test = self.X_test.copy()
                perturbed_X_test.loc[:, feature_name] = perturbed_feature_values
                perturbed_predictions = self.model.predict(perturbed_X_test)
                perturbed_mae = mean_squared_error(perturbed_predictions, self.y_test[:len(perturbed_predictions)])  # Adjusted here
                feature_perturbation += abs(perturbed_mae - mean_squared_error(self.y_test, self.model.predict(self.X_test)))
            feature_perturbation_list.append(feature_perturbation)

        print("Feature Null Values:", feature_null_values_list)
        print("Numerical Bias List:", numerical_bias_list)
        print("Feature Perturbation List:", feature_perturbation_list)
        return {"feature_description":X_test.columns, "FeatureNull":feature_null_values_list, "CategoryBias": feature_null_values_list, "FeaturePerturbation":feature_perturbation_list}
