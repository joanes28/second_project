import time
import sys
import pickle
def inference_time_and_environmental_impact_evaluation(model, X_test):
    """
    Perform inference time and environmental impact evaluation for a given machine learning model.

    Parameters:
    - model: Trained machine learning model.
    - X_test: Test features.

    Returns:
    - None (prints evaluation metrics).
    """

    def estimate_model_size(model):
        model_size = sys.getsizeof(pickle.dumps(model))

        model_size_gb = model_size / (1024**3)

        return model_size_gb

    def evaluate_model_efficiency(model, X_test):
        carbon_emission_factor = 0.865  # kgCO2/kwHr
        hardware_power_consumption = 0.06  # kw
        i = 0
        inference_times = []
        for _ in range(len(X_test)):
            start_time = time.time()
            _ = model.predict(X_test[:1])
            end_time = time.time()
            inference_times.append(end_time - start_time)
            i+=1

        average_inference_time = sum(inference_times) / len(inference_times)
        co2_emission_per_prediction = carbon_emission_factor * hardware_power_consumption * average_inference_time
        co2_emission_total = co2_emission_per_prediction*i
        model_size_gb = estimate_model_size(model)

        summary_paragraph = (
            f"Our platform evaluates the model's inference time and environmental impact. "
            f"\nThe average time taken to make an inference is measured and converted to CO2 emissions, assuming "
            f"\nCoal as a fuel for energy generation and Xeon 2.2 GHz Core. The mass of emitted CO2 by a single prediction is "
            f"\nestimated to be approximately {co2_emission_per_prediction:.4f} kgCO2. The estimated model storage size is {model_size_gb:.4f} GB."
        )
        return average_inference_time, co2_emission_per_prediction, co2_emission_total, model_size_gb, summary_paragraph


    ait, co2_per, co2_total, msize, sp = evaluate_model_efficiency(model, X_test)
    print(f"Average Inference Time: {ait:.4f} seconds")
    print(f"CO2 Emission per Prediction: {co2_per:.4f} kgCO2")
    print(f"CO2 Emission in total: {co2_total:.4f} kgCO2")
    print(f"Estimated Model Storage Size: {msize:.4f} GB")
    print("\n"+sp + "\n")
    result = {"Average Inference Time": ait,
              "CO2 Emission per Prediction": co2_per,
              "CO2 Emission in total": co2_total,
              "Estimated Model Storage Size": msize,
              "Summary":sp}
    print(result)
    return result