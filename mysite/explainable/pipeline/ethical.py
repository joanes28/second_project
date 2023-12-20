import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import shap
from openai import OpenAI
import os
from django.conf import settings
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
client = OpenAI(api_key = "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")
def get_embedding(text, model="text-embedding-ada-002"):
    print(text)

    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model = model).data[0].embedding

def ethical_input_feature_analysis(model_data, X_test):
    def get_shap_value(model, X_test):
      explainer = shap.KernelExplainer(model.predict, shap.sample(X_test, 10) )
      shap_values = explainer.shap_values(X_test)
      shap.summary_plot(shap_values, X_test, show=False)
      plt.savefig(os.path.join(settings.MEDIA_ROOT, "shap-sumary.png"), dpi = 700)
      return shap_values

    def analyze_ethical_concerns(feature_description):
        try:
            embeddings_csv = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "embeddings.csv"))
        except:
            embeddings_csv = pd.DataFrame()
        embeddings = list(embeddings_csv.columns)
        if not feature_description in embeddings: # Check if the embedding has already been calculated
            embedding = get_embedding(feature_description)
            embeddings_csv[feature_description] = embedding
            embeddings.append(feature_description)
        embeddings_csv.to_csv(os.path.join(settings.MEDIA_ROOT, "embeddings.csv"), index = False)

        unethical_features = ["Gender", "Race", "Nationality", "Religion",
                              "Sexual orientation", "Age", "Disability",
                              "Socioeconomic status", "Education level",
                              "Marital status", "Ethnicity", "Genetic information",
                              "Political affiliation", "Immigration status",
                              "Health status", "Military status", "Indigenous identity",
                              "Language", "Family status", "Veteran status"]
        
        unethical_word_embeddings = []
        for word in unethical_features:
            if not word in embeddings:
                embedding = get_embedding(word)
                embeddings_csv[word] = embedding
                embeddings.append(embedding)

        embeddings_csv.to_csv(os.path.join(settings.MEDIA_ROOT, "embeddings.csv"), index = False)

        
        unethical_word_embeddings = np.transpose(np.array(embeddings_csv[unethical_features]))
        similarities = [np.dot(np.array(embeddings_csv[feature_description]), gender_embedding) for gender_embedding in unethical_word_embeddings]
        is_biased = sum(similarities) > 16
        return is_biased

    ethical_analysis_results = {'feature_description':[],'is_unethical': [] }
    X_test = pd.DataFrame(X_test)
    print(X_test.columns)
    feature_descriptions = list(X_test.columns)[1:]
    for feature_description in feature_descriptions:
        ethical_result = analyze_ethical_concerns(feature_description)
        # Append the analysis result for the current feature to the results list
        ethical_analysis_results['feature_description'].append(feature_description)
        ethical_analysis_results['is_unethical'].append(ethical_result)

    print(ethical_analysis_results)
    shap_vals = get_shap_value(model_data, X_test)
    print(shap_vals)
    #ethical_analysis_results["shap": shap_vals]
    # Return the list of ethical analysis results for each feature
    return ethical_analysis_results

