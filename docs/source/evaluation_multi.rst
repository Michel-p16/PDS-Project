========================
Evaluation of Multi-Select Models
========================

This page presents the evaluation of the **DistilBERT** and **RoBERTa** models for multi-select classification. These models have been trained to predict multiple labels for each input and are assessed using various performance metrics.

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
============
The multi-select models are designed to assign multiple labels to each input. Their performance is evaluated using the following key metrics:

- **F1 Score**: A balance between precision and recall.
- **Hamming Loss**: Measures the proportion of incorrect labels in multi-label classification.
- **Jaccard Similarity**: Assesses the similarity between predicted and actual labels.
- **Matthews Correlation Coefficient (MCC)**: Evaluates the overall classification quality.

.. _training-loss:

Training and Validation Loss
============================
The following graph illustrates the training and validation loss for the **DistilBERT Multi-Select Model**:

.. image:: loss_bert.png
   :alt: Training and validation loss for DistilBERT Multi-Select
   :align: center
   :width: 80%

The validation loss stabilizes after a few epochs, indicating that the model is learning effectively without overfitting.

.. _f1-score:

F1 Score Over Epochs
=====================
The graph below shows the progression of the F1 score for **DistilBERT** and **RoBERTa** over training epochs:

.. image:: f1.png
   :alt: F1 score over epochs for DistilBERT and RoBERTa
   :align: center
   :width: 80%

**DistilBERT** achieves high scores faster than **RoBERTa**, though both models improve steadily throughout training.

.. _hamming-loss:

Hamming Loss for Multi-Select Models
====================================
Hamming loss represents the proportion of incorrect labels assigned by the model:

.. image:: hamming.png
   :alt: Hamming loss for DistilBERT and RoBERTa Multi-Select
   :align: center
   :width: 80%

Lower values indicate better performance. **DistilBERT** has a lower Hamming loss than **RoBERTa**, meaning it makes fewer incorrect predictions.

.. _jaccard-similarity:

Jaccard Similarity for Multi-Select Models
==========================================
Jaccard similarity measures how closely the predicted labels match the true labels:

.. image:: jaccard.png
   :alt: Jaccard Similarity for DistilBERT and RoBERTa Multi-Select
   :align: center
   :width: 80%

Again, **DistilBERT** outperforms **RoBERTa**, demonstrating better alignment with the actual labels.

.. _mcc:

Matthews Correlation Coefficient (MCC)
======================================
MCC assesses the model's ability to distinguish between classes effectively:

.. image:: matthews.png
   :alt: MCC for DistilBERT and RoBERTa Multi-Select
   :align: center
   :width: 80%

Higher MCC values indicate better classification performance, with **DistilBERT** outperforming **RoBERTa**.

.. _roberta-metrics:

Metrics Over Epochs for RoBERTa
===============================
The following graph illustrates how accuracy, precision, recall, and F1 score evolve over time for the **RoBERTa Multi-Select Model**:

.. image:: roberta_metric.png
   :alt: Metrics for RoBERTa Multi-Select
   :align: center
   :width: 80%

The consistent improvement across these metrics indicates effective model optimization.

.. _dashboard:

Interactive Evaluation Dashboard
================================
A **dashboard** was created to visualize individual model predictions:

.. image:: dashboard1.png
   :alt: Dashboard example 1
   :align: center
   :width: 80%

.. image:: dashboard2.png
   :alt: Dashboard example 2
   :align: center
   :width: 80%

Users can input questions and compare predictions made by **DistilBERT** and **RoBERTa** interactively.

.. _code-evaluation:

Code for Model Evaluation
=========================
The following code was used to compute the evaluation metrics:

.. code-block:: python

    import pandas as pd
    import matplotlib.pyplot as plt

    multi_models = {
        "distilbert_multi": "./summary_distilbert_multi.csv",
        "roberta_multi": "./summary_roberta_multi.csv",
    }

    def process_metrics(csv_path, model_name):
        df = pd.read_csv(csv_path)

        print(f"Summary for {model_name}:\n", df.tail())

        df.to_csv(f"./summary_{model_name}.csv", index=False)

    for model_name, csv_path in multi_models.items():
        process_metrics(csv_path, model_name)

    def plot_f1(models):
        plt.figure(figsize=(10, 6))
        for model_name, csv_path in models.items():
            df = pd.read_csv(csv_path)
            plt.plot(df['epoch'], df['F1'], label=model_name)

        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("F1 Score over Epochs")
        plt.legend()
        plt.show()

    plot_f1(multi_models)

.. _classification-reports:

Classification Reports
======================
Below are the **final classification reports** for the multi-select models.

**DistilBERT Multi-Select Report**:

.. code-block:: text

    Summary Values for distilbert_multi:
    Epoch  | Training Loss | Validation Loss | Accuracy | F1 Score | Precision | Recall | Hamming Loss | Jaccard Similarity | MCC
    -----------------------------------------------------------------------------------------------
    1      | 0.1336        | 0.142971        | 0.000000 | 0.000000 | 0.000000  | 0.000000 | 0.037862     | 0.000000           | 0.000000
    5      | 0.0421        | 0.046828        | 0.395973 | 0.661642 | 0.787505  | 0.622074 | 0.017982     | 0.575056           | 0.725100
    10     | 0.0147        | 0.022184        | 0.758389 | 0.906225 | 0.948690  | 0.889632 | 0.006332     | 0.880984           | 0.911176
    30     | 0.0163        | 0.024724        | 0.785235 | 0.904898 | 0.950920  | 0.882943 | 0.006078     | 0.886130           | 0.914236

**RoBERTa Multi-Select Report**:

.. code-block:: text

    Summary Values for roberta_multi:
    Epoch  | Training Loss | Validation Loss | Accuracy | F1 Score | Precision | Recall | Hamming Loss | Jaccard Similarity | MCC
    -----------------------------------------------------------------------------------------------
    1      | 0.1392        | 0.146601        | 0.000000 | 0.000000 | 0.000000  | 0.000000 | 0.037862     | 0.000000           | 0.000000
    10     | 0.0517        | 0.057982        | 0.214765 | 0.468044 | 0.512239  | 0.474916 | 0.026212     | 0.400671           | 0.580371
    30     | 0.0163        | 0.024724        | 0.785235 | 0.904898 | 0.950920  | 0.882943 | 0.006078     | 0.886130           | 0.914236

Conclusion
==========
Both models demonstrate solid performance, with **DistilBERT** generally achieving higher scores across multiple evaluation metrics.
