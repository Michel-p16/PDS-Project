.. _evaluation_single_select:

======================================
Evaluation of Single Select Models
======================================

This section evaluates the **RoBERTa**, **DistilBERT**, and **TinyLlama** models for single-select questions. We analyze various metrics such as **Confusion Matrices**, **Classification Reports**, and other performance evaluations.


---------------------------------
Confusion Matrices
---------------------------------

The Confusion Matrices visualize classification accuracy per class:

- **Green numbers** along the diagonal indicate correctly classified instances.
- **Red numbers** outside the diagonal indicate misclassified instances.
- The **darker the color**, the higher the frequency of predictions for that class.

.. image:: _static/confusion_matrix_RoBERTa_single_select.png
   :align: center
   :width: 80%
   :alt: Confusion Matrix RoBERTa Single Select

.. image:: _static/confusion_matrix_DistilBERT_single_select.png
   :align: center
   :width: 80%
   :alt: Confusion Matrix DistilBERT Single Select

.. image:: _static/confusion_matrix_LLaMA_single_select.png
   :align: center
   :width: 80%
   :alt: Confusion Matrix LLaMA Single Select

---------------------------------
Code for Generating Confusion Matrices
---------------------------------

The following code generates confusion matrices for all models:

.. code-block:: python

    def plot_confusion_matrix(true_label, pred_label, model_name, labels):
        # Compute Confusion Matrix
        cm = confusion_matrix(true_label, pred_label)
        classes = list(labels.values())

        # Create a figure
        plt.figure(figsize=(15, 11))

        # Plot the heatmap without annotations
        ax = sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                        xticklabels=classes, yticklabels=classes, linewidths=0.5, linecolor="gray")

        # Overlay custom annotations with colors
        for i in range(cm.shape[0]):  
            for j in range(cm.shape[1]):  
                value = cm[i, j]
                color = "green" if i == j else ("red" if value > 0 else "black")
                ax.text(j + 0.5, i + 0.5, str(value), ha="center", va="center", color=color)

        # Adjust model name for display
        model_display_name = {"deepset/roberta-base-squad2": "RoBERTa",
                              "distilbert-base-uncased": "DistilBERT",
                              "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "LLaMA"}.get(model_name, model_name)

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix {model_display_name} SINGLE SELECT")

        # Save the plot
        plt.savefig(f"drive/MyDrive/CapStone_models/confusion_matrix_{model_display_name}_single_select.png", bbox_inches="tight", dpi=300)
        plt.show()

---------------------------------
Classification Reports
---------------------------------

The **Classification Reports** include key evaluation metrics:

- **Precision**: The proportion of predicted positive cases that are actually positive.
- **Recall**: The proportion of actual positive cases correctly predicted.
- **F1-Score**: The harmonic mean of precision and recall.
- **Support**: The number of true instances for each class.

Here are the classification reports for each model:

**RoBERTa Classification Report**

.. code-block:: text

    Precision    Recall  F1-Score   Support
    --------------------------------------
    1-10        1.00      1.00      1.00        21
    11-15       1.00      1.00      1.00        20
    Computers & Networks  0.91      0.88      0.89        24
    Construction Company  0.94      0.65      0.77        23
    Government   1.00      1.00      1.00        27
    SAP Sales Cloud  1.00      1.00      1.00        20
    Overall Accuracy: 0.97

**DistilBERT Classification Report**

.. code-block:: text

    Precision    Recall  F1-Score   Support
    --------------------------------------
    1-10        1.00      1.00      1.00        21
    11-15       1.00      1.00      1.00        20
    Computers & Networks  0.96      0.92      0.94        24
    Construction Company  0.84      0.91      0.88        23
    Government   1.00      1.00      1.00        27
    SAP Sales Cloud  1.00      1.00      1.00        20
    Overall Accuracy: 0.97

**TinyLlama Classification Report**

.. code-block:: text

    Precision    Recall  F1-Score   Support
    --------------------------------------
    1-10        1.00      1.00      1.00        21
    11-15       1.00      1.00      1.00        20
    Computers & Networks  1.00      0.88      0.93        24
    Construction Company  1.00      0.74      0.85        23
    Government   1.00      1.00      1.00        27
    SAP Sales Cloud  1.00      1.00      1.00        20
    Overall Accuracy: 0.97

---------------------------------
Evaluation Summary
---------------------------------

All three models achieved an accuracy of approximately **97%**, but there are some key differences:

- **RoBERTa** demonstrates high precision and recall across almost all categories.
- **DistilBERT** performs similarly but shows lower recall in some rare categories like **Construction Company**.
- **TinyLlama** has comparable results but exhibits lower recall in specific categories.

### Future Improvements

To enhance performance, the following strategies could be considered:

- **Balanced Training Data**: Ensuring equal representation of all classes.
- **Hyperparameter Tuning**: Optimizing learning rates, batch sizes, and loss functions.
- **Data Augmentation**: Expanding training data through synthetic examples.
- **Ensemble Models**: Combining multiple models to improve prediction robustness.

