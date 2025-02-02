.. _evaluation_single_select:

======================================
Evaluation of Single Select Models
======================================

This section evaluates the **RoBERTa**, **DistilBERT**, and **TinyLlama** models for single-select questions. Each model is analyzed separately with its **Confusion Matrix**, key classification metrics, and an assessment of its performance.

---------------------------------
RoBERTa Evaluation
---------------------------------

RoBERTa is known for its **robust performance on NLP tasks** due to its large-scale pretraining and deep architecture.

### Confusion Matrix - RoBERTa

The confusion matrix below shows the performance of **RoBERTa** on single-select classification.

.. image:: _static/confusion_matrix_RoBERTa_single_select.png
   :align: center
   :width: 80%
   :alt: Confusion Matrix RoBERTa Single Select

**Analysis:**
- RoBERTa achieves near **perfect classification** in most categories.
- Some **misclassifications** occur in similar categories such as **Computers & Networks** and **Construction Companies**.
- The **majority of predictions align correctly**, indicating high model accuracy.

### Key Metrics - RoBERTa

+-------------------+-------+
| Metric           | Value |
+===================+=======+
| **Accuracy**     | 0.97  |
+-------------------+-------+
| **F1 Score**     | 0.97  |
+-------------------+-------+
| **Macro Avg**    | 0.97  |
+-------------------+-------+
| **Weighted Avg** | 0.97  |
+-------------------+-------+

---------------------------------
DistilBERT Evaluation
---------------------------------

DistilBERT is a **lighter and faster model** compared to RoBERTa, making it a suitable choice for real-time applications.

### Confusion Matrix - DistilBERT

.. image:: _static/confusion_matrix_DistilBERT_single_select.png
   :align: center
   :width: 80%
   :alt: Confusion Matrix DistilBERT Single Select

**Analysis:**
- DistilBERT exhibits strong classification performance but slightly more **misclassifications** than RoBERTa.
- Some **errors occur in overlapping categories**, such as **Computers & Networks**.
- The **majority of diagonal values remain high**, showing that most classifications are correct.

### Key Metrics - DistilBERT

+-------------------+-------+
| Metric           | Value |
+===================+=======+
| **Accuracy**     | 0.97  |
+-------------------+-------+
| **F1 Score**     | 0.97  |
+-------------------+-------+
| **Macro Avg**    | 0.97  |
+-------------------+-------+
| **Weighted Avg** | 0.97  |
+-------------------+-------+

---------------------------------
TinyLlama Evaluation
---------------------------------

TinyLlama is a **highly optimized small-scale model**, designed for efficiency while maintaining competitive accuracy.

### Confusion Matrix - TinyLlama

.. image:: _static/confusion_matrix_LLaMA_single_select.png
   :align: center
   :width: 80%
   :alt: Confusion Matrix TinyLlama Single Select

**Analysis:**
- TinyLlama achieves similar **overall classification accuracy** but struggles more in certain **fine-grained categories**.
- There are **noticeable misclassifications** in complex categories such as **Network Operators & Infrastructure**.
- The **model still achieves high accuracy in general categories**.

### Key Metrics - TinyLlama

+-------------------+-------+
| Metric           | Value |
+===================+=======+
| **Accuracy**     | 0.97  |
+-------------------+-------+
| **F1 Score**     | 0.97  |
+-------------------+-------+
| **Macro Avg**    | 0.97  |
+-------------------+-------+
| **Weighted Avg** | 0.97  |
+-------------------+-------+

---------------------------------
Conclusion
---------------------------------

### Key Takeaways:
| Model      | Accuracy | Strengths | Weaknesses |
|------------|----------|-----------------|------------------|
| **RoBERTa** | **97%** | Best overall accuracy, high recall | Slightly larger model |
| **DistilBERT** | **97%** | Lightweight and efficient | Slightly lower recall |
| **TinyLlama** | **97%** | Small and fast | More misclassifications in rare categories |

### Recommendations:
- **For best accuracy:** Use **RoBERTa**.
- **For speed and efficiency:** Use **DistilBERT**.
- **For lightweight applications:** Use **TinyLlama**.

Each model demonstrates **high performance**, but choosing the best one depends on your specific **trade-offs between speed and accuracy**.

---------------------------------
Code for Evaluation
---------------------------------

The following **Python code** was used to evaluate all models:

**Confusion Matrix Plotting Function**

.. code-block:: python

    def plot_confusion_matrix(true_label, pred_label, model_name, labels):
        cm = confusion_matrix(true_label, pred_label)
        classes = list(labels.values())

        plt.figure(figsize=(15, 11))
        ax = sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                        xticklabels=classes, yticklabels=classes, linewidths=0.5, linecolor="gray")

        for i in range(cm.shape[0]):  
            for j in range(cm.shape[1]):  
                value = cm[i, j]
                color = "green" if i == j else ("red" if value > 0 else "black")
                ax.text(j + 0.5, i + 0.5, str(value), ha="center", va="center", color=color)

        model_display_name = {"deepset/roberta-base-squad2": "RoBERTa",
                              "distilbert-base-uncased": "DistilBERT",
                              "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "LLaMA"}.get(model_name, model_name)

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix {model_display_name} SINGLE SELECT")
        plt.savefig(f"drive/MyDrive/CapStone_models/confusion_matrix_{model_display_name}_single_select.png", bbox_inches="tight", dpi=300)
        plt.show()

