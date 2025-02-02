Evaluation of the Models
========================

This section presents the evaluation of the fine-tuned models for both **Single-Select** and **Multi-Select** questions. We analyze performance metrics such as **accuracy, F1-score, precision, recall**, and visualize **loss curves** and **confusion matrices**.

---

Single-Select Model Evaluations
-------------------------------

RoBERTa Model
~~~~~~~~~~~~~
.. code-block:: python

   # Load metrics and compute confusion matrix
   cm = confusion_matrix(labels, preds_roberta)
   classes = list(label_mapping_single_select.values())

   plt.figure(figsize=(15, 11))
   ax = sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, linewidths=0.5, linecolor="gray")

   for i in range(cm.shape[0]):
       for j in range(cm.shape[1]):
           value = cm[i, j]
           color = "green" if i == j else "red" if value > 0 else "black"
           ax.text(j + 0.5, i + 0.5, str(value), ha="center", va="center", color=color)

   plt.xlabel("Predicted Label")
   plt.ylabel("True Label")
   plt.title("Confusion Matrix - RoBERTa Single-Select")
   plt.savefig("roberta_single_confusion_matrix.png", bbox_inches="tight", dpi=300)
   plt.show()

.. image:: /_static/A9724zzibDeRAAAAAElFTkSuQmCC.png
   :width: 600px
   :alt: RoBERTa Single-Select Confusion Matrix

---

DistilBERT Model
~~~~~~~~~~~~~~~~
.. code-block:: python

   cm = confusion_matrix(labels, preds_distilbert)
   plt.figure(figsize=(15, 11))
   sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
               xticklabels=classes, yticklabels=classes)
   plt.title("Confusion Matrix - DistilBERT Single-Select")
   plt.xlabel("Predicted Label")
   plt.ylabel("True Label")
   plt.savefig("distilbert_single_confusion_matrix.png", bbox_inches="tight", dpi=300)
   plt.show()

.. image:: /_static/GMFAAAAAC4iB0pAAAAAHARjRQAAAAAuIhGCgAAAABcRCMFAAAAAC6ikQIAAAAAF9FIAQAAAICLaKQAAAAAwEU0UgAAAADgon8BztsQ7vDmyJ4AAAAASUVORK5CYII.png
   :width: 600px
   :alt: DistilBERT Single-Select Confusion Matrix

---

TinyLLaMA Model
~~~~~~~~~~~~~~~
.. code-block:: python

   cm = confusion_matrix(labels, preds_llama)
   plt.figure(figsize=(15, 11))
   sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
               xticklabels=classes, yticklabels=classes)
   plt.title("Confusion Matrix - TinyLLaMA Single-Select")
   plt.xlabel("Predicted Label")
   plt.ylabel("True Label")
   plt.savefig("llama_single_confusion_matrix.png", bbox_inches="tight", dpi=300)
   plt.show()

---

Multi-Select Model Evaluations
------------------------------

RoBERTa Multi-Label Model
~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   cm = multilabel_confusion_matrix(labels, preds_roberta_multi)
   plt.figure(figsize=(15, 11))
   sns.heatmap(cm[0], annot=True, fmt="d", cmap="Blues")
   plt.title("Confusion Matrix - RoBERTa Multi-Select")
   plt.xlabel("Predicted Label")
   plt.ylabel("True Label")
   plt.savefig("roberta_multi_confusion_matrix.png", bbox_inches="tight", dpi=300)
   plt.show()

.. image:: /_static/INuMqt7np3QAAAAASUVORK5CYII.png
   :width: 600px
   :alt: RoBERTa Multi-Select Confusion Matrix

---

DistilBERT Multi-Label Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   cm = multilabel_confusion_matrix(labels, preds_distilbert_multi)
   plt.figure(figsize=(15, 11))
   sns.heatmap(cm[0], annot=True, fmt="d", cmap="Blues")
   plt.title("Confusion Matrix - DistilBERT Multi-Select")
   plt.xlabel("Predicted Label")
   plt.ylabel("True Label")
   plt.savefig("distilbert_multi_confusion_matrix.png", bbox_inches="tight", dpi=300)
   plt.show()

---

Loss and Performance Metrics
----------------------------
We compare **training loss and validation loss** for all models over epochs.

.. code-block:: python

   def plot_losses(csv_path, model_name):
       metrics_df = pd.read_csv(csv_path)
       plt.figure(figsize=(10, 6))
       plt.plot(metrics_df['epoch'], metrics_df['Training Loss'], label='Training Loss', color='blue', marker='o')
       plt.plot(metrics_df['epoch'], metrics_df['Validation Loss'], label='Validation Loss', color='orange', marker='o')
       plt.xlabel("Epoch")
       plt.ylabel("Loss")
       plt.title(f"Training and Validation Loss - {model_name}")
       plt.legend()
       plt.grid(True, linestyle='--', alpha=0.7)
       plt.savefig(f"{model_name}_loss_plot.png", bbox_inches="tight")
       plt.show()

.. image:: /_static/roberta_multi_loss_plot.png
   :width: 600px
   :alt: Training and Validation Loss for RoBERTa Multi-Select

.. image:: /_static/distilbert_multi_loss_plot.png
   :width: 600px
   :alt: Training and Validation Loss for DistilBERT Multi-Select

---

Performance Metrics
~~~~~~~~~~~~~~~~~~~
We also visualize accuracy, precision, recall, and F1-score for each model over training epochs.

.. code-block:: python

   def plot_metrics(csv_path, model_name):
       metrics_df = pd.read_csv(csv_path)
       plt.figure(figsize=(10, 6))
       plt.plot(metrics_df["epoch"], metrics_df["Accuracy"], label="Accuracy", marker="o")
       plt.plot(metrics_df["epoch"], metrics_df["F1"], label="F1", marker="o")
       plt.plot(metrics_df["epoch"], metrics_df["Precision"], label="Precision", marker="o")
       plt.plot(metrics_df["epoch"], metrics_df["Recall"], label="Recall", marker="o")
       plt.xlabel("Epoch")
       plt.ylabel("Metrics")
       plt.title(f"Metrics Over Epochs for {model_name}")
       plt.legend()
       plt.grid(True, linestyle="--", alpha=0.7)
       plt.savefig(f"metrics_plot_{model_name}.png", bbox_inches="tight")
       plt.show()

.. image:: /_static/metrics_roberta_multi.png
   :width: 600px
   :alt: Performance Metrics for RoBERTa Multi-Select

.. image:: /_static/metrics_distilbert_multi.png
   :width: 600px
   :alt: Performance Metrics for DistilBERT Multi-Select

---

Conclusion
----------
- **RoBERTa outperforms other models** in both **Single-Select** and **Multi-Select** tasks.
- **DistilBERT** provides a **lightweight alternative** with **slightly lower performance** but **faster training**.
- **TinyLLaMA** shows **competitive performance** but struggles with certain fine-grained distinctions.
- **Loss curves indicate effective learning**, with **validation loss stabilizing** across models.
- **Confusion matrices highlight misclassification patterns**, guiding further improvements.

Future work includes **fine-tuning hyperparameters**, **adding more labeled data**, and **testing alternative architectures**.

---

This concludes the **evaluation section** for all models used in this project.


