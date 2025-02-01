Defining Performance Metrics
----------------------------
To evaluate model performance, we define **accuracy, precision, recall, and F1-score** metrics.

.. code-block:: python

   from sklearn.metrics import accuracy_score, precision_recall_fscore_support

   def compute_metrics_single_select(pred):
       labels = pred.label_ids
       preds = pred.predictions.argmax(-1)
       precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
       acc = accuracy_score(labels, preds)
       return {
           'accuracy': acc,
           'f1': f1,
           'precision': precision,
           'recall': recall
       }

**Why this approach?**  
- **Accuracy measures overall correctness**.
- **F1-score balances precision and recall**, critical for imbalanced datasets.
- **Ensures robust model evaluation** before deployment.

