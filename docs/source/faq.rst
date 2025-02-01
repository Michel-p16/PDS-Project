Frequently Asked Questions (FAQ)
================================

**Q: How do I retrain the model?**  
A: Run the training script with your updated dataset:

.. code-block:: bash

   python train.py --dataset final_single_question_data.json

**Q: Can I use another model?**  
A: Yes! Modify the model checkpoint in `train.py`:

.. code-block:: python

   model_name = "distilbert-base-uncased"
