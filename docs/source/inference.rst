Model Inference
===============

Classifying a Question
-----------------------
To use the trained model for classification:

.. code-block:: python

   question = "What type of customer are you?"
   answer = "I'm a first-time buyer exploring your offerings."

   tokens = tokenizer(question + " " + answer, return_tensors="pt")
   predictions = model(**tokens)
   predicted_label = predictions.logits.argmax().item()
   print(f"Predicted Label: {predicted_label}")
