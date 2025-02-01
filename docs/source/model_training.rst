Model Training
==============

Fine-tuning a Transformer Model
--------------------------------
The project uses **Hugging Face Transformers** to fine-tune a classification model.

.. code-block:: python

   from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

   tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
   model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

Training the Model
-------------------
.. code-block:: python

   training_args = TrainingArguments(
       output_dir="./results",
       evaluation_strategy="epoch",
       per_device_train_batch_size=16,
       per_device_eval_batch_size=16,
       num_train_epochs=3,
       weight_decay=0.01,
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=eval_dataset,
   )

   trainer.train()
