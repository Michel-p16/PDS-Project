Model Implementation for Multi-Select Questions
=================================================

This section describes the implementation of transformer-based models for **Multi-Select Questions**

---

RoBERTa Implementation
-----------------------
We used **RoBERTa** for multi-label classification.

**1. Load the tokenizer and model:**

.. code-block:: python

   from transformers import AutoTokenizer, AutoModelForSequenceClassification

   model_name = "deepset/roberta-base-squad2"
   tokenizer_roberta = AutoTokenizer.from_pretrained(model_name)
   model_roberta = AutoModelForSequenceClassification.from_pretrained(
       model_name, num_labels=len(multi_label_binarizer.classes_),
       problem_type="multi_label_classification"
   )

**2. Tokenize input data:**

.. code-block:: python

   def preprocess_function_multi(examples):
       return tokenizer_roberta(
           examples["question"],  # Question
           examples["text"],      # Answer
           padding="max_length",
           truncation=True,
           max_length=128
       )

   tokenized_dataset_roberta = dataset_multi_select.map(preprocess_function_multi, batched=True)

**3. Define training arguments and trainer:**

.. code-block:: python

   from transformers import TrainingArguments, Trainer

   training_args_roberta = TrainingArguments(
       output_dir="./roberta_classification_multi_select",
       per_device_train_batch_size=4,
       num_train_epochs=30,
       learning_rate=6e-5,
       weight_decay=0.01,
       evaluation_strategy="epoch",
       save_strategy="epoch",
       logging_dir="./logs",
       load_best_model_at_end=True,
       metric_for_best_model="f1",
       greater_is_better=True
   )

   trainer_roberta = Trainer(
       model=model_roberta,
       args=training_args_roberta,
       train_dataset=tokenized_dataset_roberta["train"],
       eval_dataset=tokenized_dataset_roberta["test"],
       compute_metrics=compute_metrics_multi_select
   )

**4. Train the model:**

.. code-block:: python

   trainer_roberta.train()

**5. Save the trained model:**

.. code-block:: python

   model_roberta.save_pretrained("roberta_classification_multi_select")
   tokenizer_roberta.save_pretrained("roberta_classification_multi_select")

---

DistilBERT Implementation
--------------------------
We implemented **DistilBERT (distilbert-base-uncased)** for a computationally efficient alternative.

**1. Load the tokenizer and model:**

.. code-block:: python

   model_name_distilbert = "distilbert-base-uncased"
   tokenizer_distilbert = AutoTokenizer.from_pretrained(model_name_distilbert)
   model_distilbert = AutoModelForSequenceClassification.from_pretrained(
       model_name_distilbert, num_labels=len(multi_label_binarizer.classes_),
       problem_type="multi_label_classification"
   )

**2. Tokenize input data:**

.. code-block:: python

   tokenized_dataset_distilbert = dataset_multi_select.map(preprocess_function_multi, batched=True)

**3. Define training arguments and trainer:**

.. code-block:: python

   training_args_distilbert = TrainingArguments(
       output_dir="./distilbert_classification_multi_select",
       per_device_train_batch_size=4,
       num_train_epochs=20,
       learning_rate=5e-5,
       weight_decay=0.01,
       evaluation_strategy="epoch",
       save_strategy="epoch",
       logging_dir="./logs",
       load_best_model_at_end=True,
       metric_for_best_model="f1",
       greater_is_better=True
   )

   trainer_distilbert = Trainer(
       model=model_distilbert,
       args=training_args_distilbert,
       train_dataset=tokenized_dataset_distilbert["train"],
       eval_dataset=tokenized_dataset_distilbert["test"],
       compute_metrics=compute_metrics_multi_select
   )

**4. Train the model:**

.. code-block:: python

   trainer_distilbert.train()

**5. Save the trained model:**

.. code-block:: python

   model_distilbert.save_pretrained("distilbert_classification_multi_select")
   tokenizer_distilbert.save_pretrained("distilbert_classification_multi_select")

