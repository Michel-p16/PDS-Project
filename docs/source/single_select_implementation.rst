Model Implementation for Single Select Questions
=================================================

This section describes the implementation of models for **Single Select Question Classification**.

---

RoBERTa Implementation
-----------------------
We used **RoBERTa** to classify the single select question + answer pairs to the correct label.

**1. Load the tokenizer and model:**

.. code-block:: python

   from transformers import AutoTokenizer, AutoModelForSequenceClassification

   model_name = "deepset/roberta-base-squad2"
   tokenizer_roberta = AutoTokenizer.from_pretrained(model_name)
   model_roberta = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels_single_select)

**2. Tokenize input data:**

.. code-block:: python

   def tokenize_function_roberta(examples):
       text_inputs = [q + " " + a for q, a in zip(examples["question"], examples["answer_text"])]
       return tokenizer_roberta(text_inputs, truncation=True, padding="max_length", max_length=128)

   tokenized_dataset_roberta = dataset_single_select.map(tokenize_function_roberta, batched=True)

**3. Define training arguments and trainer:**

.. code-block:: python

   from transformers import TrainingArguments, Trainer

   training_args = TrainingArguments(
       output_dir="./roberta_classification_single_select",
       per_device_train_batch_size=4,
       num_train_epochs=4,
       weight_decay=0.01,
       logging_dir="./logs",
       logging_steps=10,
       lr_scheduler_type="linear",
       warmup_steps=500,
       max_grad_norm=1.0
   )

   trainer_roberta = Trainer(
       model=model_roberta,
       args=training_args,
       train_dataset=tokenized_dataset_roberta["train"],
       eval_dataset=tokenized_dataset_roberta["test"],
       tokenizer=tokenizer_roberta,
       compute_metrics=compute_metrics_single_select
   )

**4. Train the model:**

.. code-block:: python

   trainer_roberta.train()

**5. Save the trained model:**

.. code-block:: python

   model_roberta.save_pretrained("roberta_classification_single_select")
   tokenizer_roberta.save_pretrained("roberta_classification_single_select")

---

DistilBERT Implementation
--------------------------
We implemented **DistilBERT** for an efficient alternative.

**1. Load the tokenizer and model:**

.. code-block:: python

   model_name_distilbert = "distilbert-base-uncased"
   tokenizer_distilbert = AutoTokenizer.from_pretrained(model_name_distilbert)
   model_distilbert = AutoModelForSequenceClassification.from_pretrained(model_name_distilbert, num_labels=num_labels_single_select)

**2. Tokenize input data:**

.. code-block:: python

   def tokenize_function_distilbert(examples):
       text_inputs = [q + " " + a for q, a in zip(examples["question"], examples["answer_text"])]
       return tokenizer_distilbert(text_inputs, truncation=True, padding="max_length", max_length=128)

   tokenized_dataset_distilbert = dataset_single_select.map(tokenize_function_distilbert, batched=True)

**3. Define training arguments and trainer:**

.. code-block:: python

   training_args_distilbert = TrainingArguments(
       output_dir="./distilbert_classification_single_select",
       per_device_train_batch_size=16,
       per_device_eval_batch_size=16,
       num_train_epochs=5,
       learning_rate=2e-5,
       weight_decay=0.01,
       evaluation_strategy="epoch",
       logging_dir="./logs",
       load_best_model_at_end=True
   )

   trainer_distilbert = Trainer(
       model=model_distilbert,
       args=training_args_distilbert,
       train_dataset=tokenized_dataset_distilbert["train"],
       eval_dataset=tokenized_dataset_distilbert["test"],
       compute_metrics=compute_metrics_single_select
   )

**4. Train the model:**

.. code-block:: python

   trainer_distilbert.train()

**5. Save the trained model:**

.. code-block:: python

   model_distilbert.save_pretrained("distilbert_classification_single_select")
   tokenizer_distilbert.save_pretrained("distilbert_classification_single_select")

---

TinyLLaMA Implementation
-------------------------
We tested **TinyLLaMA (TinyLLaMA-1.1B-Chat-v1.0)** for fine-tuning on resource-limited environments.

**1. Load the tokenizer and model:**

.. code-block:: python

   model_name_llama = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
   tokenizer_llama = AutoTokenizer.from_pretrained(model_name_llama)
   model_llama = AutoModelForSequenceClassification.from_pretrained(model_name_llama, num_labels=num_labels_single_select)

**2. Apply LoRA fine-tuning:**

.. code-block:: python

   from peft import LoraConfig, get_peft_model

   lora_config = LoraConfig(
       r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
       lora_dropout=0.05, bias="none", task_type="SEQ_CLS"
   )

   model_llama = get_peft_model(model_llama, lora_config)

**3. Train the model:**

.. code-block:: python

   trainer_llama = Trainer(
       model=model_llama,
       args=training_args,
       train_dataset=tokenized_dataset_roberta["train"],
       eval_dataset=tokenized_dataset_roberta["test"]
   )

   trainer_llama.train()

**4. Save the trained model:**

.. code-block:: python

   model_llama.save_pretrained("tinyllama_classification_single_select")
   tokenizer_llama.save_pretrained("tinyllama_classification_single_select")

---

Notes
----------
Since every of our group members implemented a diffrent model, you can see that the training arguments for the diffrent models may vary a little, especially when it comes to training epochs and batchsize.

