Data Preprocessing for Single Select Questions
===============================================

This section describes the preprocessing steps required to **prepare the single select dataset** for fine-tuning the **Q&A model**. We detail the conversion of raw JSON data into structured datasets, encoding categorical labels, splitting data into training and test sets, and transforming the data into a format suitable for **Hugging Face Transformers**.

---

Converting JSON Data into a Structured DataFrame
------------------------------------------------
Since the raw data genrated before is stored in a JSON format, the first step is to extract **relevant fields** and convert them into a structured **Pandas DataFrame**.
We decided to **download an store relevant data** such as the training data locally, since we can access the needed data or documents easier in case the **Google Colab notebook crashes** or the runtime disconnects.

.. code-block:: python

   import json
   import pandas as pd

   def convert_json_to_df(json_file):
       rows = []

       for entry in json_file:
           question = entry["question"]
           question_type = entry["type"]
           for answer in entry["answers"]:
               answer_label = answer["answer_label"]
               if question_type == "MULTI_SELECT":
                   answer_label = answer["answer_label"].split(", ")

               rows.append({
                   "question": question,
                   "type": question_type,
                   "answer_text": answer["answer_text"],
                   "answer_label": answer_label,
                   "timestamp": answer["timestamp"]
               })

       return pd.DataFrame(rows)

   # Load JSON data
   with open('final_single_question_data.json', 'r') as f:
       single_select_model_data = json.load(f)

   df_single_select_final = convert_json_to_df(single_select_model_data)


---

Encoding Answer Labels into Numeric Classes
-------------------------------------------
Since **machine learning models require numerical inputs**, categorical labels must be converted into integer values. We achieve this using **Label Encoding**.

.. code-block:: python

   from sklearn.preprocessing import LabelEncoder
   import pickle

   label_encoder_single_select = LabelEncoder()
   df_single_select_final["label"] = label_encoder_single_select.fit_transform(df_single_select_final["answer_label"])

   # Save label mapping for later use
   label_mapping_single_select = {f"LABEL_{index}": label for index, label in enumerate(label_encoder_single_select.classes_)}

   label_mapping_single_select_path = 'label_mapping_single_select.pkl'
   with open(label_mapping_single_select_path, 'wb') as f:
       pickle.dump(label_mapping_single_select, f)




---

Splitting Data into Training and Test Sets
------------------------------------------
To evaluate the model's performance, the dataset is split into **training (80%)** and **test (20%)** sets using **stratified sampling**.

.. code-block:: python

   from sklearn.model_selection import train_test_split

   train_df_single_select, test_df_single_select = train_test_split(
       df_single_select_final,
       test_size=0.2,
       random_state=0,
       stratify=df_single_select_final["label"]
   )

   print(f"Training samples: {len(train_df_single_select)}")
   print(f"Evaluation samples: {len(test_df_single_select)}")


This **Ensures balanced class distribution** in both training and test sets using `stratify=df["label"]`.
To ensure **Reproducibility** we are using `random_state=0`.

---

Sampling Data for Inspection
----------------------------
To verify the correctness of preprocessing, we print **random samples** from the training and test sets.

.. code-block:: python

   num_samples = 5

   # Random samples from the training set
   random_samples_train = train_df_single_select.sample(n=num_samples, random_state=42)
   print("Formatted Training Data:\n")
   for index, example in random_samples_train.iterrows():
       print(f"Example {index + 1}:")
       print(f"  Question: {example['question']}")
       print(f"  Context: {example['answer_text']}")
       print(f"  Label: {example['answer_label']}")
       print("-" * 20)

   # Random samples from the test set
   random_samples_test = test_df_single_select.sample(n=num_samples, random_state=42)
   print("\n\nFormatted Evaluation Data:\n")
   for index, example in random_samples_test.iterrows():
       print(f"Example {index + 1}:")
       print(f"  Question: {example['question']}")
       print(f"  Context: {example['answer_text']}")
       print(f"  Label: {example['answer_label']}")
       print("-" * 20)

This helps to **ensure correct mapping of labels, questions, and answers** before model training. It also **helps verify the dataset integrity**, reducing preprocessing errors.

---

Creating a Hugging Face Dataset for Training
--------------------------------------------
To train a **Transformer-based model**, the dataset must be converted into a **Hugging Face DatasetDict**.

.. code-block:: python

   from datasets import Dataset, DatasetDict

   dataset_single_select = DatasetDict({
       "train": Dataset.from_pandas(train_df_single_select),
       "test": Dataset.from_pandas(test_df_single_select)
   })

   print(dataset_single_select)

This makes sure that the data is **compatibile with Hugging Face Transformers** for seamless model training.
