Data Preprocessing for Multi-Select Questions
=============================================

This section describes the preprocessing steps required to **prepare the dataset** for fine-tuning the **Multi-Select QA-Models**.

---

Loading Data
------------
We begin by loading the dataset from a JSON file. Make sure to upload the data to your directory and insert your own path.

.. code-block:: python

   import json

   with open('final_multi_question_data.json', 'r') as file:
       multi_select_model_data = json.load(file)

---

Encoding Labels with MultiLabelBinarizer
----------------------------------------
Since multi-select classification requires **binary vectors** for labels, we use `MultiLabelBinarizer`.

**1. Extract all unique labels:**

.. code-block:: python

   from sklearn.preprocessing import MultiLabelBinarizer

   all_labels_multi_select = set()
   for example in multi_select_model_data:
       for answer in example["answers"]:
           labels = answer.get("answer_label", "").split(",")
           labels = [label.strip() for label in labels]
           all_labels_multi_select.update(labels)

   multi_label_binarizer = MultiLabelBinarizer(classes=sorted(list(all_labels_multi_select)))
   multi_label_binarizer.fit([list(all_labels_multi_select)])

**2. Convert dataset into the required format:**

.. code-block:: python

   def convert_to_multi_select_format(data, mlb):
       formatted_data = []
       for example in data:
           question = example["question"]
           answers = example["answers"]
           for answer in answers:
               text = answer.get("answer_text", "")
               labels = answer.get("answer_label", "").split(",")
               labels = [label.strip() for label in labels]
               if labels:
                   label_vector = mlb.transform([labels])[0].astype(float)
                   formatted_data.append({
                       "question": question,
                       "text": text,
                       "labels": label_vector
                   })
       return formatted_data

   formatted_multi_dataset = convert_to_multi_select_format(multi_select_model_data, multi_label_binarizer)

---

Verifying Data Formatting
-------------------------
To confirm proper formatting, we print sample data.

.. code-block:: python

   print(f"Total formatted examples: {len(formatted_multi_dataset)}")
   if formatted_multi_dataset:
       print(f"Sample: {formatted_multi_dataset[0]}")
   print(f"All possible labels: {multi_label_binarizer.classes_}")

---

Splitting Data into Train and Test Sets
---------------------------------------
The dataset is split into **80% training** and **20% test** data.

.. code-block:: python

   from sklearn.model_selection import train_test_split

   train_data_multi_formatted, eval_data_multi_formatted = train_test_split(
       formatted_multi_dataset, test_size=0.2, random_state=42)

---

Creating a Hugging Face Dataset
--------------------------------
Again we convert the processed data into a `DatasetDict` for Hugging Face models.

.. code-block:: python

   from datasets import Dataset, DatasetDict

   dataset_multi_select = DatasetDict({
       "train": Dataset.from_list(train_data_multi_formatted),
       "test": Dataset.from_list(eval_data_multi_formatted)
   })

   print(f"Training samples: {len(dataset_multi_select['train'])}")
   print(f"Evaluation samples: {len(dataset_multi_select['test'])}")

---

Now the data is prepared to implement models for recognizing multiple possible labels based on an question + answer input.
