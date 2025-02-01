Data Extraction and Dataset Creation with Gemini
===============================================

This section describes the steps taken to extract and process questionnaire data from JSON files and enhance it using **Google Gemini AI**.
We decided to use **Gemini AI** because its quite easy accesible in the **free version** but still provides good generations to create useful training data.
---

Merging JSON Files
-------------------
The first step is to merge multiple JSON files from the `Raw_data/` directory into a single dataset. This ensures that all questionnaire data is collected in a structured format.

.. code-block:: python

   import os
   import json

   json_dir = "Raw_data"
   merged_data = []

   for filename in os.listdir(json_dir):
       if filename.endswith(".json"):
           file_path = os.path.join(json_dir, filename)

           with open(file_path, "r", encoding="utf-8") as file:
               json_data = json.load(file)
               merged_data.extend(json_data)

   with open("merged_questionnaires.json", "w", encoding="utf-8") as output_file:
       json.dump(merged_data, output_file, indent=4)

   print("Merged JSON data with all information saved successfully!")

This script iterates over all JSON files in the `Raw_data/` directory, loads their contents, and merges them into `merged_questionnaires.json`.

---

Creating a Structured Dataset
-----------------------------
Once the JSON files are merged, relevant fields are extracted and stored in a **Pandas DataFrame**.
This makes the data easier to handle and process.

.. code-block:: python

   import pandas as pd

   with open("merged_questionnaires.json", "r", encoding="utf-8") as file:
       json_data = json.load(file)

   data = []
   for entry in json_data:
       question_type = entry["type"]
       question = entry["question"]
       for option in entry["options"]:
           data.append([question_type, question, option["option"]])

   df_questionnaires = pd.DataFrame(data, columns=["Type", "Question", "Label"])
   print(df_questionnaires.head())

This results in a DataFrame containing three columns:
- **Type**: Specifies whether the question is `SINGLE_SELECT` or `MULTI_SELECT`.
- **Question**: The question being asked.
- **Label**: The true answer to the question.

To facilitate further processing, the dataset is split into two subsets:

.. code-block:: python

   df_single_select_questions = df_questionnaires[df_questionnaires["Type"] == "SINGLE_SELECT"]
   df_multi_select_questions = df_questionnaires[df_questionnaires["Type"] == "MULTI_SELECT"]

---

This allows us to handle the generation of answers to the quesions diffrent, according to their respective question type.

Enhancing Questions with Gemini AI
-----------------------------------
Some questions in the questionaires are not expressed as a typical question. For example the first "question" ist: "Data processing consent". Since we want to ask Gemini to answer the question from the viewpoint of a user, this may confuse the Gemini model and produce bad outputs.
To improve clarity, we use **Google Gemini AI** to generate more refined versions of the questions.

**Step 1: Configuring the API**

.. code-block:: python

   import google.generativeai as genai
   import time

   genai.configure(api_key="")  # Enter your API key here

**Step 2: Defining the API Call Function**

.. code-block:: python

   def api_call_for_generating_question(question):
       try:
           model = genai.GenerativeModel("gemini-1.5-flash")
           prompt = f"Generate a full understandable and short question based on the following: {question}. Direct the message to me. Print the question only!"
           response = model.generate_content(prompt)
           return response.text.strip()
       except Exception as e:
           print(f"Error with Gemini API: {e}")
           return question  # Fallback to the original question

**Step 3: Applying the Function to the Dataset**

Since the free version of the Gemini API only handles limited requests per minute, we delay the requests accordingly.

.. code-block:: python

   def generate_question(df):
       generated_questions = {}

       for question in df["Question"]:
           if question not in generated_questions:
               full_question = api_call_for_generating_question(question)
               generated_questions[question] = full_question
               time.sleep(3)  # Prevent API rate limiting

       df["Question"] = df["Question"].map(generated_questions)
       print("Questions in dataframe with new questions replaced.")
       return df

   df_single_select_questions = generate_question(df_single_select_questions)

This process replaces vague or incomplete questions with **more informative and precise versions**.

---

Generating Diverse Answer Options with Gemini
---------------------------------------------
To improve response diversity, **Gemini AI** generates a wide range of possible answers.

**Step 1: Function for Answer Generation**

We used a lot of prompt engeneering strategies here, to improve the generated outputs. The Gemini AI is told to create 100 different answers to each question and each label. That ensures a big dataset with a lot of variation to allow a good training in the following.

.. code-block:: python

   import datetime
   import re

   def make_api_call_for_answers(question, label, type):
       try:
           model = genai.GenerativeModel("gemini-1.5-flash")
           prompt = f"Generate 100 full diverse answers as one sentence split in rows for the following context '{question}' with the answer label: '{label}'. Print the answers ONLY."

           if type == "MULTI_SELECT":
               prompt = f"Generate 100 full diverse answers for '{question}' with multiple labels: '{label}'. Include all possible combinations. Print only the answers."

           response = model.generate_content(prompt)
           print(f"Answers for Question \"{question}\" with label \"{label}\" generated.")
           return response.text.strip()
       except Exception as e:
           print(f"Error with Gemini API: {e}")
           return question  # Fallback to the original question

**Step 2: Applying the Function to the Dataset**

.. code-block:: python

   def generate_diverse_answers(df):
       generated_answers = []
       processed_questions = set()

       for _, row in df.iterrows():
           type = row["Type"]
           question = row["Question"]

           if type == "SINGLE_SELECT":
               label = row["Label"]
               answers = make_api_call_for_answers(question, label, type)
               for answer in answers.split("\n"):
                   generated_answers.append({
                       "question": question,
                       "type": type,
                       "answer_text": answer,
                       "answer_label": label,
                       "timestamp": datetime.datetime.now().isoformat()
                   })
               time.sleep(3)

           else:
               labels = df[df["Question"] == question]["Label"].tolist()
               if question not in processed_questions:
                   processed_questions.add(question)
                   answers = make_api_call_for_answers(question, labels, type)

                   pattern = r"^(.*?)\s+\[([^\]]+)\]$"
                   for answer in answers.split("\n"):
                       match = re.match(pattern, answer)
                       if match:
                           generated_answers.append({
                               "question": question,
                               "type": type,
                               "answer_text": match.group(1),
                               "answer_label": match.group(2),
                               "timestamp": datetime.datetime.now().isoformat()
                           })
                   time.sleep(3)

       return generated_answers

   df_single_select_with_new_q_and_a = generate_diverse_answers(df_single_select_questions)

This approach ensures that:
- **Single-choice answers** are well-structured.
- **Multi-choice responses** contain valid combinations.
- **Answers are diverse**, improving dataset richness.

---

Final Thoughts
--------------
This section detailed how:
- **Raw JSON data is processed and structured**.
- **Gemini AI refines questions** to ensure clarity.
- **Diverse answer sets are generated** to enhance data quality.

These steps form the foundation for **training and evaluating QA-models** on high-quality labeled data.

