3. Data Processing

3.1 Merging JSON Files

The project loads multiple JSON files from Raw_data/ and merges them into a single dataset:

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

This merges all questionnaire data into merged_questionnaires.json.

3.2 Dataframe Creation

Extract relevant fields and store them in a Pandas DataFrame:

import pandas as pd

with open("merged_questionnaires.json", "r", encoding="utf-8") as file:
    json_data = json.load(file)

data = []
for entry in json_data:
    question_type = entry["type"]
    question = entry["question"]
    for option in entry["options"]:
        data.append([question_type, question, option["option"]])

df = pd.DataFrame(data, columns=["Type", "Question", "Label"])
print(df.head())

