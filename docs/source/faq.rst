6. FAQ
===

**Frage:** Wie installiere ich das Projekt?  
**Antwort:** Siehe `installation.rst`.

**Frage:** Wo finde ich den Quellcode?  
**Antwort:** Im GitHub-Repository.

Q: How do I retrain the model?

A: Run the training script with your updated dataset:

python train.py --dataset final_single_question_data.json

Q: Can I use another model?

A: Yes! Modify the model checkpoint in train.py:

model_name = "distilbert-base-uncased"


