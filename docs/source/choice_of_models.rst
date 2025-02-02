Model Selection for Q&A-Model Evaluation
===========================================

This section gives an overview of the models used in our Q&A-model evaluation. Our key goal was to test out different strategies to get insights to the performance of the models as well understanding their differences.

---

RoBERTa:
----------------------------------------
We selected **RoBERTa** due to its ability to handle complex question-answer relationships. One of the main advantages of RoBERTa is the **Optimization for Q&A tasks** due to its training on **SQuAD2**. We used this model to predict possible answer labels for both single select and multi select questions.


---

DistilBERT:
------------------------------
We included **DistilBERT** for a more efficient alternative. As we wanted to evaluate the performance of another model, DistilBERT became quite interesting because although it's **60% smaller and twice as fast** as BERT, it still retains **97% accuracy**.
Another advantege with this model is the **faster training and inference** compared to larger transformer models.

---

TinyLLaMA:
-------------------------------------
We also experimented with **TinyLLaMA-1.1B-Chat-v1.0**, a lightweight LLaMA model because it allows quite **low-resource fine-tuning** with **LoRA (Low-Rank Adaptation)**. Especially compared to the bigger LLaMA-models with around 7.7 billion parameters. 
Despite its far smaller size it **retains strong contextual understanding**. Especially when working with Google Colab it's important to use a **more memory-efficient** model. Otherwise you will often experience the limitations of freemium software like Google Colab, when the GPU-RAM is used up.
The small size of this LLaMA-Model made it very appealing to try out.


The next section will cover **implementation and training strategies** for these models.

