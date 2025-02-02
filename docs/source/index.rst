PDS Final Project
=================

Welcome to the documentation of the **PDS Final Project**.

This project focuses on:
- Creating and processing **questionnaires**, containing questions and answers.
- Classifying questions and answers using **fine-tuned Q&A models**, to asign the answers to a predifined label.
- Using **LLMs (Large Language Models)** for improved classification.

This documentation provides a full guide on the generation and preprocessing of training data as well as the implementation and evaluation of Q&A-Models for both single select and multi select questions. The documentation also gives insights to our decisions regarding model selection and evaluation strategies. We also look into challenges we had to face during our project and furhter improvement.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   data_generation
   choice_of_models
   data_preprocessing_single_select
   single_select_implementation
   evaluation_single
   data_preprocessing_multi_select
   multi_select_implementation
   evaluation_multi
   challenges
   development
