Challenges and Future Work
==========================

In this section we want to outline our key challenges encountered during the development of the project and opportunities for future improvements.

---

Challenges
----------

Several challenges arose during the implementation and fine-tuning of the models:

1. **Finding Optimal Hyperparameters**  

   - Finding an optimal **learning rate** was not as easy as it seemed and cost us a lot of time. The same applies to a good **threshold** for multilabel classification probabilities.  

2. **Handeling Google Colabs Computation Limits**  

   - We often faced the challenge of GPU-RAM limits in Google Colabs free version. This got expotentionally worse the more complex the model and the training got. We had to try a lot of different models until we were able to efficently train the models in Google Colab. Several diffrent Google accounts were necessary to finaly finetune our models and make good predictions. 

3. **Limited Training Data**  

   - The dataset, while structured, lacked **diversity** in some categories. Even with all approaches to use prompt engeneering, it was still very challenging to create a dataset which could be used for meaningful training and results.

4. **Underestimating Training Duration**  

   - It took us a huge amount of time to train and finetune our models. We definetly did not expect the time needed to train to be as long as it actually was. Errors and debugging made it very time consuming to build our models. This brought us in a situation where we were not able anymore to fulfill all our goals cosidering the extra tasks.

---

Future Work
-----------

Unfortunatly we were not able to implement everything that could be done and focused on the implementation and evaluation of important models. Several improvements could enhance the project:

1. **Dataset Expansion**  

   - Collecting more **real-world** question-answer pairs.  
   - Using other and maybe even better **data augmentation techniques** to introduce more variability and allow a broad-based training. That would make our project usable for more real world applications.   

2. **Model Optimization**  

   - Using **quantization** and **pruning** techniques to reduce model size and improve inference speed could make the model more efficient  

3. **Alternative Model Architectures**  

   - Testing newer architectures like **Mistral** or **GPT-4** for improved contextual understanding. With more computing power or more ressources in general the use of other architectures would have been possible. This could have highly increasd the possibilities.
   - Investigating **multi-modal approaches**, integrating text with **speech or image-based inputs** could be interesting too. 

4. **Improved Multi-Label Classification**  

   - Implementing **hierarchical classification** for better handling of related labels.  
   - **Summerization of multiple connected question - answer pairs** to create one large input. On this basis one could try to predict the related labels to all the different answers given in the input. This allows more real world applications.

5. **Deployment and Real-World Integration**  

   - Developing an **API** for real-time classification in applications.    
