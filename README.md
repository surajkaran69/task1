1. Introduction
The first task during my internship at NullClass focused on building an age detection model using machine learning techniques. This task was essential for understanding the practical application of AI in age estimation, a fundamental task in computer vision and artificial intelligence. The goal was to create a robust model capable of detecting the age of individuals based on facial images using pre-existing datasets.

2. Background
Age detection is an important application of computer vision, particularly in sectors such as security, retail, healthcare, and marketing. The project utilized the UTKFace dataset, a widely used dataset for age detection that includes images of people categorized by their age and gender. For this task, I used a combination of pre-trained models and transfer learning to fine-tune a ResNet architecture to detect the age of individuals in images.

3. Learning Objectives
The main learning objectives for this task were:

Understanding age detection algorithms: Learning how machine learning models can be applied to predict the age of individuals from facial images.
Hands-on experience with transfer learning: Fine-tuning pre-trained models such as ResNet for age detection.
Developing a machine learning pipeline: From dataset preprocessing to model evaluation and testing.
Model evaluation: Learning how to evaluate the model’s performance using metrics like Mean Absolute Error (MAE).
4. Activities and Tasks
The activities I performed during this task included:

Data Preprocessing: Extracted and preprocessed the UTKFace dataset to ensure it was in the correct format for training.
Model Training: Fine-tuned a pre-trained ResNet model using the FastAI framework for age detection.
Model Evaluation: After training, I evaluated the model’s performance on a test set, calculating accuracy and loss metrics.
Saving the Model: The trained model was saved as a .pkl file for future use.
Testing: I tested the model on random images to ensure its generalization ability.
5. Skills and Competencies
During this task, I developed and improved the following skills:

Machine Learning Frameworks: Gained hands-on experience using FastAI and PyTorch for model training and fine-tuning.
Data Preprocessing: Learned techniques for cleaning and preprocessing image data to make it suitable for training machine learning models.
Model Evaluation: Developed the ability to evaluate model performance using metrics like Mean Absolute Error (MAE).
Problem-Solving: Learned how to handle challenges related to dataset quality and model performance.
6. Feedback and Evidence
The task allowed me to apply theoretical knowledge into practice, and I received valuable feedback from my supervisor at NullClass. The feedback highlighted areas where I excelled, particularly in utilizing transfer learning effectively. I was able to deliver a well-trained model, and the results met the expectations for age detection accuracy on the test set.

Evidence of my work includes:

Code repository for the model.
Screenshots of the model training and testing results.
A .pkl file of the trained model.
7. Challenges and Solutions
Some challenges I faced during the task included:

Dataset Imbalance: The dataset had uneven age group distributions, which could have led to poor model performance on underrepresented age groups.
Solution: I implemented techniques like oversampling and data augmentation to mitigate this issue.

Model Overfitting: Initially, the model showed signs of overfitting.
Solution: I applied regularization techniques and adjusted the learning rate to prevent overfitting.

8. Outcomes and Impact
By completing this task, I successfully built an age detection model that performs well on unseen data. This experience has had a significant impact on my understanding of machine learning and model development. The skills learned in this task are directly applicable to other computer vision projects and have enhanced my proficiency with deep learning frameworks.

9. Conclusion
The completion of Task 1 during my internship at NullClass provided me with valuable practical experience in the field of machine learning and computer vision. The task allowed me to gain hands-on experience with model training, testing, and evaluation. I also learned valuable skills in handling real-world datasets, improving model accuracy, and solving problems such as overfitting and data imbalance.
