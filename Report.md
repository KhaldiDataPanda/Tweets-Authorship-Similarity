# Tweets Authorship Similarity

## Introduction and Goals

This project aims to develop a machine learning model that can determine the authorship similarity between pairs of tweets. In other words, given two tweets, the model should be able to predict whether they were written by the same author or different authors. This task has several practical applications, such as identifying sockpuppet accounts on social media, detecting plagiarism or ghostwriting, and potentially aiding forensic linguistics investigations.

## Data Preparation

The project starts by loading two datasets: a training set from a CSV file and a test set from an Excel file. These datasets presumably contain tweets along with their authors' information. The code combines these two datasets into a single DataFrame and performs random shuffling to ensure a well-mixed distribution of data points.

Next, a text preprocessing step is applied to the tweet text. This includes converting the text to lowercase and removing all non-alphanumeric characters, leaving only letters and spaces. This basic preprocessing is a common first step in NLP tasks to reduce noise and normalize the input data.

## Data Processing

The core of the data processing step is the `create_tweet_pairs` function. This function takes the dataset and generates pairs of tweets, either from the same author or different authors. The pairs are assigned binary labels: 1 for same-author pairs and 0 for different-author pairs.

The function has two key parameters:

- `num_pairs_per_author`: This controls the number of tweet pairs to generate for each author in the dataset.
- `same_author_ratio`: This determines the ratio of same-author pairs to different-author pairs in the generated dataset.

The function ensures that no tweet is used more than once for the same author when creating pairs. It also tries to distribute the different-author pairs evenly across all other authors in the dataset.

After generating the tweet pairs, the function splits the data into training and test sets using scikit-learn's `train_test_split` function.

## Modeling and Evaluation

The project defines a PyTorch model called `TweetSimilarityModel`, which inherits from `nn.Module`. This model takes a pre-trained BERT model (`bert-base-uncased`) and adds a linear layer on top to output a single similarity score between 0 and 1.

During the forward pass, the model computes the BERT output for each input tweet, performs mean pooling over the token representations, and passes the pooled representation through the linear layer. The final output is then passed through a sigmoid activation to obtain a similarity score between 0 and 1.

The model is trained using binary cross-entropy loss and the Adam optimizer. The training loop iterates over the generated tweet pairs in the training set, computes the similarity scores for each pair, calculates the loss, and updates the model parameters accordingly.

After training, the model is evaluated on the test set. For each pair in the test set, the model computes the similarity score, and a threshold (in this case, 0.5) is applied to convert the score into a binary prediction (0 or 1). The predicted labels are then compared to the true labels, and precision, recall, and F1 scores are calculated using scikit-learn's evaluation metrics.

The notebook mentions that the actual scores achieved are Precision: 0.7819526627218935, Recall: 1.0, and F1 Score: 0.8776357297028059, indicating a reasonably good performance of the model on this task.
