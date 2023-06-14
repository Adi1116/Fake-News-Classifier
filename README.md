# Fake-News-Classifier
This python work implements a fake news classifier model using LSTM (Long Short-Term Memory), a type of recurrent neural network (RNN). The model is trained to classify news articles as either fake or real based on their text content.

Here's a description of the steps in the code:

Importing Libraries: The necessary libraries are imported, including numpy, pandas, Keras (a deep learning library), and scikit-learn for data preprocessing and model building.

Loading and Preprocessing the Dataset: The dataset is loaded from a CSV file using pandas. The text data is preprocessed by removing unnecessary characters, converting to lowercase, and performing any other desired text cleaning techniques.

Creating Word Embeddings: Pre-trained word vectors (such as GloVe) are used to create word embeddings. These word vectors provide vector representations of words that capture semantic relationships. The code loads the pre-trained word vectors into memory for later use.

Preparing the Data for Training: The text data is tokenized using Keras' Tokenizer, which converts the text into sequences of word indices. The maximum number of words to consider as features is specified with max_features. The text sequences are then padded to have the same length using Keras' pad_sequences function. The labels are created based on the 'label' column in the dataset. Finally, the data is split into training and testing sets using scikit-learn's train_test_split function.

Building and Training the LSTM Model: The LSTM model architecture is defined using Keras' Sequential API. It consists of an Embedding layer, SpatialDropout1D layer for regularization, LSTM layer, and a Dense layer with sigmoid activation for binary classification. The model is compiled with binary cross-entropy loss and Adam optimizer. The model is then trained on the training data using the fit function.

Evaluating the Model's Performance: The trained model is evaluated on the test set using the evaluate function, which computes the loss and accuracy. The test loss and accuracy are printed to the console.
