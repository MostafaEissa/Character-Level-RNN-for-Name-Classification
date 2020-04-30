# Character Level RNN for Name classification

This is my implementation of [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) on name classification using Character Level RNN

# Model 

The model was trained using [PyTorch](http://pytorch.org/).

## Basic Usage 

To train the model with the default hyperparameters, simply run:

```
python main.py train-model PATH_TO_DATA
```

To predict the output of the model on new data, simply run:

```
python main.py predict-model PATH_TO_MODEL name_to_predict
```
