# Language Detection Perceptron Network

This project implements a perceptron-based neural network to detect the language of a given text. It uses a set of text files in different languages to train and test the model.

## Features

- **Language Classification**: Detects the language of input text based on character frequency.
- **Customizable Training**: Allows users to train the network with specified epochs.
- **Interactive Menu**: Provides an interactive menu for predictions, training, and inspection of language vectors and perceptron layers.

## How It Works

1. **Data Processing**:
   - Reads language files from the `lang` directory.
   - Extracts character frequency vectors for each language.
2. **Training**:
   - Trains the perceptron network using the extracted frequency vectors.
3. **Prediction**:
   - Takes an input text, computes the frequency vector, and predicts the language based on the trained perceptrons.

## Project Structure

- **Main Script**: Contains the training, testing, and menu-based interaction logic.
- **Perceptron Class**: Implements the perceptron logic (stored in `perceptron.py`).
- **Language Files**: Text samples in different languages stored in the `lang` directory.
