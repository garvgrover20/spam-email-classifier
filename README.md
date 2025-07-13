


# Spam Email Classifier

A machine learning project to classify SMS messages as spam or not spam using Python and scikit-learn.

## Overview

This project uses natural language processing (NLP) techniques and a Naive Bayes classifier to identify spam messages. The dataset is based on SMS text messages and includes preprocessing, model training, and evaluation steps.

## Features

- Data preprocessing (lowercasing, punctuation removal, stopword removal)
- TF-IDF vectorization
- Multinomial Naive Bayes classifier
- Model evaluation with accuracy, precision, and recall

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/garvgrover20/spam-email-classifier.git
   cd spam-email-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (If requirements.txt is missing, see **Dependencies** below.)

### Usage

Run the classifier script:

```bash
python spam_classifier.py
```

You should see the output metrics (accuracy, precision, recall) in your terminal.

### Dependencies

- pandas
- scikit-learn
- nltk

Install them with:
```bash
pip install pandas scikit-learn nltk
```

## Project Structure

```
.
├── spam_classifier.py
├── README.md
└── (requirements.txt)
```

## Dataset

The dataset is loaded automatically from [this public source](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv) and contains SMS messages labeled as "ham" (not spam) or "spam".

## Results

After running the script, you will see evaluation metrics (accuracy, precision, recall) printed in the terminal.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions or improvements.

## License

This project is licensed under the MIT License.

