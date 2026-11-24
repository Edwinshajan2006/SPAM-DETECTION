# SMS Spam Detection

<div align="center">

### SMS Shield · Machine-learning based message classification

[![Python](https://img.shields.io/badge/Python-3.x-111827?style=flat-square&logo=python)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-111827?style=flat-square&logo=scikitlearn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-111827?style=flat-square&logo=streamlit)](https://streamlit.io/)

**[Open the live SMS Shield app](https://spam-detection1124.streamlit.app/)**

</div>

A text-classification project that predicts whether an SMS message is **spam** or **ham** (likely legitimate). The training pipeline converts message text into TF-IDF features and uses a Linear Support Vector Classifier.

## How it works

1. Load and clean the SMS Spam Collection dataset.
2. Split the data into training and test sets using a stratified split.
3. Convert message text into numerical features with `TfidfVectorizer`.
4. Train a `LinearSVC` classifier.
5. Evaluate the model and save the fitted pipeline as `spam_model.joblib`.
6. Load the saved model in the Streamlit application to classify user-entered messages.

## Model details

| Component | Configuration |
|---|---|
| Feature extraction | TF-IDF |
| N-grams | Unigrams and bigrams |
| Term-frequency scaling | Sublinear TF |
| Classifier | LinearSVC |
| Class weighting | Balanced |
| Test split | 20% |
| Random state | 42 |
| Evaluation | Accuracy, spam precision, spam recall, F1-score and confusion matrix |

### Reported evaluation

The project's held-out evaluation reported:

| Metric | Score |
|---|---:|
| Accuracy | 98.45% |
| Spam precision | 93.08% |
| Spam recall | 94.53% |
| Spam F1-score | 93.80% |

These are results from the project's evaluation run, not a guarantee that every future message will be classified correctly.

## Application

The Streamlit interface provides:

- A message input area and analysis action
- Spam / ham prediction feedback
- Example spam and ham messages
- Model-performance information
- A warning that automated predictions should be independently verified

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Edwinshajan2006/SPAM-DETECTION.git
cd SPAM-DETECTION
```

### 2. Create and activate a virtual environment

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

The repository includes a trained model file:

```bash
streamlit run app.py
```

### Retrain the model

The training script expects the SMS Spam Collection dataset at `data/SMSSpamCollection`. The dataset is not included in this repository.

After obtaining the dataset, place it at that path and run:

```bash
python train_model.py
```

The script evaluates the model and writes `spam_model.joblib` in the project directory.

## Project structure

```text
SPAM-DETECTION/
├── app.py              # Streamlit application
├── train_model.py      # Data preparation, training and evaluation
├── spam_model.joblib   # Saved trained model
├── requirements.txt    # Python dependencies
└── data/
    └── SMSSpamCollection  # Add the dataset here to retrain
```

## Limitations

- Predictions depend on the training data and learned vocabulary.
- New slang, languages, misspellings and changing spam tactics can affect performance.
- A message classified as ham is not guaranteed to be safe.
- Do not use the prediction as the only basis for trusting links, payment requests or senders.

## Future improvements

- [ ] Add a confusion-matrix visualization to the app
- [ ] Add reproducible evaluation output and dataset attribution
- [ ] Add automated tests for preprocessing and prediction
- [ ] Capture an up-to-date application screenshot

---

<div align="center">

**Built as a machine-learning project · Python / NLP / Linear SVM**

</div>
