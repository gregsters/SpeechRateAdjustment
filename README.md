# Adaptive Synthetic Speech Rate Adjustment through Processing Dialogue Voice Records via Neural Networks

This repository contains an end-to-end experimental pipeline for investigating and modelling **user-preferred synthetic speech rates** through speech analysis, questionnaire responses, and machine-learning prediction. The project accompanies the research presented in the thesis and reproduces all preprocessing, feature extraction, imputation, and modelling steps described therein.

The system includes:

- Crowdsourced speech-perception dataset collected via Toloka  
- Synthetic speech generation (Yandex SpeechKit, 4 texts × 5 speeds)  
- Questionnaire-based preference elicitation with behavioural sanity checks  
- Extensive audio preprocessing (Levenshtein filtering, duration normalisation)  
- Extraction of short-term acoustic features  
- Ensemble methods for imputing missing preferred-speed labels  
- Supervised and deep learning models (Linear Regression, KNN, MLP, LSTM)  
- Evaluation using classification and error-based metrics  

---

## Abstract

Modern voice-assistant technologies rely heavily on pre-defined, static speech rates, despite substantial evidence that individuals differ in their preferred pace of computer-generated speech. This research investigates the feasibility of predicting a user’s **preferred synthetic speech rate** by analysing their own speech patterns and perceptual responses to controlled stimuli.

A dataset was collected on the Toloka crowdsourcing platform. Each participant listened to short texts synthesised at five different speeds (0.5, 0.8, 1.0, 1.2, 1.5) using Yandex SpeechKit and answered questions regarding clarity, pleasantness, and perceived need to slow down or speed up the recording. A noisy control text and behavioural sanity checks (rewind-blocking, answer-logic checks) ensured data reliability. Participants also recorded a short passage, providing speech data for analysis.

Recordings were filtered using Levenshtein distance applied to automatic transcriptions, duration-normalised through repetition, and transformed into a rich acoustic representation: MFCCs and deltas, spectral centroid, bandwidth, rolloff, zero-crossing rate, and CENS. Missing preferred-speed labels were imputed using several ensemble models, with a hybrid ensemble (SVC + Logistic Regression + XGBoost + LightGBM) achieving the lowest MSE.

Predictive models were then trained using acoustic features alone. Among all models, **LSTM networks** demonstrated the highest robustness across accuracy, precision, recall, and F1, validating the feasibility of adaptive speech-rate personalisation for future voice-assistant systems.

---

## System Overview

![Pipeline](./images/pipeline.png)

---

## Table of Contents

- [Data Collection](#data-collection)  
- [Installation](#installation)  
- [Notebook Catalogue](#notebook-catalogue)  
- [Evaluation Protocol](#evaluation-protocol)  
- [Metrics](#metrics)  
- [Results](#results)  
- [Limitations & Future Work](#limitations--future-work)

---

## Data Collection

### Synthetic Speech Stimuli

- Four simple English texts (A2 level) were selected to minimise lexical difficulty.  
- Each text was synthesised using **Yandex SpeechKit** at speeds: **0.5, 0.8, 1.0, 1.2, 1.5**  
- An additional *noisy control text* was introduced to detect inattentive or dishonest respondents.

### Questionnaire

Each participant answered:

1. Was this recording pleasant?  
2. Was it understandable?  
3. Would it be more comfortable if slower?  
4. Would it be more comfortable if faster?  

Logical combinations of answers were used to determine whether the current playback speed was satisfactory.

### Sanity Checks

- Rewinding was disabled programmatically.  
- Control text required specific answer patterns.  
- Submissions failing these checks were rejected.

### User Audio Recordings

Participants were given one of ten short texts to read aloud. These recordings were used for:

- transcription and quality filtering,  
- duration normalisation,  
- feature extraction,  
- predictive modelling.

---

## Installation

Requirements:

- Python ≥ 3.10  
- `ffmpeg` installed system-wide  
- All Python libraries listed in `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
