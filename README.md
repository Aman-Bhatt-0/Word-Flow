<div align="center">

# 🌌✨ Next Word Predictor  
### A Deep Learning Language Model with Real-Time Flask Web App  

<img src="https://img.shields.io/badge/Model-LSTM-blueviolet?style=for-the-badge" />
<img src="https://img.shields.io/badge/Framework-TensorFlow-ff6f00?style=for-the-badge" />
<img src="https://img.shields.io/badge/WebApp-Flask-000000?style=for-the-badge" />
<img src="https://img.shields.io/badge/Dataset-211653 Sentences-success?style=for-the-badge" />

A fully interactive, beautifully designed **Next Word Prediction System** trained on  
⭐ **211,653 carefully processed sentences** ⭐  

Predict the next word in real time using a clean & modern Flask-powered UI.  
</div>

---

## 🌠 Introduction

Human language is magical — a flowing stream of patterns, probabilities, and meaning.  
This project captures a small piece of that magic using a deep learning model that predicts  
**the next word** in a sentence with remarkable accuracy.

It blends:

- 🧠 Deep Learning (LSTMs)  
- 🧹 NLP preprocessing  
- 🌐 Real-time Flask web architecture  
- 🎨 A simple yet aesthetic frontend UI  

Together forming a complete end-to-end ML + Web deployment pipeline.

---

## 🚀 Project Highlights

- 🔤 **Trained on 211,653 real-world sentences**
- 🧩 **Tokenizer-driven preprocessing**
- 🔮 **Next-word prediction using LSTM**
- ⚡ **Instant real-time results on a sleek UI**
- 🌐 **REST API for integration**
- 🔧 **Extendable to GPT, Transformer models, or full text generation**

This project serves as a perfect template for:
- ML practise  
- NLP exploration  
- Flask deployment  
- AI-based interactive apps  

---

## 📊 Dataset Overview

The dataset consists of **211,653 natural language sentences**, cleaned and standardized before training.

### 🔧 Preprocessing Steps
- Normalize & lowercase text  
- Remove punctuation and unnecessary symbols  
- Tokenize using `Keras Tokenizer`  
- Build vocabulary index maps  
- Generate input-output training sequences  
- Pad all sequences to uniform length  

This ensures the LSTM model receives well-structured and meaningful data for learning.

---

## 🧠 Deep Learning Model Architecture

A carefully-tuned LSTM neural network powers the predictions.

### 🔨 Architecture Breakdown
- **Embedding Layer**  
  Converts word indices into dense vector embeddings  

- **LSTM Layer**  
  Learns temporal/contextual relationships between words  

- **Dense Softmax Layer**  
  Predicts probabilities across vocabulary  

### ⚙️ Training Parameters
- **Optimizer:** Adam  
- **Loss:** Categorical Crossentropy  
- **Metric:** Accuracy  
- **Epochs:** 20  
- **Batch Size:** 256  

Example training code:

