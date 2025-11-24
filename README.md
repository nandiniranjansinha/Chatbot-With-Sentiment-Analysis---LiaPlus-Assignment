# **Chatbot With Sentiment Analysis — Assignment Submission**

This repository contains my implementation of the **“Chatbot With Sentiment Analysis”** assignment.
The project fulfills **Tier 1 (mandatory)** and **Tier 2 (additional credit)** requirements using Python, a local LLM, and a Hugging Face sentiment model.

---

## 📌 **Project Overview**

This chatbot conducts a multi-turn conversation with the user, while simultaneously analyzing the emotional tone of each message.

The system performs:

### ✔ **Tier 1 (Mandatory) – Conversation-Level Sentiment**

* Maintains full conversation history
* Performs sentiment analysis across the entire chat
* Produces a **final sentiment summary** including:

  * Overall sentiment
  * Mood trend (improved / worsened / stable)

### ✔ **Tier 2 (Additional Credit) – Statement-Level Sentiment**

* Performs **sentiment classification for each user message**
* Displays:

  ```
  User: "<message>"
  → Sentiment: <POSITIVE/NEGATIVE/NEUTRAL>
  Chatbot: "<response>"
  ```
* Tracks sentiment shift across the conversation
* Shows a trend summary at the end

---

## 🚀 **How to Run the Project**

### **1️⃣ Create Virtual Environment**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Install & Run Ollama**

(Required for running the Phi LLM locally)

Download Ollama:
[https://ollama.com/download](https://ollama.com/download)

Verify installation:

```bash
ollama --version
```

Pull the Phi model:

```bash
ollama pull phi
```

### **4️⃣ Start the Chatbot**

```bash
python app.py
```

You will see:

```
Chatbot (Local Phi LLM + Sentiment Analysis). Type exit/quit/done.
```

Chat normally, then type **exit** to see the final sentiment report.

---

## 🧠 **Chosen Technologies**

### **1. Python**

* Core conversational loop
* Modular design (`chatbot.py`, `sentiment.py`, `app.py`)
* Clean separation of responsibilities

### **2. Local LLM — Phi via Ollama**

* Runs offline
* No API keys
* Fast and lightweight
* Used for generating chatbot replies

### **3. Hugging Face Sentiment Model**

Model used:

```
cardiffnlp/twitter-roberta-base-sentiment-latest
```

Why this model?

* Better sentiment accuracy
* Handles casual conversational text
* Supports **3-class sentiment**:

  * Negative
  * Neutral
  * Positive

---

## 🎚️ **Sentiment Logic (Explained)**

### **1️⃣ Per-Message Sentiment (Tier 2)**

Each input message goes through HuggingFace’s classifier:

* Tokenize message
* Disable truncation warnings
* Pass through RoBERTa model
* Compute:

```
sentiment = argmax(logits)
```

Mapped to:

* **0 → Negative**
* **1 → Neutral**
* **2 → Positive**

Stored in a list:

```python
self.sentiment_history.append(sentiment_label)
```

Displayed immediately to the user.

---

### **2️⃣ Conversation-Level Sentiment (Tier 1)**

After the user exits:

* All sentiments are aggregated
* Majority class → **overall conversation sentiment**
* First vs last sentiment → **mood trend**

Example:

```
Overall: POSITIVE
Trend: Mood improved.
```

---

## 📊 **Tier 2 Implementation Status**

| Tier                      | Requirement                                         | Status      |
| ------------------------- | --------------------------------------------------- | ----------- |
| **Tier 1**                | Full conversation sentiment analysis                | ✅ Completed |
| **Tier 2 (A)**            | Message-level sentiment                             | ✅ Completed |
| **Tier 2 (B)**            | Display sentiment beside each message               | ✅ Completed |
| **Tier 2 (C)**            | Sentiment trend analysis                            | ✅ Completed |
| **Optional Enhancements** | Customized emotional responses, cleaner LLM outputs | ✅ Partial   |

---

## 📁 **Repository Structure**

```
📦 Sentiment-Chatbot
 ┣ app.py                 # Entry point
 ┣ chatbot.py             # LLM response system
 ┣ sentiment.py           # Sentiment analyzer class
 ┣ requirements.txt
 ┗ README.md
```

---

## ⭐ **Additional Notes / Enhancements**

* Fully local, no API usage
* Clean modular code
* Neutrality adjustments for conversational messages
* Trend comparison logic added
* Reduced truncation warnings
* More natural fallback responses

---

## 📌 Notes

* This project is **strictly for an assignment**, not a production chatbot.
* Sentiment model runs on **CPU**, so first load takes ~1–2 minutes.
* Responses may vary since Phi is a generative model.

---

## 📜 **License**

This project is created **only for academic/assignment purposes**.
Not intended for commercial use.

---
