import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class SentimentAnalyzer:
    def __init__(self):
        print("Loading sentiment model... This may take 1–2 minutes on CPU.\n")

        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=256,
            truncation=True
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

        print("\nSentiment model loaded successfully.\n")

    # ------------------------------------------------
    # STRONG OVERRIDES FOR INDIAN ENGLISH EXPRESSIONS
    # ------------------------------------------------
    def keyword_override(self, text):
        t = text.lower()

        negative = [
            "not feeling great", "not great", "not okay",
            "i am not okay", "i'm not okay",
            "not fine", "not really fine",
            "exhausted", "tired", "sad", "upset", "angry",
            "disappointed", "stress", "stressed", "anxious",
            "worried", "frustrated"
        ]

        # special Indian English cases (NEUTRAL)
        neutral = [
            "fine only", "fine yaar", "thik thak", "theek thak",
            "thik hai", "theek hai", "ok only", "okay only",
            "just fine", "just ok", "just okay", "it's fine"
        ]

        positive = [
            "amazing", "so good", "very good",
            "i feel great", "i feel good",
            "very happy", "super happy", "awesome"
        ]

        if any(w in t for w in negative):
            return "NEGATIVE"
        if any(w in t for w in neutral):
            return "NEUTRAL"
        if any(w in t for w in positive):
            return "POSITIVE"

        return None

    # ------------------------------------------------
    # MODEL PREDICTION
    # ------------------------------------------------
    def single_sentiment(self, text):
        override = self.keyword_override(text)
        if override:
            return override

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        scores = F.softmax(outputs.logits, dim=1)[0]
        label = self.labels[torch.argmax(scores)]
        return label

    def sentiment_score(self, label):
        if label == "NEGATIVE": return -1
        if label == "NEUTRAL": return 0
        return 1

    def conversation_sentiment(self, messages):
        sentiments = [self.single_sentiment(m) for m in messages]
        scores = [self.sentiment_score(s) for s in sentiments]

        avg = sum(scores)/len(scores)

        if avg > 0.25:
            final = "POSITIVE"
        elif avg < -0.25:
            final = "NEGATIVE"
        else:
            final = "NEUTRAL"

        trend = (
            "Mood improved." if scores[-1] > scores[0]
            else "Mood worsened." if scores[-1] < scores[0]
            else "Mood stable."
        )

        return final, trend, sentiments
