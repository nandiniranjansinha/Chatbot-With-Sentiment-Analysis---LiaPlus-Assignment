import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class SentimentAnalyzer:
    def __init__(self, use_cache=True):
        """
        Initialize sentiment analyzer with optimizations
        Args:
            use_cache: Use local cache to speed up loading
        """
        print("⏳ Loading sentiment model (cardiffnlp/twitter-roberta-base)...")

        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        # Speed optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=128,
            truncation=True,
            local_files_only=False
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torchscript=False
        )
        
        # Set to eval mode for inference
        self.model.eval()
        
        self.labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        print("✓ Sentiment model loaded successfully!\n")

    def keyword_override(self, text):
        """
        Strong overrides for Indian English and common expressions
        Returns: Sentiment label or None
        """
        t = text.lower()

        # Negative expressions (expanded)
        negative_patterns = [
            # Tiredness/Exhaustion
            "tired", "exhausted", "drained", "worn out", "fatigued",
            "sleepy", "burnout", "burnt out",
            
            # Hunger
            "hungry", "starving", "starved", "famished",
            
            # Negative feelings
            "not feeling great", "not great", "not okay", "not fine",
            "i am not okay", "i'm not okay", "not really fine",
            "sad", "upset", "angry", "frustrated", "annoyed",
            "disappointed", "disappointing", "terrible", "awful",
            "horrible", "worst", "hate", "dislike",
            
            # Stress/Anxiety
            "stress", "stressed", "anxious", "worried", "nervous",
            "overwhelmed", "tense", "panic",
            
            # Pain/Discomfort
            "hurt", "pain", "painful", "sick", "ill", "unwell"
        ]

        # Neutral expressions (especially Indian English)
        neutral_patterns = [
            # Indian English colloquialisms
            "fine only", "fine yaar", "thik thak", "theek thak",
            "thik hai", "theek hai", "ok only", "okay only",
            "acha hai", "badhiya", "chalta hai",
            
            # Mild/Qualified statements
            "just fine", "just ok", "just okay", "it's fine",
            "could be better", "not bad", "so-so", "alright",
            "decent", "average", "okay i guess"
        ]

        # Positive expressions (expanded)
        positive_patterns = [
            # Strong positive
            "amazing", "awesome", "fantastic", "excellent",
            "wonderful", "brilliant", "superb", "outstanding",
            "perfect", "love", "loving", "loved",
            
            # Good feelings
            "so good", "very good", "great", "feel great", "feel good",
            "happy", "very happy", "super happy", "excited",
            "thrilled", "delighted", "glad", "pleased",
            
            # Special occasions
            "birthday", "celebration", "celebrate", "congratulations",
            "yay", "hooray", "finally", "at last"
        ]

        # Check patterns in order of priority
        if any(pattern in t for pattern in negative_patterns):
            # Exception: "not tired" should not be negative
            if "not tired" in t or "not stressed" in t or "not sad" in t:
                return "NEUTRAL"
            return "NEGATIVE"
        
        if any(pattern in t for pattern in neutral_patterns):
            return "NEUTRAL"
        
        if any(pattern in t for pattern in positive_patterns):
            return "POSITIVE"

        return None

    @torch.no_grad()
    def single_sentiment(self, text):
        """
        Analyze sentiment of a single message
        Returns: NEGATIVE, NEUTRAL, or POSITIVE
        """
        # Check keyword overrides first (faster)
        override = self.keyword_override(text)
        if override:
            return override

        # Fallback to ML model
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=False
            )
            
            outputs = self.model(**inputs)
            scores = F.softmax(outputs.logits, dim=1)[0]
            
            # Get highest scoring label
            label_idx = int(torch.argmax(scores))
            confidence = float(scores[label_idx])
            
            # Require higher confidence for neutral (reduce over-classification)
            if label_idx == 1 and confidence < 0.5:
                # If unsure about neutral, lean toward sentiment
                if scores[0] > scores[2]:
                    return "NEGATIVE"
                else:
                    return "POSITIVE"
            
            return self.labels[label_idx]
            
        except Exception as e:
            print(f"⚠️  Sentiment analysis error: {e}")
            return "NEUTRAL"

    def sentiment_score(self, label):
        """Convert sentiment label to numeric score"""
        score_map = {
            "NEGATIVE": -1,
            "NEUTRAL": 0,
            "POSITIVE": 1
        }
        return score_map.get(label, 0)

    def conversation_sentiment(self, messages):
        """
        Analyze overall conversation sentiment
        Returns: (overall_sentiment, trend, individual_sentiments)
        """
        if not messages:
            return "NEUTRAL", "No messages", []
        
        # Analyze each message
        sentiments = [self.single_sentiment(msg) for msg in messages]
        scores = [self.sentiment_score(s) for s in sentiments]

        # Calculate overall sentiment
        avg_score = sum(scores) / len(scores)

        if avg_score > 0.3:
            overall = "POSITIVE"
        elif avg_score < -0.3:
            overall = "NEGATIVE"
        else:
            overall = "NEUTRAL"

        # Analyze trend (compare first and last third)
        if len(scores) >= 3:
            first_third = sum(scores[:len(scores)//3])
            last_third = sum(scores[-(len(scores)//3):])
            
            if last_third > first_third + 0.5:
                trend = "Mood improved significantly"
            elif last_third > first_third:
                trend = "Mood improved slightly"
            elif last_third < first_third - 0.5:
                trend = "Mood worsened significantly"
            elif last_third < first_third:
                trend = "Mood worsened slightly"
            else:
                trend = "Mood remained stable"
        else:
            trend = "Conversation too short to detect trend"

        return overall, trend, sentiments

    def batch_sentiment(self, messages):
        """
        Analyze multiple messages efficiently (for future optimization)
        Returns: List of sentiment labels
        """
        return [self.single_sentiment(msg) for msg in messages]
