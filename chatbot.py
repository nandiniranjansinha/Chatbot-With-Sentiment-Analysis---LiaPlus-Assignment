import subprocess
from sentiment import SentimentAnalyzer
import random


class ChatBot:
    def __init__(self, model="phi"):
        self.model = model
        self.history = []
        self.sentiment = SentimentAnalyzer()

    def generate_reply(self, user_msg, sentiment):
        # Build better context with sentiment info
        context = ""
        # if len(self.history) > 1:
        #     recent = self.history[-3:]
        #     context = "Previous messages:\n" + "\n".join([f"- {msg}" for msg in recent]) + "\n\n"
        
        # More specific tone instructions based on sentiment
        tone_prompts = {
        "NEGATIVE": "The user seems to be experiencing difficulty or a negative emotion. Respond in a calm, supportive, and professional manner. Keep responses concise and clear.",
        "POSITIVE": "The user seems to be in a positive mood. Match their tone in a professional and warm way. Keep responses concise.",
        "NEUTRAL": "Respond in a clear, professional, and neutral manner. Be helpful and concise."
        }

        
        tone = tone_prompts.get(sentiment, tone_prompts["NEUTRAL"])
        
        # More explicit prompt structure
        prompt = f"""{context}Current message: {user_msg}

{tone}

Respond naturally as a friendly chatbot. Do not use phrases like "as an AI" or "language model". Just reply directly.


Response:"""

        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=15,  # Increased timeout
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode == 0 and result.stdout.strip():
                reply = self.clean_reply(result.stdout)
                
                # Fallback to sentiment-based response if cleaned reply is too generic
                if self.is_generic_reply(reply):
                    return self.fallback_reply(user_msg, sentiment)
                    
                return reply
            else:
                return self.fallback_reply(user_msg, sentiment)

        except subprocess.TimeoutExpired:
            return self.fallback_reply(user_msg, sentiment)
        except Exception:
            return self.fallback_reply(user_msg, sentiment)

    def is_generic_reply(self, text):
        """Check if reply is too generic"""
        generic_phrases = [
            "i'm here to help",
            "how can i assist",
            "what else can i help",
            "let me think",
            "i'm here to chat",
            "tell me more",
            "as a professional"
        ]

        lower = text.lower()
        return any(phrase in lower for phrase in generic_phrases) or len(text) < 20

    def fallback_reply(self, user_msg, sentiment):
        import random
        lower = user_msg.lower()

        # --- NEGATION UNDERSTANDING ---
        if ("not" in lower or "ain't" in lower or "no longer" in lower):
            if "sad" in lower or "down" in lower or "upset" in lower or "tired" in lower:
                return "Understood. Glad you're not feeling that way anymore."

        # --- SENTIMENT-BASED PROFESSIONAL RESPONSES ---
        positive = [
        "That's good to hear. Would you like to share more?",
        "I'm glad things are going well. How else can I assist you?",
        "Sounds good. Let me know if there's anything specific you'd like to discuss."
        ]

        neutral = [
        "Alright. How can I assist further?",
        "Understood. Feel free to tell me more.",
        "Okay. Let me know how I can help."
        ]

        negative = [
        "I'm sorry you're feeling this way. Would you like to talk about it?",
        "I understand this can be difficult. I'm here to listen if you'd like to share more.",
        "That sounds challenging. Tell me more when you're ready."
        ]

        if sentiment == "POSITIVE":
            return random.choice(positive)
        elif sentiment == "NEGATIVE":
            return random.choice(negative)
        else:
            return random.choice(neutral)


    def clean_reply(self, text):
        if not text:
            return ""

        text = text.strip()
        
        # Remove common LLM artifacts
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            lower = line.lower()
            
            # Skip meta-commentary and AI-speak
            skip_phrases = [
                'as an ai', 'language model', 'i am an ai', 'i\'m an ai',
                'imagine', 'let\'s imagine', 'step 1', 'step 2',
                'consider', 'suppose', 'here are some', "here's a",
                'based on the', 'according to', 'in this scenario',
                'response:', 'bot:', 'assistant:'
            ]
            
            if any(phrase in lower for phrase in skip_phrases):
                continue
                
            # Keep only substantive lines
            if len(line) > 5:
                lines.append(line)
        
        text = ' '.join(lines)
        
        # Extract meaningful sentences
        sentences = []
        for s in text.split('.'):
            s = s.strip()
            if len(s) > 15 and not any(skip in s.lower() for skip in ['as an ai', 'language model']):
                sentences.append(s)
        
        # Take first 2 sentences
        sentences = sentences[:2]
        
        if not sentences:
            return ""
        
        result = '. '.join(sentences)
        if not result.endswith('.'):
            result += '.'
        
        # Word limit
        words = result.split()
        if len(words) > 40:
            result = ' '.join(words[:40]) + '...'
        
        return result

    def run(self):
        print("=" * 60)
        print("Chatbot Ready (Phi LLM + Sentiment Analysis)")
        print("=" * 60)
        print("Type 'exit', 'quit', or 'bye' to end.\n")

        while True:
            try:
                user = input("You: ").strip()
                
                if not user:
                    continue
                    
                if user.lower() in ["exit", "quit", "bye", "done"]:
                    print("\n👋 Goodbye!\n")
                    break

                self.history.append(user)
                senti = self.sentiment.single_sentiment(user)
                
                print(f"    [Sentiment: {senti}]")
                bot_reply = self.generate_reply(user, senti)
                print(f"Bot: {bot_reply}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted.\n")
                break
            except Exception as e:
                print(f"Error: {e}\n")
                continue

        if self.history:
            final, trend, per_msg = self.sentiment.conversation_sentiment(self.history)

            print("=" * 60)
            print("SENTIMENT REPORT")
            print("=" * 60)
            print(f"Overall: {final}")
            print(f"Trend: {trend}")
            print(f"\nMessages:")
            for i, (msg, sent) in enumerate(zip(self.history, per_msg), 1):
                preview = msg[:50] + '...' if len(msg) > 50 else msg
                print(f"  {i}. [{sent}] {preview}")
            print("=" * 60)
