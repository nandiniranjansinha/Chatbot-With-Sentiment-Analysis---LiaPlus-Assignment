import subprocess
from sentiment import SentimentAnalyzer

class ChatBot:
    def __init__(self, model="phi"):
        self.model = model
        self.history = []
        self.sentiment = SentimentAnalyzer()

    def generate_reply(self, user_msg, sentiment):
        # Build better context with sentiment info
        context = ""
        if len(self.history) > 1:
            recent = self.history[-3:]
            context = "Previous messages:\n" + "\n".join([f"- {msg}" for msg in recent]) + "\n\n"
        
        # More specific tone instructions based on sentiment
        tone_prompts = {
            "NEGATIVE": "The user is feeling down. Be empathetic, supportive, and caring. Ask if they want to talk about it. Keep it brief (1-2 sentences).",
            "POSITIVE": "The user is happy! Match their enthusiasm! Be warm and celebratory. Keep it brief (1-2 sentences).", 
            "NEUTRAL": "Be friendly and conversational. Keep it brief (1-2 sentences)."
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
            "what else can i help",
            "let me think",
            "i'm here to chat",
            "tell me more"
        ]
        lower = text.lower()
        return any(phrase in lower for phrase in generic_phrases) or len(text) < 20

    def fallback_reply(self, user_msg, sentiment):
        """Generate sentiment-aware fallback responses"""
        lower_msg = user_msg.lower()
        
        # Specific keyword-based responses
        if "tired" in lower_msg or "exhausted" in lower_msg:
            return "I'm sorry you're feeling tired. Make sure to get some rest when you can. Is there anything I can help with?"
        
        if "sad" in lower_msg or "upset" in lower_msg:
            return "I'm sorry you're feeling sad. Do you want to talk about what's bothering you? I'm here to listen."
        
        if "birthday" in lower_msg:
            return "Happy Birthday! 🎉 I hope you have an amazing day filled with joy and celebration!"
        
        if "happy" in lower_msg or "great" in lower_msg:
            return "That's wonderful to hear! I'm so glad you're feeling good!"
        
        if "hungry" in lower_msg or "starving" in lower_msg:
            return "Sounds like it's time for a good meal! Hope you get to eat something delicious soon."
        
        # General sentiment-based fallbacks
        if sentiment == "NEGATIVE":
            return "I hear you. That sounds tough. Would you like to talk more about it?"
        elif sentiment == "POSITIVE":
            return "That's great! I'm happy for you!"
        else:
            return "I'm listening. Tell me more!"

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
