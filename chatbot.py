import subprocess
from sentiment import SentimentAnalyzer

class ChatBot:
    def __init__(self, model="phi"):
        self.model = model
        self.history = []
        self.sentiment = SentimentAnalyzer()

    def generate_reply(self, user_msg, sentiment):
        context = ""
        if len(self.history) > 1:
            recent = self.history[-3:]
            context = "\n".join([f"User: {msg}" for msg in recent[-2:]]) + "\n"
        
        tone = {
            "NEGATIVE": "Be empathetic.",
            "POSITIVE": "Be enthusiastic.", 
            "NEUTRAL": "Be friendly."
        }.get(sentiment, "Be friendly.")
        
        prompt = f"{context}User: {user_msg}\n{tone} Reply briefly (1-2 sentences).\nBot:"

        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode == 0:
                return self.clean_reply(result.stdout)
            else:
                return "I'm here to help! What would you like to talk about?"

        except subprocess.TimeoutExpired:
            return "Let me think... what else can I help you with?"
        except Exception:
            return "I'm listening. Tell me more!"

    def clean_reply(self, text):
        if not text:
            return "I'm here to chat!"

        text = text.strip()
        lines = []
        
        for line in text.split('\n'):
            line = line.strip()
            lower = line.lower()
            
            skip = ['as an ai', 'language model', 'imagine', "let's", 'step 1', 
                    'consider', 'suppose', 'here are', "here's", 'based on']
            
            if any(s in lower for s in skip):
                continue
            if line:
                lines.append(line)
        
        text = ' '.join(lines)
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10][:2]
        
        if not sentences:
            return "I'm here to help!"
        
        result = '. '.join(sentences)
        if not result.endswith('.'):
            result += '.'
            
        words = result.split()
        if len(words) > 35:
            result = ' '.join(words[:35]) + '...'
        
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
