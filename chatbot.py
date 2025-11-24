import subprocess
from sentiment import SentimentAnalyzer

class ChatBot:
    def __init__(self, model="phi"):
        self.model = model
        self.history = []
        self.sentiment = SentimentAnalyzer()

    # ------------------------------
    # Utility: Clean hallucinations
    # ------------------------------
    def clean_reply(self, text):
        if not text:
            return "(No response)"

        t = text.strip()

        # Remove disclaimers
        remove_phrases = [
            "as an ai", 
            "as a language model",
            "i cannot", 
            "i am not capable",
            "i do not have the capability",
            "i do not have the ability"
        ]
        for p in remove_phrases:
            if p in t.lower():
                t = t.split(".")[0]  # keep first sentence only

        # Remove paragraphs that are NOT conversational
        stop_keywords = [
            "consider that you are", "let's assume", "the following statements",
            "based on this information", "first,", "step"
        ]
        for key in stop_keywords:
            idx = t.lower().find(key)
            if idx != -1:
                t = t[:idx].strip()

        # Limit extreme long outputs
        if len(t.split()) > 45:
            t = " ".join(t.split()[:45]) + "..."

        return t.strip()

    # ------------------------------
    # LLM Call
    # ------------------------------
    def generate_reply(self, user_msg):

        prompt = (
            "You are a friendly, concise assistant.\n"
            "RULES:\n"
            "- Keep responses short (2–3 sentences max).\n"
            "- Stay on topic of the user message ONLY.\n"
            "- Do NOT generate stories, tasks, assignments, comparisons, or lists.\n"
            "- Do NOT explain your reasoning.\n"
            "- Do NOT say 'as an AI language model'.\n"
            "- Never mention capabilities or limitations.\n"
            "- Never provide instructions unless asked.\n"
            "- If user expresses emotion, respond with empathy.\n\n"
            f"User: {user_msg}\nAssistant:"
        )

        try:
            process = subprocess.Popen(
                ["ollama", "run", self.model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False
            )

            out_bytes, _ = process.communicate(
                input=prompt.encode("utf-8", "ignore")
            )

            out = out_bytes.decode("utf-8", "ignore")

            return self.clean_reply(out)

        except Exception as e:
            return f"(LLM error: {e})"

    # ------------------------------
    # Chat Loop
    # ------------------------------
    def run(self):
        print("Chatbot (Local Phi LLM + Sentiment Analysis). Type exit/quit/done.\n")

        while True:
            user = input("You: ")
            if user.lower() in ["exit", "quit", "done"]:
                break

            self.history.append(user)

            senti = self.sentiment.single_sentiment(user)
            bot_reply = self.generate_reply(user)

            print(f'\nUser: "{user}"')
            print(f"→ Sentiment: {senti}")
            print(f'Bot: "{bot_reply}"\n')

        final, trend, per_msg = self.sentiment.conversation_sentiment(self.history)

        print("\n--- FINAL SENTIMENT REPORT ---")
        print("Overall Conversation Sentiment:", final)
        print("Trend:", trend)
        print("Each message sentiment:", per_msg)
