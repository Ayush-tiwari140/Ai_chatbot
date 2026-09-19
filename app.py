import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INCODE_VISION_CODE = """
# Incode Vision / Biometric Identity Check Simulation
import cv2, numpy as np

def run_incode_vision_pipeline(image_frame):
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Image Sharpness (Blur Detection)
    sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Biometric Face Detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    return {
        "verified": sharpness_score > 100 and len(faces) > 0,
        "sharpness": round(sharpness_score, 2),
        "faces_detected": len(faces)
    }
"""


KNOWLEDGE_BASE = [
    {
        "intent": "greeting",
        "patterns": ["hello", "hi", "hey there", "good morning", "greetings"],
        "response": "Hello! How can I assist you today?"
    },
    {
        "intent": "identity",
        "patterns": ["who are you", "what is your name", "tell me about yourself"],
        "response": "I am SmartBot, an intermediate NLP-driven terminal chatbot."
    },
    {
        "intent": "capabilities",
        "patterns": ["what can you do", "how do you work", "help me"],
        "response": "I analyze text using TF-IDF vectorization and cosine similarity to match your question to known intents."
    },
    {
        "intent": "creator",
        "patterns": ["who made you", "who is your developer", "who created you"],
        "response": "I was built using Python and Scikit-Learn's natural language processing tools."
    },
    {
        "intent": "farewell",
        "patterns": ["bye", "goodbye", "see you later", "exit", "quit"],
        "response": "Goodbye! Have a productive day."
    },
    {
        "intent": "incodevision_explanation",
        "patterns": [
            "what is incodevision", 
            "explain incodevision", 
            "what is in code vision",
            "what is incode vision",
            "tell me about incode vision",
            "incode vision company",
            "incode company",
            "what does incode do"
        ],
        "response": (
            "Incode (Incode Technologies) is an enterprise AI company providing identity "
            "verification and biometric authentication solutions (Incode Omni/Vision). "
            "Their technology automates document verification, passive liveness detection, "
            "and facial recognition for secure onboarding across banking, gaming, and enterprise systems."
        )
    },
    {
        "intent": "incodevision_code",
        "patterns": [
            "incodevision code", 
            "show me incodevision code", 
            "write code for incodevision", 
            "incodevision code example",
            "how to program incodevision"
        ],
        "response": f"Here is a code example simulating an Incode Vision biometric verification pipeline:\n{INCODE_VISION_CODE}"
    }
]


class NLPChatbot:
    def __init__(self, knowledge_base: list, confidence_threshold: float = 0.20):
        self.confidence_threshold = confidence_threshold
        # analyzer='word' with lowercase ensures compound tokens are matched properly
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        
        self.patterns = []
        self.responses = []
        
        for item in knowledge_base:
            for pattern in item["patterns"]:
                self.patterns.append(pattern)
                self.responses.append(item["response"])
                
        self.tfidf_matrix = self.vectorizer.fit_transform(self.patterns)

    def get_response(self, user_input: str) -> tuple[str, float]:
        user_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        best_match_idx = int(np.argmax(similarities))
        highest_score = float(similarities[best_match_idx])
        
        if highest_score >= self.confidence_threshold:
            return self.responses[best_match_idx], highest_score
        
        return "I'm not quite sure I understand. Could you rephrase your question?", highest_score


def run_bot():
    bot = NLPChatbot(KNOWLEDGE_BASE, confidence_threshold=0.20)
    print("--- SmartBot NLP Terminal Interface ---")
    print("Type 'exit' or 'bye' to quit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
            
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("SmartBot: Goodbye!")
            break
            
        response, confidence = bot.get_response(user_input)
        print(f"SmartBot (Confidence: {confidence:.2f}):\n{response}\n")


if __name__ == "__main__":
    run_bot()