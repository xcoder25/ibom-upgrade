import re
import math
from typing import Dict, Any, List, Tuple

# --- Pre-defined ML training dataset of intents ---
DATASET: List[Tuple[str, str]] = [
    # GREETINGS
    ("hello", "greetings"),
    ("hi nora", "greetings"),
    ("hello there", "greetings"),
    ("hi", "greetings"),
    ("hey", "greetings"),
    ("good morning", "greetings"),
    ("good afternoon", "greetings"),
    ("good evening", "greetings"),
    ("howdy", "greetings"),
    ("what's up", "greetings"),
    ("sup", "greetings"),
    ("yo", "greetings"),
    
    # BALANCE
    ("check balance", "balance"),
    ("how much money is in my account", "balance"),
    ("what is my balance", "balance"),
    ("check account balance", "balance"),
    ("show my savings balance", "balance"),
    ("how much do i have", "balance"),
    ("my account balance", "balance"),
    ("check savings balance", "balance"),
    ("show usd balance", "balance"),
    ("how many dollars do i have", "balance"),
    ("my dollar balance", "balance"),
    ("check wallet", "balance"),
    ("check mortgage savings", "balance"),
    ("mortgage goal status", "balance"),
    
    # TRANSFER
    ("transfer money", "transfer"),
    ("send money", "transfer"),
    ("send 5000 to gtbank", "transfer"),
    ("transfer 10000 naira to access bank", "transfer"),
    ("make a transfer to zenith", "transfer"),
    ("pay five thousand naira to palm pay", "transfer"),
    ("transfer fifty thousand to access", "transfer"),
    ("wire cash to zenith", "transfer"),
    ("send money to my friend", "transfer"),
    ("transfer funds", "transfer"),
    ("send 20000 to ibom bank", "transfer"),
    
    # PAY BILLS
    ("pay dstv", "pay_bill"),
    ("buy airtime", "pay_bill"),
    ("recharge mtn airtime", "pay_bill"),
    ("recharge glo airtime", "pay_bill"),
    ("buy data bundle", "pay_bill"),
    ("pay gotv", "pay_bill"),
    ("renew starlink subscription", "pay_bill"),
    ("pay spectranet", "pay_bill"),
    ("pay electricity bill", "pay_bill"),
    ("pay nepa bill", "pay_bill"),
    ("buy electricity token", "pay_bill"),
    ("pay waec pin", "pay_bill"),
    ("register jamb", "pay_bill"),
    ("fund sportybet", "pay_bill"),
    ("pay bet9ja", "pay_bill"),
    ("netflix subscription", "pay_bill"),
    
    # FREEZE CARD
    ("freeze card", "freeze_card"),
    ("lock card", "freeze_card"),
    ("block card", "freeze_card"),
    ("suspend my card", "freeze_card"),
    ("lock debit card", "freeze_card"),
    ("freeze my mastercard", "freeze_card"),
    
    # UNFREEZE CARD
    ("unfreeze card", "unfreeze_card"),
    ("unlock card", "unfreeze_card"),
    ("activate my card", "unfreeze_card"),
    ("unblock card", "unfreeze_card"),
    ("unfreeze mastercard", "unfreeze_card"),
    
    # LOAN / MORTGAGE
    ("mortgage rate", "loan_query"),
    ("apply for a loan", "loan_query"),
    ("how much can i borrow", "loan_query"),
    ("home loan query", "loan_query"),
    ("national housing fund", "loan_query"),
    ("interest rates for home purchase", "loan_query"),
    ("how to get a mortgage", "loan_query"),
    
    # PIN / SETTINGS
    ("change pin", "change_pin"),
    ("update security pin", "change_pin"),
    ("reset transaction pin", "change_pin"),
    ("change account pin", "change_pin"),
    ("change transaction settings", "change_pin"),
    
    # HISTORY
    ("recent transactions", "history"),
    ("show my statement", "history"),
    ("view activity history", "history"),
    ("print account statement", "history"),
    ("generate statement", "history"),
    ("what was my last payment", "history"),
    
    # HELP
    ("help", "help"),
    ("what can you do", "help"),
    ("commands", "help"),
    ("nora capabilities", "help"),
    ("show help menu", "help"),
    ("list options", "help"),
    
    # SMALLTALK
    ("tell me a joke", "smalltalk"),
    ("joke", "smalltalk"),
    ("how is the weather", "smalltalk"),
    ("weather forecast", "smalltalk"),
    ("what time is it", "smalltalk"),
    ("what date is today", "smalltalk"),
    ("who are you", "smalltalk"),
    ("what is your name", "smalltalk"),
    ("introduce yourself", "smalltalk"),
    ("how are you", "smalltalk"),
    ("are you ok", "smalltalk"),
    ("thank you", "smalltalk"),
    ("thanks", "smalltalk"),
    ("goodbye", "smalltalk"),
    ("bye", "smalltalk")
]

# --- Pure Python TF-IDF Vectorizer & Classifier ---
class AINLPProcessor:
    def __init__(self, dataset: List[Tuple[str, str]]):
        self.dataset = dataset
        self.documents = [doc.lower().strip() for doc, _ in dataset]
        self.labels = [label for _, label in dataset]
        
        # Tokenize and build vocabulary
        self.vocab = set()
        for doc in self.documents:
            words = self._tokenize(doc)
            self.vocab.update(words)
        self.vocab = sorted(list(self.vocab))
        self.vocab_idx = {word: i for i, word in enumerate(self.vocab)}
        
        # Compute Document Frequency (DF)
        self.df = {word: 0 for word in self.vocab}
        for doc in self.documents:
            words = set(self._tokenize(doc))
            for word in words:
                if word in self.df:
                    self.df[word] += 1
                    
        # Compute Inverse Document Frequency (IDF)
        self.num_docs = len(self.documents)
        self.idf = {}
        for word, freq in self.df.items():
            # smooth IDF to avoid division by zero
            self.idf[word] = math.log((1 + self.num_docs) / (1 + freq)) + 1
            
        # Compute TF-IDF vectors for dataset
        self.doc_vectors = [self._vectorize(doc) for doc in self.documents]

    def _tokenize(self, text: str) -> List[str]:
        # Clean text: remove non-alphanumeric (except standard spaces)
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return cleaned.split()

    def _vectorize(self, text: str) -> List[float]:
        words = self._tokenize(text)
        if not words:
            return [0.0] * len(self.vocab)
            
        # Term Frequency
        tf = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
            
        # TF-IDF
        vector = [0.0] * len(self.vocab)
        for w, count in tf.items():
            if w in self.vocab_idx:
                tf_val = count / len(words)
                vector[self.vocab_idx[w]] = tf_val * self.idf[w]
                
        # Normalize vector (L2 norm)
        norm = math.sqrt(sum(val ** 2 for val in vector))
        if norm > 0:
            vector = [val / norm for val in vector]
            
        return vector

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        return dot_product

    def predict_intent(self, text: str) -> Tuple[str, float]:
        query_vector = self._vectorize(text)
        best_similarity = -1.0
        best_label = "fallback"
        
        for doc_vector, label in zip(self.doc_vectors, self.labels):
            sim = self._cosine_similarity(query_vector, doc_vector)
            if sim > best_similarity:
                best_similarity = sim
                best_label = label
                
        # If similarity is too low, treat as fallback
        if best_similarity < 0.15:
            return "fallback", best_similarity
            
        return best_label, best_similarity

    # --- Natural Language Entity Extractor ---
    @staticmethod
    def parse_number_words(text: str) -> float:
        """Parses digits or text words into numbers (e.g. 'ten thousand five hundred' -> 10500)"""
        num_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
            "hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000
        }
        
        text = text.lower().replace(",", "").replace("-", " ")
        
        # Check if there is a direct numeric match (e.g. "5000" or "10000.50")
        digits_match = re.search(r'\d+(\.\d+)?', text)
        if digits_match:
            # Handle K/M suffix (e.g. "20k" -> 20000)
            val = float(digits_match.group(0))
            if re.search(rf'{digits_match.group(0)}k', text):
                return val * 1000
            if re.search(rf'{digits_match.group(0)}m', text):
                return val * 1000000
            return val
            
        # Parse words
        words = text.split()
        total = 0
        current = 0
        
        for w in words:
            if w in num_words:
                scale = num_words[w]
                if scale >= 100:
                    if current == 0:
                        current = 1
                    current *= scale
                    if scale >= 1000:
                        total += current
                        current = 0
                else:
                    current += scale
                    
        total += current
        return float(total) if total > 0 else 0.0

    @staticmethod
    def extract_entities(text: str, intent: str) -> Dict[str, Any]:
        entities = {}
        q = text.lower()
        
        # 1. Parse common banking names
        banks = ["opay", "palmpay", "palm pay", "ibom mortgage bank", "ibom bank", "gtbank", "access bank", "zenith bank", "access", "zenith", "firstbank", "first bank"]
        for bank in banks:
            if bank in q:
                entities["recipient_bank"] = bank.title()
                break
                
        # 2. Parse utility services
        utilities = {
            "airtime": ["mtn", "glo", "9mobile", "airtel", "etisalat"],
            "data": ["mtn data", "glo data", "airtel data", "9mobile data"],
            "tv": ["dstv", "gotv", "showmax", "netflix", "canal plus", "canal+"],
            "internet": ["starlink", "spectranet"],
            "electricity": ["phcn", "eedc", "ikeja electric", "ibadan electric", "nepa"],
            "education": ["waec", "jamb", "neco"],
            "betting": ["bet9ja", "sportybet", "1xbet"]
        }
        
        for util_type, keywords in utilities.items():
            for kw in keywords:
                if kw in q:
                    entities["bill_type"] = util_type.title()
                    entities["biller_name"] = kw.upper()
                    break
            if "bill_type" in entities:
                break
                
        # 3. Parse amount
        amount = AINLPProcessor.parse_number_words(text)
        if amount > 0:
            entities["amount"] = amount
            
        # 4. Parse account number or bill identifier
        acc_match = re.search(r'\b\d{10}\b', text)
        if acc_match:
            entities["account_number"] = acc_match.group(0)
            entities["identifier"] = acc_match.group(0)
            
        phone_match = re.search(r'\b(070|080|090|081|091|071)\d{8}\b', text)
        if phone_match:
            entities["identifier"] = phone_match.group(0)
            
        return entities

    # --- Conversational Response Generator ---
    def generate_response(self, text: str, intent: str, entities: Dict[str, Any], db: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        q = text.lower().strip()
        user_name = db["user"]["name"].split()[0]
        
        if intent == "greetings":
            import random
            greetings = [
                f"Hello {user_name}! I am Nora, your virtual banking assistant. How can I help you today?",
                f"Hi there! Ready to assist you with transactions, bills, or loan checks. What do you need?",
                f"Hey {user_name}! I'm active and listening. What can I do for you?"
            ]
            return random.choice(greetings), {"action": "speak"}
            
        elif intent == "balance":
            bal_ngn = db["user"]["savings_balance_ngn"]
            bal_usd = db["user"]["savings_balance_usd"]
            msg = f"Your current savings balance is ₦{bal_ngn:,.2f}, and your USD wallet balance is ${bal_usd:,.2f}."
            return msg, {"action": "display_balance", "balance_ngn": bal_ngn, "balance_usd": bal_usd}
            
        elif intent == "transfer":
            amount = entities.get("amount", 0.0)
            bank = entities.get("recipient_bank", "")
            acc = entities.get("account_number", "")
            
            if amount == 0.0:
                return "I understand you want to transfer money. How much would you like to send?", {"action": "prompt_more_info", "missing": "amount"}
            if not bank:
                return f"Sure, let's send ₦{amount:,.2f}. Which destination bank should I transfer to?", {"action": "prompt_more_info", "missing": "recipient_bank"}
                
            # If we have bank and amount, we can navigate to transfer and populate fields
            msg = f"Okay, preparing a transfer of ₦{amount:,.2f} to {bank}."
            if acc:
                msg += f" Account number: {acc}."
            else:
                msg += " Please input the 10-digit account number."
                
            return msg, {
                "action": "execute_transfer_form",
                "amount": amount,
                "recipient_bank": bank,
                "account_number": acc
            }
            
        elif intent == "pay_bill":
            amount = entities.get("amount", 0.0)
            biller = entities.get("biller_name", "")
            bill_type = entities.get("bill_type", "")
            ident = entities.get("identifier", "")
            
            if not biller:
                return "Which utility or provider would you like to pay? I support DSTV, GOTV, Starlink, MTN, Glo, WAEC, Jamb, Bet9ja and more.", {"action": "prompt_more_info", "missing": "biller_name"}
            if amount == 0.0:
                return f"How much would you like to pay for {biller}?", {"action": "prompt_more_info", "missing": "amount"}
                
            msg = f"Preparing to pay ₦{amount:,.2f} to {biller}."
            if ident:
                msg += f" Reference ID: {ident}."
            else:
                msg += " Please provide your phone or subscriber ID."
                
            return msg, {
                "action": "execute_bill_form",
                "amount": amount,
                "biller_name": biller,
                "bill_type": bill_type,
                "identifier": ident
            }
            
        elif intent == "freeze_card":
            return "Freezing your Mastercard. Hold on.", {"action": "freeze_card"}
            
        elif intent == "unfreeze_card":
            return "Unfreezing your Mastercard. Your card is active now.", {"action": "unfreeze_card"}
            
        elif intent == "loan_query":
            goal = db["user"]["mortgage_savings_ngn"]
            return f"Ibom Mortgage Bank offers home loans from 12% per annum. You can apply via the Mortgage tab. Your current mortgage goal is ₦{goal:,.2f}. Shall I open the mortgage section?", {"action": "navigate", "view": "mortgages"}
            
        elif intent == "change_pin":
            return "Opening your security settings page to update your transaction PIN.", {"action": "navigate", "view": "settings"}
            
        elif intent == "history":
            return "Displaying your recent transactions and statement logs.", {"action": "show_statement"}
            
        elif intent == "help":
            return "Here's what I can do: check balances, transfer money, pay utility bills, freeze/unfreeze cards, check mortgage limits, and answer general banking queries. Just ask!", {"action": "show_help"}
            
        elif intent == "smalltalk":
            if "joke" in q:
                return "Why did the bank robber take a bath? Because he wanted to make a clean getaway! 😄", {"action": "speak"}
            if "weather" in q:
                return "I don't have real-time weather logs, but the financial weather in your wallet is looking bright! Your savings balance is ₦{db['user']['savings_balance_ngn']:,.2f}.", {"action": "speak"}
            if "time" in q:
                from datetime import datetime
                t = datetime.now().strftime("%I:%M %p")
                return f"The current time is {t}.", {"action": "speak"}
            if "date" in q:
                from datetime import datetime
                d = datetime.now().strftime("%A, %B %d, %Y")
                return f"Today is {d}.", {"action": "speak"}
            if "who are you" in q or "your name" in q or "introduce yourself" in q:
                return "I'm Nora, your super AI banking assistant for Ibom Mortgage Bank. I can handle transactions, navigate pages, freeze cards, and help you check balances using ML-driven intents!", {"action": "speak"}
            if "how are you" in q:
                return "I'm doing excellent! Fully charged and ready to assist you. What transaction shall we run?", {"action": "speak"}
            if "thank" in q:
                return "You're very welcome! I'm always happy to help. What else do you need?", {"action": "speak"}
            if "bye" in q or "goodbye" in q:
                return "Goodbye! Have a great day. Just say 'Hi Nora' when you need me.", {"action": "close_assistant"}
            return "I am Nora, your personal bank assistant. I'm here to help you navigate and perform transactions.", {"action": "speak"}
            
        else: # Fallback conversational reply (Human-like)
            # Try to answer from general banking knowledge or suggest prompts
            greetings_or_help = [
                "I didn't quite get that. Did you mean to 'check balance', 'transfer money', or 'pay a bill'?",
                "I'm here to assist with banking. Try asking: 'send 5000 to gtbank', 'freeze card', or 'pay DSTV 3500'.",
                "I can process that for you if you rephrase it. Say 'help' to see my commands."
            ]
            import random
            return random.choice(greetings_or_help), {"action": "help_options"}

# Instantiate a global instance of our trained classifier
nlp_processor = AINLPProcessor(DATASET)
