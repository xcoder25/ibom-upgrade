import os
import sys
import subprocess
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our AI underwriter
from ai_model import AICreditUnderwriter
from nlp_engine import nlp_processor

app = FastAPI(title="Ibom Mortgage Bank API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory Bank State Database ---
db = {
    "user": {
        "name": "Anietie Udo",
        "account_no": "1002938475",
        "bvn": "22233445566",
        "nhf_id": "NHF-8893-AKS",
        "savings_balance_ngn": 750000.00,
        "savings_balance_usd": 1250.00,
        "mortgage_savings_ngn": 1200000.00,
        "rsa_balance_ngn": 4500000.00,
        "pin": "1234",
    },
    "cards": [
        {
            "id": "card-1",
            "type": "Mastercard Gold",
            "number": "5399 2200 4499 8812",
            "expiry": "12/29",
            "cvv": "392",
            "frozen": False,
            "limit_ngn": 500000.00,
        }
    ],
    "transactions": [
        {"id": "tx-001", "sender": "SYSTEM", "recipient": "Anietie Udo", "amount": 1000000.00, "timestamp": "2026-06-15T10:00:00Z"},
        {"id": "tx-002", "sender": "Anietie Udo", "recipient": "NEPA/AEDC", "amount": 25000.00, "timestamp": "2026-06-18T14:30:00Z"},
        {"id": "tx-003", "sender": "SYSTEM", "recipient": "Anietie Udo", "amount": 950000.00, "timestamp": "2026-06-20T09:15:00Z"},
    ],
    "loans": [
        {
            "id": "loan-01",
            "type": "National Housing Fund (NHF)",
            "amount": 15000000.00,
            "status": "In Progress",
            "step": 2, # Out of 4 steps
            "date": "2026-06-19"
        }
    ],
    "ussd_sessions": {} # Tracks interactive phone USSD state
}

# --- Request Schemas ---
class TransferRequest(BaseModel):
    recipient_bank: str
    account_number: str
    amount: float
    pin: str

class BillRequest(BaseModel):
    bill_type: str # e.g. Airtime, MTN, Electricity, Cable
    biller_name: str # e.g. MTN, AEDC, DSTV
    identifier: str # phone number or meter number
    amount: float

class LoanApplicationRequest(BaseModel):
    requested_amount: float
    loan_term_years: int
    monthly_income: float
    monthly_expenses: float
    existing_loans: float
    use_rsa_equity: bool

class CardLimitRequest(BaseModel):
    card_id: str
    new_limit: float

class ChangePinRequest(BaseModel):
    old_pin: str
    new_pin: str

class USSDRequest(BaseModel):
    session_id: str
    input_text: str

class ChatRequest(BaseModel):
    message: str

# --- Helper function to run Rust audit engine ---
def run_rust_ledger_audit() -> Dict[str, Any]:
    # Check possible paths for Rust executable
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rust_bin = os.path.join(base_dir, "ledger_engine", "target", "release", "ledger_engine.exe")
    
    # Check debug target fallback
    if not os.path.exists(rust_bin):
        rust_bin = os.path.join(base_dir, "ledger_engine", "target", "debug", "ledger_engine.exe")
        
    # Check non-windows fallback
    if not os.path.exists(rust_bin):
        rust_bin = os.path.join(base_dir, "ledger_engine", "target", "release", "ledger_engine")
        
    if not os.path.exists(rust_bin):
        rust_bin = os.path.join(base_dir, "ledger_engine", "target", "debug", "ledger_engine")

    # Audit data structure matching Rust
    audit_data = {
        "transactions": db["transactions"],
        "initial_balance": 0.0 # Standard start base
    }

    if os.path.exists(rust_bin):
        try:
            # Spawn Rust ledger engine process
            process = subprocess.Popen(
                [rust_bin, "audit"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(input=json.dumps(audit_data))
            if process.returncode == 0:
                result = json.loads(stdout)
                result["engine_source"] = "Rust WebAssembly/Compiled Binary"
                return result
            else:
                return {
                    "verified": False,
                    "checksum": "",
                    "final_balance": 0.0,
                    "transaction_count": len(db["transactions"]),
                    "error": f"Rust execution failed: {stderr}",
                    "engine_source": "Rust Error Fallback"
                }
        except Exception as e:
            return {
                "verified": False,
                "checksum": "",
                "final_balance": 0.0,
                "transaction_count": len(db["transactions"]),
                "error": f"Error running Rust binary: {str(e)}",
                "engine_source": "Rust Process Exception"
            }
            
    # Python fallback emulation
    # Calculate checksum by chaining MD5/SHA-256 of transactions
    hasher = hashlib.sha256()
    current_balance = 0.0
    verified = True
    err_msg = None
    
    for tx in db["transactions"]:
        if tx["sender"] != "SYSTEM" and current_balance < tx["amount"]:
            verified = False
            err_msg = f"Insufficient balance for transaction ID: {tx['id']}"
            break
            
        if tx["sender"] == "SYSTEM":
            current_balance += tx["amount"]
        else:
            current_balance -= tx["amount"]
            
        tx_str = f"{tx['id']}{tx['sender']}{tx['recipient']}{tx['amount']}{tx['timestamp']}"
        hasher.update(tx_str.encode('utf-8'))
        
    return {
        "verified": verified,
        "checksum": hasher.hexdigest(),
        "final_balance": current_balance,
        "transaction_count": len(db["transactions"]),
        "error": err_msg,
        "engine_source": "Python Emulation Engine (Rust not compiled)"
    }

# --- API Routes ---

@app.get("/api/dashboard")
def get_dashboard():
    return {
        "user": db["user"],
        "cards": db["cards"],
        "transactions": db["transactions"][-5:], # Return last 5
        "loans": db["loans"]
    }

@app.post("/api/transfer")
def transfer_funds(req: TransferRequest):
    if req.pin != db["user"].get("pin", "1234"):
        raise HTTPException(status_code=400, detail="Invalid Transaction PIN")
        
    if req.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than zero")
        
    if db["user"]["savings_balance_ngn"] < req.amount:
        raise HTTPException(status_code=400, detail="Insufficient funds in savings account")
        
    # Process transfer
    tx_id = f"tx-00{len(db['transactions']) + 1}"
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # 1. Update balance
    db["user"]["savings_balance_ngn"] -= req.amount
    
    # 2. Add transaction
    new_tx = {
        "id": tx_id,
        "sender": db["user"]["name"],
        "recipient": f"{req.recipient_bank} ({req.account_number})",
        "amount": req.amount,
        "timestamp": timestamp
    }
    db["transactions"].append(new_tx)
    
    # 3. Trigger automation ledger integrity audit (Rust with Python fallback)
    audit_report = run_rust_ledger_audit()
    
    return {
        "status": "Success",
        "message": f"₦{req.amount:,.2f} transferred successfully",
        "transaction": new_tx,
        "audit": audit_report
    }

@app.post("/api/pay-bills")
def pay_bills(req: BillRequest):
    if req.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than zero")
        
    if db["user"]["savings_balance_ngn"] < req.amount:
        raise HTTPException(status_code=400, detail="Insufficient funds")
        
    tx_id = f"tx-00{len(db['transactions']) + 1}"
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Deduct balance
    db["user"]["savings_balance_ngn"] -= req.amount
    
    new_tx = {
        "id": tx_id,
        "sender": db["user"]["name"],
        "recipient": f"{req.bill_type}: {req.biller_name} ({req.identifier})",
        "amount": req.amount,
        "timestamp": timestamp
    }
    db["transactions"].append(new_tx)
    
    # Run audit
    audit_report = run_rust_ledger_audit()
    
    return {
        "status": "Success",
        "message": f"{req.bill_type} payment of ₦{req.amount:,.2f} processed",
        "transaction": new_tx,
        "audit": audit_report
    }

@app.post("/api/loans/apply")
def apply_loan(req: LoanApplicationRequest):
    # Check if they want to secure the downpayment using 25% of RSA balance
    rsa_balance = db["user"]["rsa_balance_ngn"] if req.use_rsa_equity else 0.0
    
    # Invoke AI credit engine
    assessment = AICreditUnderwriter.evaluate_credit_risk(
        monthly_income=req.monthly_income,
        monthly_expenses=req.monthly_expenses,
        existing_loans=req.existing_loans,
        requested_amount=req.requested_amount,
        loan_term_years=req.loan_term_years,
        credit_history_months=24, # Mock credit history
        rsa_balance=rsa_balance
    )
    
    if assessment["approved"]:
        # Register new loan application to database
        loan_id = f"loan-0{len(db['loans']) + 1}"
        new_loan = {
            "id": loan_id,
            "type": "Personal Finance" if req.requested_amount < 2000000 else "Instant AI Mortgage",
            "amount": req.requested_amount,
            "status": "Approved" if assessment["risk_level"] == "Low" else "Conditionally Approved",
            "step": 3 if assessment["risk_level"] == "Low" else 2,
            "date": datetime.utcnow().strftime("%Y-%m-%d")
        }
        db["loans"].append(new_loan)
        
    return {
        "decision": assessment,
        "current_loans": db["loans"]
    }

@app.post("/api/cards/toggle")
def toggle_card(data: Dict[str, str]):
    card_id = data.get("card_id")
    for card in db["cards"]:
        if card["id"] == card_id:
            card["frozen"] = not card["frozen"]
            status_text = "frozen" if card["frozen"] else "unfrozen"
            return {"status": "Success", "message": f"Card has been {status_text}", "card": card}
    raise HTTPException(status_code=404, detail="Card not found")

@app.post("/api/cards/limit")
def update_card_limit(req: CardLimitRequest):
    for card in db["cards"]:
        if card["id"] == req.card_id:
            card["limit_ngn"] = req.new_limit
            return {"status": "Success", "message": f"Daily limit updated to ₦{req.new_limit:,.2f}", "card": card}
    raise HTTPException(status_code=404, detail="Card not found")

@app.post("/api/user/change-pin")
def change_pin(req: ChangePinRequest):
    if db["user"].get("pin", "1234") != req.old_pin:
        raise HTTPException(status_code=400, detail="Incorrect old PIN")
    if len(req.new_pin) != 4 or not req.new_pin.isdigit():
        raise HTTPException(status_code=400, detail="New PIN must be a 4-digit number")
    db["user"]["pin"] = req.new_pin
    return {"status": "Success", "message": "Transaction PIN changed successfully"}

@app.post("/api/ussd")
def process_ussd(req: USSDRequest):
    sess_id = req.session_id
    user_input = req.input_text
    
    # State machine for USSD menu
    # Dialing *614*629# triggers the initial state
    if user_input == "*614*629#":
        db["ussd_sessions"][sess_id] = "HOME"
        return {
            "menu": (
                "--- IBOM MORTGAGE BANK ---\n"
                "Welcome to Akwa-Ibom e-Portal\n"
                "1. Check Account Balance\n"
                "2. Apply for Quick Mortgage\n"
                "3. RSA Pension Contribution Check\n"
                "4. Check Active NHF Application\n"
                "5. Cancel"
            ),
            "status": "ACTIVE"
        }
        
    current_state = db["ussd_sessions"].get(sess_id)
    if not current_state:
        return {"menu": "Error: Session timed out or invalid code.", "status": "CLOSED"}
        
    if current_state == "HOME":
        if user_input == "1":
            balance_msg = (
                f"Ibom Account Balances:\n"
                f"Savings: ₦{db['user']['savings_balance_ngn']:,.2f}\n"
                f"Mortgage Savings: ₦{db['user']['mortgage_savings_ngn']:,.2f}\n"
                f"USD Wallet: ${db['user']['savings_balance_usd']:,.2f}\n"
                f"0. Back"
            )
            db["ussd_sessions"][sess_id] = "BALANCE"
            return {"menu": balance_msg, "status": "ACTIVE"}
            
        elif user_input == "2":
            quick_mortgage_msg = (
                "Dialing AI Loan Assistant...\n"
                "Select mortgage purpose:\n"
                "1. National Housing Fund (NHF)\n"
                "2. Outright Home Purchase\n"
                "3. Home Renovation loan\n"
                "0. Back"
            )
            db["ussd_sessions"][sess_id] = "MORTGAGE_SELECT"
            return {"menu": quick_mortgage_msg, "status": "ACTIVE"}
            
        elif user_input == "3":
            rsa_msg = (
                f"RSA Equity Facilitation:\n"
                f"Current RSA Balance: ₦{db['user']['rsa_balance_ngn']:,.2f}\n"
                f"25% Equity Contribution Capacity: ₦{db['user']['rsa_balance_ngn'] * 0.25:,.2f}\n"
                f"Eligible Home Value: ₦{(db['user']['rsa_balance_ngn'] * 0.25) / 0.10:,.2f} (with 10% downpayment)\n"
                f"0. Back"
            )
            db["ussd_sessions"][sess_id] = "RSA_VIEW"
            return {"menu": rsa_msg, "status": "ACTIVE"}
            
        elif user_input == "4":
            # Show progress of active applications
            apps_text = "Active Loan Applications:\n"
            for app in db["loans"]:
                steps = ["Submitted", "Appraisal", "FMBN Board Review", "Disbursed"]
                step_name = steps[min(app["step"], 3)]
                apps_text += f"- {app['type']}: ₦{app['amount']:,.0f} [{step_name}]\n"
            apps_text += "0. Back"
            db["ussd_sessions"][sess_id] = "LOAN_PROGRESS"
            return {"menu": apps_text, "status": "ACTIVE"}
            
        else:
            # Terminate
            db["ussd_sessions"].pop(sess_id, None)
            return {"menu": "Thank you for banking with Ibom Mortgage Bank.", "status": "CLOSED"}
            
    # Handle Back Buttons (0)
    if user_input == "0":
        db["ussd_sessions"][sess_id] = "HOME"
        return {
            "menu": (
                "--- IBOM MORTGAGE BANK ---\n"
                "Welcome to Akwa-Ibom e-Portal\n"
                "1. Check Account Balance\n"
                "2. Apply for Quick Mortgage\n"
                "3. RSA Pension Contribution Check\n"
                "4. Check Active NHF Application\n"
                "5. Cancel"
            ),
            "status": "ACTIVE"
        }
        
    # Mortgage Select Logic
    if current_state == "MORTGAGE_SELECT":
        if user_input in ["1", "2", "3"]:
            loan_type = "NHF" if user_input == "1" else ("Outright Purchase" if user_input == "2" else "Renovation")
            db["ussd_sessions"][sess_id] = "HOME"
            return {
                "menu": f"Offer Generated for {loan_type} mortgage!\nCheck mobile app dashboard to complete underwriting analysis.\nPress 0 for main menu.",
                "status": "ACTIVE"
            }
            
    # Default fallbacks
    db["ussd_sessions"].pop(sess_id, None)
    return {"menu": "Session ended. Thank you.", "status": "CLOSED"}

# Session storage for Nora conversational memory
nora_session = {
    "pending_intent": None,
    "entities": {}
}

@app.post("/api/nora/chat")
def nora_chat(req: ChatRequest):
    global nora_session
    text = req.message
    q = text.lower().strip()
    
    # 1. Check for cancel/stop commands to reset session memory
    if any(kw in q for kw in ["cancel", "stop", "never mind", "nevermind", "reset"]):
        nora_session = {"pending_intent": None, "entities": {}}
        return {
            "intent": "cancel",
            "confidence": 1.0,
            "entities": {},
            "response": "Transaction cancelled. How else can I help you?",
            "action": {"action": "speak"}
        }

    # 2. Predict intent of the current message
    intent, confidence = nlp_processor.predict_intent(text)
    
    # 3. Handle context/memory
    if nora_session["pending_intent"] is not None:
        # High confidence greetings, help or smalltalk resets memory
        if intent in ["greetings", "help", "smalltalk"] and confidence > 0.6:
            nora_session = {"pending_intent": None, "entities": {}}
        else:
            intent = nora_session["pending_intent"]
            confidence = 1.0
            
    # 4. Extract entities
    new_entities = nlp_processor.extract_entities(text, intent)
    
    # 5. Merge new entities into session entities
    if nora_session["pending_intent"] == intent:
        nora_session["entities"].update(new_entities)
        entities = nora_session["entities"]
    else:
        entities = new_entities
        
    # 6. Generate response
    response_msg, action_data = nlp_processor.generate_response(text, intent, entities, db)
    
    # 7. Update session context based on response outcome
    if action_data.get("action") == "prompt_more_info":
        nora_session["pending_intent"] = intent
        nora_session["entities"] = entities
    else:
        # All slots satisfied or other action -> clear session state
        nora_session = {"pending_intent": None, "entities": {}}
        
    # Process side effects on backend DB
    if action_data.get("action") == "freeze_card":
        for card in db["cards"]:
            card["frozen"] = True
    elif action_data.get("action") == "unfreeze_card":
        for card in db["cards"]:
            card["frozen"] = False
            
    return {
        "intent": intent,
        "confidence": confidence,
        "entities": entities,
        "response": response_msg,
        "action": action_data
    }

