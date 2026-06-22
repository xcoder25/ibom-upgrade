import math
from typing import Dict, Any, List

class AICreditUnderwriter:
    """
    Automated Credit Underwriting System.
    Uses structural decision logic to simulate a Random Forest / Decision Tree model
    trained on personal mortgage risk databases.
    """
    
    @staticmethod
    def evaluate_credit_risk(
        monthly_income: float,
        monthly_expenses: float,
        existing_loans: float,
        requested_amount: float,
        loan_term_years: int,
        credit_history_months: int,
        rsa_balance: float = 0.0
    ) -> Dict[str, Any]:
        
        # 1. Calculate key ratios
        total_debt_service = monthly_expenses + existing_loans
        debt_to_income_ratio = (total_debt_service / monthly_income) if monthly_income > 0 else 1.0
        
        # 2. Base Credit Score Calculation (ranging from 300 to 850)
        # Factors: Payment history (35%), Debt ratio (30%), Credit history length (15%), RSA security (20%)
        
        # Base starts at 500
        score = 500
        
        # Credit history length factor (max +100 points)
        history_points = min(100, int((credit_history_months / 120) * 100))
        score += history_points
        
        # Debt-to-income factor (max +150 points for low ratio, penalizes for high ratio)
        if debt_to_income_ratio < 0.20:
            score += 150
        elif debt_to_income_ratio < 0.35:
            score += 100
        elif debt_to_income_ratio < 0.50:
            score += 50
        else:
            score -= 100 # Heavy penalty
            
        # RSA Pension Security factor (RSA 25% equity contribution option)
        # Adds points if they have substantial retirement savings to back up their equity (max +100 points)
        if rsa_balance > 0:
            rsa_to_loan_ratio = rsa_balance / requested_amount if requested_amount > 0 else 0
            if rsa_to_loan_ratio > 0.25:
                score += 100
            elif rsa_to_loan_ratio > 0.10:
                score += 50
                
        # Income bracket booster
        if monthly_income > 1000000: # 1M NGN
            score += 50
        elif monthly_income > 500000:
            score += 25
            
        # Bound score between 300 and 850
        score = max(300, min(850, score))
        
        # 3. Decision Tree Rules (Underwriting Logic)
        approved = True
        rejection_reasons = []
        
        # Rule A: Debt-to-Income is too high (> 50%)
        if debt_to_income_ratio > 0.50:
            approved = False
            rejection_reasons.append("Debt-to-Income (DTI) ratio exceeds safety threshold of 50%.")
            
        # Rule B: Insufficient Credit History
        if credit_history_months < 6:
            approved = False
            rejection_reasons.append("Credit history is too short (minimum 6 months required).")
            
        # Rule C: Credit Score limits
        if score < 580:
            approved = False
            rejection_reasons.append("Credit score is below the minimum required mortgage threshold of 580.")
            
        # Rule D: Maximum Loan Capacity Check (typically 4x annual income for retail banking)
        max_loan_limit = (monthly_income * 12) * 5.0 # Up to 5x annual income for mortgage
        
        # If RSA balance covers 25% equity, boost capacity by 20%
        if rsa_balance >= (requested_amount * 0.25):
            max_loan_limit *= 1.2
            
        if requested_amount > max_loan_limit:
            approved = False
            rejection_reasons.append(
                f"Requested loan amount exceeds maximum lending capacity (Limit: ₦{max_loan_limit:,.2f})."
            )
            
        # 4. Risk Level determination
        if score >= 720:
            risk_level = "Low"
        elif score >= 620:
            risk_level = "Medium"
        else:
            risk_level = "High"
            
        # 5. Recommendations
        recommendations = []
        if not approved:
            if debt_to_income_ratio > 0.40:
                recommendations.append("Reduce non-essential monthly expenses or pay down existing retail loans to lower your DTI.")
            if credit_history_months < 12:
                recommendations.append("Build a longer track record by maintaining small, active micro-savings and micro-credit accounts.")
            if score < 600:
                recommendations.append("Increase your target mortgage savings plan balances to demonstrate structured saving habits.")
            if requested_amount > max_loan_limit:
                recommendations.append(f"Consider selecting a property value within ₦{max_loan_limit:,.2f} or increasing your initial equity down payment.")
        else:
            recommendations.append("Approved for instant mortgage facilitation! Complete document upload to finalize.")
            if score < 750:
                recommendations.append("To unlock premium lower interest rates, maintain your current positive balance track record for 3 more months.")
                
        # 6. Simulated Monthly Amortization (using Python math)
        # Interest rate varies based on credit score (premium tiers)
        # Base rate 18% NGN. Best credit score gets 12%. High risk gets 24%.
        if score >= 750:
            applied_rate = 12.0
        elif score >= 680:
            applied_rate = 15.0
        elif score >= 600:
            applied_rate = 18.0
        else:
            applied_rate = 22.0
            
        monthly_rate = (applied_rate / 100.0) / 12.0
        total_months = loan_term_years * 12
        
        if monthly_rate == 0.0:
            monthly_payment = requested_amount / total_months
        else:
            monthly_payment = requested_amount * (monthly_rate * math.pow(1.0 + monthly_rate, total_months)) / (math.pow(1.0 + monthly_rate, total_months) - 1.0)
            
        return {
            "credit_score": score,
            "risk_level": risk_level,
            "approved": approved,
            "interest_rate": applied_rate,
            "max_eligible_loan": max_loan_limit,
            "monthly_payment": monthly_payment,
            "reasons": rejection_reasons,
            "recommendations": recommendations,
            "dti_ratio": debt_to_income_ratio * 100.0,
            "metrics": {
                "history_score": history_points,
                "dti_score": 150 if debt_to_income_ratio < 0.20 else (100 if debt_to_income_ratio < 0.35 else 50),
                "rsa_booster": 100 if rsa_balance >= (requested_amount * 0.25) else 0
            }
        }
