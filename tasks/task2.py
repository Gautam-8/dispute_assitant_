

def suggest_action(row):
    cat = row["predicted_category"].upper()
    status = str(row.get("status","")).upper()

    if cat == "DUPLICATE_CHARGE":
        return "Auto-refund", "Confirmed duplicate transaction; refund can be processed automatically."
    elif cat == "FAILED_TRANSACTION":
        return "Auto-refund", f"Transaction shows {status or 'FAILED'} status; funds not captured."
    elif cat == "FRAUD":
        return "Mark as potential fraud", "Customer indicated unauthorized or suspicious activity."
    elif cat == "REFUND_PENDING":
        # Bank involvement is sometimes required if refund is stuck
        return "Escalate to bank", "Refund initiated but not settled; requires bank follow-up."
    else:  # OTHERS
        return "Manual review", "No rule-based resolution possible; agent should investigate."


