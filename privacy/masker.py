import re, hashlib, os
SALT = os.environ.get("MASK_SALT","dev_salt_change_me")
PAN_RE = re.compile(r'\b([A-Z]{5}[0-9]{4}[A-Z])\b', flags=re.IGNORECASE)
IFSC_RE = re.compile(r'\b([A-Z]{4}0[0-9A-Z]{6})\b', flags=re.IGNORECASE)
AADHAAR_RE = re.compile(r'\b(\d{4}\s?\d{4}\s?\d{4})\b')
ACCOUNT_RE = re.compile(r'\b(\d{10,16})\b')
PHONE_RE = re.compile(r'\b(\+?91[\-\s]?[6-9]\d{9}|[6-9]\d{9})\b')
EMAIL_RE = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')

def _pseudonymize(value: str, prefix: str = "PID"):
    if value is None:
        return None
    h = hashlib.sha256((SALT + str(value)).encode('utf-8')).hexdigest()
    return f"{prefix}:{h[:12]}"

def mask_pan(text: str) -> str:
    def repl(m):
        pan = m.group(1).upper()
        return f"PAN:{pan[-4:]}"
    return PAN_RE.sub(repl, text)

def mask_ifsc(text: str) -> str:
    def repl(m):
        ifsc = m.group(1).upper()
        return ifsc[:4] + "0XXXXXX"
    return IFSC_RE.sub(repl, text)

def mask_aadhaar(text: str) -> str:
    def repl(m):
        digits = re.sub(r'\s+','', m.group(1))
        return f"AADHAAR:{digits[-4:]}"
    return AADHAAR_RE.sub(repl, text)

def mask_account_numbers(text: str) -> str:
    def repl(m):
        acct = m.group(1)
        return _pseudonymize(acct, prefix="ACC")
    return ACCOUNT_RE.sub(repl, text)

def mask_phone(text: str) -> str:
    def repl(m):
        num = re.sub(r'\D','', m.group(1))
        return f"PHONE:{num[-4:]}"
    return PHONE_RE.sub(repl, text)

def mask_email(text: str) -> str:
    return EMAIL_RE.sub("[EMAIL_REDACTED]", text)

def mask_text(text: str) -> str:
    if text is None:
        return text
    t = str(text)
    t = mask_pan(t)
    t = mask_ifsc(t)
    t = mask_aadhaar(t)
    t = mask_account_numbers(t)
    t = mask_email(t)
    t = mask_phone(t)
    return t

def mask_dataframe(df, columns=None, inplace=False):
    import pandas as pd
    if not inplace:
        df = df.copy()
    if columns is None:
        columns = [c for c in df.columns if df[c].dtype == 'O']
    for col in columns:
        df[col] = df[col].fillna("").astype(str).apply(mask_text)
    return df

if __name__ == '__main__':
    sample = "PAN ABCDE1234F account 123456789012 IFSC SBIN0001234 Aadhaar 1234 5678 9012 +919876543210 test@example.com"
    print(mask_text(sample))
