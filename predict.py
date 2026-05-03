# ============================================================
#  predict.py — Predict on new feature vectors
#  Run this AFTER phishing_detection.py has been executed
#  Usage: python predict.py
# ============================================================

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from phishing_detection import load_dataset, preprocess, train_random_forest

print("[INFO] Loading data and training Random Forest for prediction...")
df = load_dataset()
X_train, X_test, y_train, y_test, X_train_s, X_test_s = preprocess(df)
model = train_random_forest(X_train, y_train)

feature_names = list(df.drop(columns=['Result']).columns)

print("\n" + "="*55)
print("  PHISHING URL PREDICTOR")
print("="*55)
print("Enter feature values for a URL (30 features).")
print("Values: -1 = phishing indicator | 0 = suspicious | 1 = legitimate")
print("Press ENTER to use default suspicious example.\n")

# Default example — suspicious/phishing URL features
default_features = [
    1,   # having_IP_Address         — has IP in URL (phishing)
    1,   # URL_Length                — long URL (phishing)
    1,   # Shortining_Service        — uses shortener (phishing)
    1,   # having_At_Symbol          — has @ symbol (phishing)
    1,   # double_slash_redirecting  — double slash (phishing)
    1,   # Prefix_Suffix             — has hyphen (phishing)
    1,   # having_Sub_Domain         — many subdomains (phishing)
   -1,   # SSLfinal_State            — no SSL (phishing)
   -1,   # Domain_registeration_length — short-lived domain (phishing)
    1,   # Favicon                   — external favicon (phishing)
   -1,   # port                      — non-standard port (phishing)
    1,   # HTTPS_token               — HTTPS in wrong place (phishing)
    1,   # Request_URL               — external requests (phishing)
    1,   # URL_of_Anchor             — anchor links off-domain (phishing)
    1,   # Links_in_tags             — off-domain tag links (phishing)
    1,   # SFH                       — server form handler suspicious (phishing)
    1,   # Submitting_to_email       — submits to email (phishing)
    1,   # Abnormal_URL              — abnormal URL pattern (phishing)
    1,   # Redirect                  — multiple redirects (phishing)
    1,   # on_mouseover              — status bar changes (phishing)
    1,   # RightClick                — right-click disabled (phishing)
    1,   # popUpWidnow               — popup window (phishing)
    1,   # Iframe                    — uses iframe (phishing)
   -1,   # age_of_domain             — young domain (phishing)
   -1,   # DNSRecord                 — no DNS record (phishing)
   -1,   # web_traffic               — low traffic (phishing)
   -1,   # Page_Rank                 — low page rank (phishing)
   -1,   # Google_Index              — not indexed (phishing)
   -1,   # Links_pointing_to_page   — few backlinks (phishing)
    1,   # Statistical_report        — in phishing DB (phishing)
]

print("Using default suspicious URL example.")
print("Features:", default_features)

features = np.array(default_features).reshape(1, -1)
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0]

print("\n" + "-"*40)
if prediction == 1:
    print("  RESULT: ⚠  PHISHING WEBSITE DETECTED")
else:
    print("  RESULT: ✓  LEGITIMATE WEBSITE")

print(f"  Confidence — Legitimate: {probability[0]*100:.1f}%  |  Phishing: {probability[1]*100:.1f}%")
print("-"*40)
