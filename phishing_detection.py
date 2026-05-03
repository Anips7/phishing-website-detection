# ============================================================
#  Phishing Website Detection Using Machine Learning
#  Class Project | All 5 Models
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os

# ── Output directory ──────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# 1. LOAD / GENERATE DATASET
# =============================================================
def load_dataset():
    """
    Tries to load UCI Phishing dataset.
    Falls back to a realistic synthetic dataset if file not found.
    To use the real dataset:
        1. Download from: https://archive.ics.uci.edu/ml/datasets/Phishing+Websites
        2. Save as data/phishing.csv
        3. This function will load it automatically.
    """
    csv_path = "data/phishing.csv"
    if os.path.exists(csv_path):
        print("[INFO] Loading real UCI dataset...")
        df = pd.read_csv(csv_path)
        # UCI dataset uses -1/0/1 encoding; map target to 0/1
        if 'Result' in df.columns:
            df['Result'] = df['Result'].map({-1: 1, 1: 0})
        return df
    else:
        print("[INFO] Real dataset not found. Generating synthetic dataset...")
        return generate_synthetic_dataset()


def generate_synthetic_dataset(n=5000, seed=42):
    """
    Generates a realistic synthetic phishing dataset with 30 features.
    Labels: 1 = phishing, 0 = legitimate
    """
    np.random.seed(seed)
    n_phish = n // 2
    n_legit = n - n_phish

    def make_samples(n, phish):
        p = phish
        return {
            'having_IP_Address':       np.random.choice([-1, 1], n, p=[0.85, 0.15] if not p else [0.25, 0.75]),
            'URL_Length':              np.random.choice([-1, 0, 1], n, p=[0.6, 0.2, 0.2] if not p else [0.1, 0.2, 0.7]),
            'Shortining_Service':      np.random.choice([-1, 1], n, p=[0.95, 0.05] if not p else [0.3, 0.7]),
            'having_At_Symbol':        np.random.choice([-1, 1], n, p=[0.99, 0.01] if not p else [0.4, 0.6]),
            'double_slash_redirecting':np.random.choice([-1, 1], n, p=[0.97, 0.03] if not p else [0.35, 0.65]),
            'Prefix_Suffix':           np.random.choice([-1, 1], n, p=[0.92, 0.08] if not p else [0.2, 0.8]),
            'having_Sub_Domain':       np.random.choice([-1, 0, 1], n, p=[0.6, 0.3, 0.1] if not p else [0.1, 0.3, 0.6]),
            'SSLfinal_State':          np.random.choice([-1, 0, 1], n, p=[0.05, 0.1, 0.85] if not p else [0.6, 0.2, 0.2]),
            'Domain_registeration_length': np.random.choice([-1, 1], n, p=[0.15, 0.85] if not p else [0.7, 0.3]),
            'Favicon':                 np.random.choice([-1, 1], n, p=[0.1, 0.9] if not p else [0.6, 0.4]),
            'port':                    np.random.choice([-1, 1], n, p=[0.05, 0.95] if not p else [0.5, 0.5]),
            'HTTPS_token':             np.random.choice([-1, 1], n, p=[0.97, 0.03] if not p else [0.3, 0.7]),
            'Request_URL':             np.random.choice([-1, 1], n, p=[0.8, 0.2] if not p else [0.2, 0.8]),
            'URL_of_Anchor':           np.random.choice([-1, 0, 1], n, p=[0.7, 0.2, 0.1] if not p else [0.15, 0.25, 0.6]),
            'Links_in_tags':           np.random.choice([-1, 0, 1], n, p=[0.6, 0.3, 0.1] if not p else [0.1, 0.3, 0.6]),
            'SFH':                     np.random.choice([-1, 0, 1], n, p=[0.7, 0.2, 0.1] if not p else [0.1, 0.2, 0.7]),
            'Submitting_to_email':     np.random.choice([-1, 1], n, p=[0.98, 0.02] if not p else [0.3, 0.7]),
            'Abnormal_URL':            np.random.choice([-1, 1], n, p=[0.85, 0.15] if not p else [0.2, 0.8]),
            'Redirect':                np.random.choice([0, 1], n, p=[0.95, 0.05] if not p else [0.4, 0.6]),
            'on_mouseover':            np.random.choice([-1, 1], n, p=[0.97, 0.03] if not p else [0.3, 0.7]),
            'RightClick':              np.random.choice([-1, 1], n, p=[0.95, 0.05] if not p else [0.2, 0.8]),
            'popUpWidnow':             np.random.choice([-1, 1], n, p=[0.9, 0.1] if not p else [0.25, 0.75]),
            'Iframe':                  np.random.choice([-1, 1], n, p=[0.93, 0.07] if not p else [0.2, 0.8]),
            'age_of_domain':           np.random.choice([-1, 1], n, p=[0.1, 0.9] if not p else [0.75, 0.25]),
            'DNSRecord':               np.random.choice([-1, 1], n, p=[0.05, 0.95] if not p else [0.65, 0.35]),
            'web_traffic':             np.random.choice([-1, 0, 1], n, p=[0.1, 0.1, 0.8] if not p else [0.5, 0.2, 0.3]),
            'Page_Rank':               np.random.choice([-1, 1], n, p=[0.1, 0.9] if not p else [0.7, 0.3]),
            'Google_Index':            np.random.choice([-1, 1], n, p=[0.05, 0.95] if not p else [0.6, 0.4]),
            'Links_pointing_to_page':  np.random.choice([-1, 0, 1], n, p=[0.1, 0.2, 0.7] if not p else [0.5, 0.3, 0.2]),
            'Statistical_report':      np.random.choice([-1, 1], n, p=[0.98, 0.02] if not p else [0.3, 0.7]),
            'Result':                  [1] * n if p else [0] * n
        }

    df_phish = pd.DataFrame(make_samples(n_phish, True))
    df_legit = pd.DataFrame(make_samples(n_legit, False))
    df = pd.concat([df_phish, df_legit], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"[INFO] Synthetic dataset created: {len(df)} samples, {df['Result'].sum()} phishing, {(df['Result']==0).sum()} legitimate")
    return df


# =============================================================
# 2. PREPROCESSING
# =============================================================
def preprocess(df):
    target_col = 'Result'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (important for LR, SVM, DNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"[INFO] Features: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


# =============================================================
# 3. MODELS
# =============================================================

# ── 3a. Logistic Regression ───────────────────────────────
def train_logistic_regression(X_train, y_train):
    print("\n[MODEL] Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


# ── 3b. Random Forest ────────────────────────────────────
def train_random_forest(X_train, y_train):
    print("[MODEL] Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


# ── 3c. Gradient Boosting ────────────────────────────────
def train_gradient_boosting(X_train, y_train):
    print("[MODEL] Training Gradient Boosting...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                       max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model


# ── 3d. SVM ───────────────────────────────────────────────
def train_svm(X_train, y_train):
    print("[MODEL] Training SVM...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    model.fit(X_train, y_train)
    return model


# ── 3e. Deep Neural Network ───────────────────────────────
def train_dnn(X_train, y_train, input_dim):
    print("[MODEL] Training Deep Neural Network...")
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_split=0.1, verbose=0)
    return model


# =============================================================
# 4. EVALUATION
# =============================================================
def evaluate_model(name, model, X_test, y_test, is_dnn=False):
    if is_dnn:
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Phishing'])}")

    return {'Model': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1 Score': f1, 'CM': cm}


# =============================================================
# 5. PLOTTING
# =============================================================
def plot_comparison(results):
    models = [r['Model'] for r in results]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#378ADD', '#1D9E75', '#7F77DD', '#D85A30']

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F8F8F8')

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [r[metric] for r in results]
        bars = ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, color='#333')

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison — Phishing Website Detection', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[PLOT] Saved → {path}")


def plot_confusion_matrices(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.patch.set_facecolor('white')

    for ax, r in zip(axes, results):
        cm = r['CM']
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(r['Model'], fontsize=9, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('Actual', fontsize=8)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Legit', 'Phish'], fontsize=8)
        ax.set_yticklabels(['Legit', 'Phish'], fontsize=8)
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black', fontsize=11)

    plt.suptitle('Confusion Matrices', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved → {path}")


def plot_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F8F8F8')

    bars = ax.bar(range(15), importances[idx], color='#378ADD', alpha=0.85)
    ax.set_xticks(range(15))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('Importance')
    ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved → {path}")


# =============================================================
# 6. SAVE RESULTS TABLE
# =============================================================
def save_results_table(results):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'CM'} for r in results])
    df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    path = os.path.join(OUTPUT_DIR, "results.csv")
    df.to_csv(path, index=False)
    print(f"\n[RESULTS] Saved → {path}")
    print("\n" + df.to_string(index=False))
    return df


# =============================================================
# 7. MAIN
# =============================================================
def main():
    print("\n" + "="*60)
    print("  PHISHING WEBSITE DETECTION — CLASS PROJECT")
    print("="*60)

    # Load data
    df = load_dataset()

    # Preprocess
    X_train, X_test, y_train, y_test, X_train_s, X_test_s = preprocess(df)
    feature_names = list(df.drop(columns=['Result']).columns)
    input_dim = X_train_s.shape[1]

    # Train all 5 models
    lr_model  = train_logistic_regression(X_train_s, y_train)
    rf_model  = train_random_forest(X_train, y_train)
    gb_model  = train_gradient_boosting(X_train, y_train)
    svm_model = train_svm(X_train_s, y_train)
    dnn_model = train_dnn(X_train_s, y_train, input_dim)

    # Evaluate all models
    print("\n\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)

    results = []
    results.append(evaluate_model("Logistic Regression", lr_model,  X_test_s, y_test))
    results.append(evaluate_model("Random Forest",       rf_model,  X_test,   y_test))
    results.append(evaluate_model("Gradient Boosting",  gb_model,  X_test,   y_test))
    results.append(evaluate_model("SVM",                svm_model, X_test_s, y_test))
    results.append(evaluate_model("Deep Neural Network",dnn_model, X_test_s, y_test, is_dnn=True))

    # Save results & plots
    save_results_table(results)
    plot_comparison(results)
    plot_confusion_matrices(results)
    plot_feature_importance(rf_model, feature_names)

    print("\n" + "="*60)
    print("  ALL DONE! Check the outputs/ folder for results.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
