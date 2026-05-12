import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from tensorflow import keras
from tensorflow.keras import layers

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# AMLAN PARIDA & SOHAM GHOSH PART
def load_dataset():
    path = "data/phishing.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)

    if 'Result' in df.columns:
        df['Result'] = df['Result'].map({-1: 1, 1: 0})

    print(f"[INFO] Dataset Loaded : {df.shape}")
    return df


# AMLAN PARIDA PART
def preprocess(df):
    X = df.drop(columns=['Result'])
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_s, X_test_s


# DESETTY GOURAV PART
def train_logistic_regression(X, y):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


# DESETTY GOURAV PART
def train_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model


# DESETTY GOURAV PART
def train_gradient_boosting(X, y):
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X, y)
    return model


# DESETTY GOURAV PART
def train_svm(X, y):
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X, y)
    return model


# DESETTY GOURAV PART
def train_dnn(X, y, input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X, y,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    return model


# MOHAMMAD IMRAN PART
def evaluate_model(name, model, X_test, y_test, is_dnn=False):

    if is_dnn:
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X_test)

    result = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'CM': confusion_matrix(y_test, y_pred)
    }

    print(f"\n{name}")
    print(classification_report(
        y_test,
        y_pred,
        target_names=['Legitimate', 'Phishing']
    ))

    return result


# MOHAMMAD IMRAN PART
def plot_comparison(results):

    models = [r['Model'] for r in results]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#378ADD', '#1D9E75', '#7F77DD', '#D85A30']

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]

        ax.bar(
            x + i * width,
            values,
            width,
            label=metric,
            color=colors[i]
        )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Model Comparison")
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "model_comparison.png"),
        dpi=150
    )

    plt.close()


# MOHAMMAD IMRAN PART
def plot_confusion_matrices(results):

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(4 * len(results), 4)
    )

    for ax, r in zip(axes, results):

        cm = r['CM']

        ax.imshow(cm, cmap='Blues')

        ax.set_title(r['Model'], fontsize=9)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(['Legit', 'Phish'])
        ax.set_yticklabels(['Legit', 'Phish'])

        for i in range(2):
            for j in range(2):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha='center',
                    va='center'
                )

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "confusion_matrices.png"),
        dpi=150
    )

    plt.close()


# SOHAM GHOSH PART
def plot_feature_importance(model, feature_names):

    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1][:15]

    plt.figure(figsize=(10, 5))

    plt.bar(
        range(15),
        importance[idx],
        color='#378ADD'
    )

    plt.xticks(
        range(15),
        [feature_names[i] for i in idx],
        rotation=40,
        ha='right'
    )

    plt.title("Top Feature Importances")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
        dpi=150
    )

    plt.close()


# ANIPS KUMAR JENA PART
def save_results(results):

    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'CM'}
        for r in results
    ])

    df = df.sort_values('Accuracy', ascending=False)

    df.to_csv(
        os.path.join(OUTPUT_DIR, "results.csv"),
        index=False
    )

    print(df)
# ANIPS KUMAR JENA PART
def main():

    print("\nPHISHING WEBSITE DETECTION USING MACHINE LEARNING\n")

    df = load_dataset()

    X_train, X_test, y_train, y_test, X_train_s, X_test_s = preprocess(df)

    feature_names = list(df.drop(columns=['Result']).columns)

    input_dim = X_train_s.shape[1]

    lr_model = train_logistic_regression(X_train_s, y_train)
    rf_model = train_random_forest(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)
    svm_model = train_svm(X_train_s, y_train)

    dnn_model = train_dnn(
        X_train_s,
        y_train,
        input_dim
    )

    results = [
        evaluate_model("Logistic Regression", lr_model, X_test_s, y_test),
        evaluate_model("Random Forest", rf_model, X_test, y_test),
        evaluate_model("Gradient Boosting", gb_model, X_test, y_test),
        evaluate_model("SVM", svm_model, X_test_s, y_test),
        evaluate_model("Deep Neural Network", dnn_model, X_test_s, y_test, is_dnn=True)
    ]
    save_results(results)
    plot_comparison(results)
    plot_confusion_matrices(results)
    plot_feature_importance(rf_model, feature_names)
    print("\n[INFO] All Results Saved Successfully")
if __name__ == "__main__":
    main()
