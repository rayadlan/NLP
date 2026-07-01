

import numpy as np
import matplotlib
matplotlib.use("Agg")                 
import matplotlib.pyplot as plt
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from faq_data import build_dataset, ANSWERS, STEMMER_NAME, XGBStringWrapper

try:
    import xgboost  
    HAS_XGB = True
except Exception:
    HAS_XGB = False

MODEL_PATH = "binus_chatbot.joblib"
NAVY, AMBER, STEEL, GREEN = "#15264F", "#F2A900", "#3E5C9A", "#1E8E5A"


def build_models():
    """Kembalikan dict {nama: estimator} berisi 4 model fresh."""
    models = {
        "Naive Bayes":   MultinomialNB(),
        "Linear SVM":    CalibratedClassifierCV(LinearSVC(), cv=3),  
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBStringWrapper(
            n_estimators=300, max_depth=4, learning_rate=0.3, random_state=42)
    else:
        models["Logistic Regression"] = LogisticRegression(max_iter=1000)
    return models


def plot_comparison(results, path):
    names = list(results)
    acc = [results[n]["acc"] for n in names]
    cv = [results[n]["cv"] for n in names]
    x = np.arange(len(names)); w = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, acc, w, label="Akurasi Uji", color=NAVY)
    b2 = ax.bar(x + w/2, cv, w, label="CV 5-fold", color=AMBER)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor")
    ax.set_title("Perbandingan 4 Model — Intent Classification FAQ Binus",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.015,
                    f"{r.get_height():.2f}", ha="center", fontsize=9, color="#333")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(y_true, y_pred, labels, best_name, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name} (data uji)",
                 fontsize=13, fontweight="bold", color=NAVY, pad=14)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thr else "#15264F", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    X_raw, y = build_dataset()
    y = np.array(y)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(X_raw)
    labels = sorted(set(y))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    print("=" * 66)
    print("  LATIH & BANDINGKAN 4 MODEL — FAQ Binus")
    print(f"  Data: {X.shape[0]} contoh | {len(labels)} kategori | "
          f"Fitur TF-IDF: {X.shape[1]} | Stemmer: {STEMMER_NAME}")
    print("=" * 66)
    print(f"{'Model':<22}{'Akurasi Uji':>14}{'CV (5-fold)':>16}")
    print("-" * 66)

    results = {}
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        cv = cross_val_score(build_models()[name], X, y, cv=cv5).mean()
        results[name] = {"acc": acc, "cv": cv}
        print(f"{name:<22}{acc:>13.3f}{cv:>16.3f}")
    print("-" * 66)

    best = max(results, key=lambda k: (results[k]["cv"], results[k]["acc"]))
    print(f"  Model terbaik (CV): {best}\n")

    # confusion matrix + report untuk model terbaik
    best_est = build_models()[best].fit(X_tr, y_tr)
    y_pred = best_est.predict(X_te)
    print(f"  Classification report — {best}:")
    print("    " + classification_report(y_te, y_pred, zero_division=0).replace("\n", "\n    "))

    plot_comparison(results, "comparison_models.png")
    plot_confusion(y_te, y_pred, labels, best, "confusion_matrix.png")
    print("  Grafik disimpan: comparison_models.png, confusion_matrix.png")

   
    trained = {name: build_models()[name].fit(X, y) for name in results}
    bundle = {
        "vectorizer": vectorizer,
        "models": trained,
        "X_train": X,                
        "labels": labels,
        "best": best,
        "scores": results,
        "answers": ANSWERS,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"  Model disimpan: {MODEL_PATH}\n  Selesai.")


if __name__ == "__main__":
    main()
