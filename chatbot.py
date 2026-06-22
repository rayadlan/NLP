
import os
import sys
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from faq_data import preprocess
from train import MODEL_PATH

CONF_THRESHOLD = 0.20   
SIM_THRESHOLD = 0.18   
FALLBACK = ("Maaf, saya belum punya info soal itu. Coba tanya hal lain seputar Binus "
            "(mis. jurusan, biaya, pendaftaran, lokasi kampus).")


def load_bundle():
    """Muat model dari disk; latih dulu bila belum ada."""
    if not os.path.exists(MODEL_PATH):
        print(f"[info] '{MODEL_PATH}' belum ada — melatih model terlebih dahulu...\n")
        import train
        train.main()
        print()
    return joblib.load(MODEL_PATH)


class MLBot:
    def __init__(self, bundle):
        self.vec = bundle["vectorizer"]
        self.models = bundle["models"]
        self.X_train = bundle["X_train"]
        self.answers = bundle["answers"]
        self.active = bundle["best"]

    def set_model(self, name):
        match = next((m for m in self.models if m.lower() == name.lower()), None)
        if match:
            self.active = match
            return match
        return None

    def answer(self, text, debug=False):
        q = preprocess(text)
        qv = self.vec.transform([q])
        max_sim = float(cosine_similarity(qv, self.X_train).max())

        model = self.models[self.active]
        proba = model.predict_proba(qv)[0]
        idx = int(np.argmax(proba))
        intent = model.classes_[idx]
        conf = float(proba[idx])

        if debug:
            print(f"   [debug] model={self.active} | intent={intent} "
                  f"| conf={conf:.3f} | max_sim={max_sim:.3f}")

        if conf < CONF_THRESHOLD or max_sim < SIM_THRESHOLD:
            return FALLBACK
        return self.answers[intent]


def main():
    bundle = load_bundle()
    bot = MLBot(bundle)

    sc = bundle["scores"]
    print("=" * 64)
    print("  Chatbot FAQ Binus (ML — 4 model, model dimuat dari disk)")
    print("  Skor model (CV 5-fold):")
    for n in sc:
        print(f"    - {n:<20} acc={sc[n]['acc']:.3f}  cv={sc[n]['cv']:.3f}")
    print(f"  Model aktif: {bot.active}")
    print("  Perintah: 'model <nama>', 'debug on/off', 'keluar'")
    print("=" * 64)

    debug = False
    while True:
        try:
            text = input("\nAnda : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot  : Sampai jumpa!"); break
        if not text:
            continue
        low = text.lower()
        if low in ("keluar", "exit", "quit"):
            print("Bot  : Sampai jumpa!"); break
        if low == "debug on":
            debug = True; print("Bot  : Mode debug AKTIF."); continue
        if low == "debug off":
            debug = False; print("Bot  : Mode debug NONAKTIF."); continue
        if low.startswith("model "):
            m = bot.set_model(text[6:].strip())
            print(f"Bot  : Model aktif diganti ke '{m}'." if m
                  else f"Bot  : Model tidak dikenal. Pilihan: {', '.join(bot.models)}")
            continue
        print("Bot  :", bot.answer(text, debug=debug))


if __name__ == "__main__":
    main()
