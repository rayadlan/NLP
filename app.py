import streamlit as st
from chatbot import load_bundle, MLBot

st.set_page_config(
    page_title="BINUS FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

@st.cache_resource
def load_bot():
    bundle = load_bundle()
    bot = MLBot(bundle)
    return bot, bundle

bot, bundle = load_bot()

st.title("🤖 BINUS FAQ Chatbot")

st.write(
    """
Chatbot ini menggunakan Machine Learning
(TF-IDF + Intent Classification).

Silakan tanyakan mengenai:

- Program Studi
- Biaya Kuliah
- Pendaftaran
- Lokasi Kampus
- Beasiswa
- Fasilitas
- Online Learning
- Kalender Akademik
"""
)

model = st.selectbox(
    "Pilih Model",
    list(bundle["models"].keys()),
    index=list(bundle["models"].keys()).index(bundle["best"])
)

bot.set_model(model)

question = st.text_input(
    "Masukkan pertanyaan",
    placeholder="Contoh: Berapa biaya kuliah Binus?"
)

if st.button("Kirim"):

    if question.strip():

        answer = bot.answer(question)

        st.success(answer)

st.divider()

st.subheader("Performa Model")

scores = bundle["scores"]

for name in scores:

    st.write(
        f"""
**{name}**

Accuracy : **{scores[name]['acc']:.3f}**

Cross Validation : **{scores[name]['cv']:.3f}**
"""
    )