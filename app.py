import streamlit as st
import pickle
import numpy as np
from PIL import Image

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Embedding,
    Dropout,
    Reshape,
    add,
    concatenate
)

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Image Caption Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SESSION STATE
# =====================================================

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #060d1a;
        color: #f8fafc;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* ---- HERO ---- */

    .hero-wrap {
        text-align: center;
        padding: 4rem 1rem 2rem;
    }

    .hero-eyebrow {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #a5b4fc;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 999px;
        padding: 0.3rem 1rem;
        margin-bottom: 1.4rem;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.4rem, 5vw, 4rem);
        font-weight: 800;
        line-height: 1.1;
        color: #f8fafc;
        margin-bottom: 1rem;
    }

    .hero-title span {
        background: linear-gradient(90deg, #818cf8, #6366f1, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-sub {
        font-size: 1.05rem;
        font-weight: 400;
        color: #94a3b8;
        max-width: 560px;
        margin: 0 auto 2.6rem;
        line-height: 1.7;
    }

    /* ---- CTA BUTTON ---- */

    .cta-btn-wrap {
        display: flex;
        justify-content: center;
        margin-bottom: 4rem;
    }

    /* ---- SECTION LABEL ---- */

    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #6366f1;
        margin-bottom: 0.6rem;
    }

    .section-heading {
        font-family: 'Syne', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }

    .section-sub {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 2.2rem;
    }

    /* ---- USE CASE CARDS ---- */

    .uc-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;

        padding: 1.6rem;
        height: 305px;

        display: flex;
        flex-direction: column;

        box-sizing: border-box;

        transition: transform 0.22s ease,
                    border-color 0.22s ease,
                    box-shadow 0.22s ease;
    }

    .uc-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99,102,241,0.45);
        box-shadow: 0 0 28px rgba(99,102,241,0.12);
    }

    .uc-icon {
        font-size: 2.3rem;
        margin-bottom: 1rem;
        display: block;
    }

    .uc-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.8rem;
    }

    .uc-desc {
        font-size: 0.95rem;
        color: #cbd5e1;
        line-height: 1.7;
        flex-grow: 1;
    }

    /* Make Streamlit columns equal height */

    div[data-testid="column"] {
        display: flex;
    }

    div[data-testid="column"] > div {
        width: 100%;
    }

    /* ---- DIVIDER ---- */

    .soft-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 2rem 0 2.5rem;
    }

    /* ---- UPLOADER SECTION HEADING ---- */

    .upload-heading {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.3rem;
    }

    .upload-sub {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    /* ---- CAPTION BOX ---- */

    .caption-box {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 400;
        color: #e2e8f0;
        min-height: 56px;
        display: flex;
        align-items: center;
        justify-content: center;
        line-height: 1.5;
    }

    /* ---- IMAGE HEIGHT ---- */

    [data-testid="stImage"] img {
        height: 225px !important;
        object-fit: cover;
        border-radius: 14px;
    }

    /* ---- Streamlit button override ---- */

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #6366f1, #818cf8);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2.2rem;
        font-size: 1rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        cursor: pointer;
        transition: opacity 0.2s ease, transform 0.2s ease;
        box-shadow: 0 4px 20px rgba(99,102,241,0.35);
    }

    div[data-testid="stButton"] > button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 6px 28px rgba(99,102,241,0.5);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("## Model Information")
    st.markdown(
        """
        - **CNN:** DenseNet201
        - **Sequence Model:** LSTM
        - **Dataset:** Flickr30k
        - **Framework:** TensorFlow + Streamlit
        """
    )
    st.markdown("---")
    st.markdown(
        """
        ## Team Members
        - Manan Jain : 23/EC/120
        - Manya Sharma : 23/EC/126
        """
    )

# =====================================================
# HERO SECTION
# =====================================================

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-eyebrow">✦ DenseNet201 + LSTM · Flickr30k</div>
        <div class="hero-title">
            Your images,<br><span>now with words.</span>
        </div>
        <div class="hero-sub">
            An AI that reads your images and writes what it sees —
            powered by a CNN-LSTM architecture trained on 30,000 real-world photos.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# USE CASES SECTION
# =====================================================

st.markdown(
    """
    <div class="section-label">Real-world applications</div>
    <div class="section-heading">Where caption AI makes a difference</div>
    <div class="section-sub">From accessibility tools to search engines — image captioning is everywhere.</div>
    """,
    unsafe_allow_html=True
)

use_cases = [
    ("♿", "Accessibility",         "Helps visually impaired users understand image content through screen readers and audio descriptions."),
    ("🔍", "Visual Search",          "Powers reverse image search and product discovery in e-commerce platforms like Amazon and Pinterest."),
    ("📸", "Photo Management",       "Automatically tags and organises personal photo libraries so you can search 'A Day in Moscow' and find it instantly."),
    ("🏥", "Medical Imaging",        "Assists radiologists by generating preliminary descriptions of X-rays, MRIs, and pathology slides."),
    ("🚗", "Autonomous Vehicles",    "Provides scene understanding for self-driving systems to detect pedestrians, signs, and road conditions."),
    ("📱", "Social Media",           "Auto-generates alt-text for uploaded images, improving inclusivity across platforms like Instagram and Twitter."),
    ("🎓", "Education",              "Turns images in textbooks into descriptive text, supporting students with learning difficulties."),
    ("🛡️", "Content Moderation",     "Flags potentially harmful or policy-violating images before human review — faster and at scale."),
    ("🌍", "Journalism & Archives",  "Helps newsrooms and libraries describe historical photographs for searchable digital archives."),
]

# Render cards using native st.columns (3 per row)
rows = [use_cases[i:i+3] for i in range(0, len(use_cases), 3)]
for row in rows:
    cols = st.columns(3, gap="medium")
    for col, (icon, title, desc) in zip(cols, row):
        with col:
            st.markdown(
                f"""
                <div class="uc-card">
                    <span class="uc-icon">{icon}</span>
                    <div class="uc-title">{title}</div>
                    <div class="uc-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

# =====================================================
# CTA BUTTON
# =====================================================

col_l, col_c, col_r = st.columns([2, 1.5, 2])
with col_c:
    if st.button("✦  Try it now!"):
        st.session_state.show_uploader = True

# =====================================================
# DIVIDER
# =====================================================

if st.session_state.show_uploader:
    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # =====================================================
    # LOAD MODEL
    # =====================================================

    @st.cache_resource
    def load_caption_model():
        with open("model/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        vocab_size = len(tokenizer.word_index) + 1
        max_length = 74

        input1 = Input(shape=(1920,))
        input2 = Input(shape=(max_length,))

        img_features         = Dense(256, activation='relu')(input1)
        img_features_reshaped = Reshape((1, 256))(img_features)
        sentence_features    = Embedding(vocab_size, 256, mask_zero=False)(input2)
        merged               = concatenate([img_features_reshaped, sentence_features], axis=1)
        sentence_features    = LSTM(256)(merged)
        x                    = Dropout(0.5)(sentence_features)
        x                    = add([x, img_features])
        x                    = Dense(128, activation='relu')(x)
        x                    = Dropout(0.5)(x)
        output               = Dense(vocab_size, activation='softmax')(x)

        caption_model = Model(inputs=[input1, input2], outputs=output)
        caption_model.load_weights("model/weights.h5")

        return caption_model, tokenizer, max_length

    @st.cache_resource
    def load_feature_extractor():
        base_model = DenseNet201()
        return Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    caption_model, tokenizer, max_length = load_caption_model()
    feature_extractor = load_feature_extractor()

    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}

    def predict_caption_fast(model, image_features):
        in_text = "startseq"
        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat     = model.predict([image_features, sequence], verbose=0)
            yhat     = np.argmax(yhat)
            word     = index_word.get(yhat)
            if word is None or word == "endseq":
                break
            in_text += " " + word
        return in_text.replace("startseq", "").strip()

    # =====================================================
    # UPLOAD + GRID
    # =====================================================

    st.markdown(
        """
        <div class="upload-heading">Generate captions</div>
        <div class="upload-sub">Upload up to 4 images — JPG, PNG, or WebP.</div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        uploaded_files = uploaded_files[:4]
        cols = st.columns(2)

        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % 2]:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, use_container_width=True)

                img = image.resize((224, 224))
                img = img_to_array(img)
                img = preprocess_input(img)
                img = np.expand_dims(img, axis=0)

                with st.spinner("Generating caption..."):
                    features = feature_extractor.predict(img, verbose=0)
                    caption  = predict_caption_fast(caption_model, features)

                st.markdown(
                    f'<div class="caption-box">{caption}</div>',
                    unsafe_allow_html=True
                )