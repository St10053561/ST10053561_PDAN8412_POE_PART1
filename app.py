import streamlit as st
import tensorflow as tf
import joblib
import json
import numpy as np
import pandas as pd
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# Load Model + Encoders
# ---------------------------
@st.cache_resource
def load_model_and_artifacts():
    model = tf.keras.models.load_model("final_author_model.keras")
    label_encoder = joblib.load("label_encoder.pkl")
    with open("char_to_int.json", "r") as f:
        char_to_int = json.load(f)
    return model, label_encoder, char_to_int

model, label_encoder, char_to_int = load_model_and_artifacts()
MAX_LEN = 450

# ---------------------------
# Load large reference dataset (500k chunks)
# ---------------------------
@st.cache_data
def load_reference_data():
    return pd.read_csv("dataset_splits/train.csv")

reference_df = load_reference_data()

# ---------------------------
# Helper Functions
# ---------------------------
def encode_text(text, mapping):
    return [mapping.get(c, 0) for c in text]

def predict_author(text, top_k=3):
    seq = encode_text(text, char_to_int)
    seq = pad_sequences([seq], maxlen=MAX_LEN, padding='post')
    probs = model.predict(seq, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    preds = [(label_encoder.inverse_transform([i])[0], float(probs[i]))
             for i in top_indices]
    return preds, probs

def get_book_details(author, samples=3):
    """Get random book samples for given author"""
    try:
        rows = reference_df[reference_df['author'] == author].sample(samples)
        details = []
        for _, row in rows.iterrows():
            details.append({
                "Title": row.get('title', "N/A"),
                "Author": row.get('author', "N/A"),
                "Genre": row.get('genre', "N/A"),
                "Preview": row.get('chunk_text', "")[:300] + "..."
            })
        return details
    except:
        return None

import streamlit as st
import string

# ---------------------------
# Custom CSS for ChatGPT-like UI
# ---------------------------
st.markdown("""
<style>
.book-card {
    background: #f9f9f9;
    border-radius: 10px;
    padding: 12px;
    margin: 8px 0;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    border-left: 5px solid #4CAF50;
}
.book-card a {
    text-decoration: none;
    color: #2a7ae2;
    font-weight: bold;
    font-size: 15px;
}
.book-card a:hover {
    text-decoration: underline;
    color: #1b4f99;
}
.book-genre {
    font-size: 13px;
    color: #555;
}
.book-preview {
    font-size: 13px;
    margin-top: 5px;
    color: #333;
}
/* Assistant Messages (already styled by you) */
.chat-message.assistant {
    background: linear-gradient(145deg, #f1f1f1, #e6e6e6);
    border-radius: 10px;
    padding: 12px;
    margin: 8px 0;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    border-left: 5px solid #4CAF50;
}

/* User Messages — give them a nice blue bubble on the right */
.chat-message.user {
    background: linear-gradient(145deg, #e9f3ff, #d0e7ff);
    border-radius: 10px;
    padding: 12px;
    margin: 8px 0;
    margin-left: 20%; /* push it to the right side */
    box-shadow: 2px 2px 6px rgba(0,0,0,0.15);
    border-right: 5px solid #2196F3;
    text-align: right;
    color: #003366; /* darker text for contrast */
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Streamlit Chat UI
# ---------------------------
st.title("💬 Author Identification Chatbot")

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    greeting = (
        "👋 Hello there, welcome to the Author Identification Chatbot!\n\n"
        "Paste or type a passage, and I’ll try to guess which author wrote it.\n\n"
        "⚠️ Note: This model is experimental and predictions may not always be accurate."
    )
    st.session_state.messages.append({"role": "assistant", "content": greeting})

# Display past conversation
for message in st.session_state.messages:
    role, content = message["role"], message["content"]
    css_class = "assistant" if role == "assistant" else "user"
    st.markdown(f'<div class="chat-message {css_class}">{content}</div>', unsafe_allow_html=True)

# Chat input box
if prompt := st.chat_input("Type or paste a passage here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-message user">{prompt}</div>', unsafe_allow_html=True)

    with st.spinner("🤖 Thinking... analyzing your text..."):
        # Model prediction
        preds, probs = predict_author(prompt, top_k=3)
        top_author, top_prob = preds[0]

        # Assistant reply
        reply = (
            f"👉 I think this text most resembles **{top_author}** "
            f"({top_prob:.2%} confidence)."
        )
        st.markdown(f'<div class="chat-message assistant">{reply}</div>', unsafe_allow_html=True)

        # Text Stats
        st.info(
            f"📊 **Text Statistics:**\n\n"
            f"- Words: {len(prompt.split())}\n"
            f"- Characters: {len(prompt)}\n"
            f"- Punctuation: {sum(c in string.punctuation for c in prompt)}"
        )

        # Book recs - improved layout
        books = get_book_details(top_author, 2)
        if books:
            st.markdown(f"📚 **Works by {top_author} you might like:**")
            for b in books:
                title = b['Title']
                author = b['Author']
                genre = b['Genre']
                preview = b['Preview']
                google_link = f"https://www.google.com/search?q={title.replace(' ', '+')}+{author.replace(' ', '+')}"
                
                # Try to get Open Library cover by title and author (fallback to placeholder)
                import urllib.parse
                query = urllib.parse.quote(f"{title} {author}")
                cover_url = f"https://covers.openlibrary.org/b/olid/OLID-MISSING-M.png"  # default placeholder
                
                # Try to fetch Open Library search results for a cover
                import requests
                try:
                    resp = requests.get(f"https://openlibrary.org/search.json?title={urllib.parse.quote(title)}&author={urllib.parse.quote(author)}")
                    data = resp.json()
                    if data.get("docs"):
                        cover_id = data["docs"][0].get("cover_i")
                        if cover_id:
                            cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"
                except Exception:
                    pass  # fallback to placeholder
                
                st.markdown(
                    f"""
                    <div class="book-card">
                        <img src="{cover_url}" alt="cover" style="height:110px;float:right;margin-left:12px;border-radius:6px;">
                        <a href="{google_link}" target="_blank">📖 {title}</a>
                        <div class="book-genre">Genre: {genre}</div>
                        <div class="book-preview">"{preview[:150]}..."</div>
                        <div style="clear:both"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Disclaimer
        st.warning("⚠️ Predictions may not always be accurate — this model is still experimental.")

        # --- New Feature: Show all author probabilities as a bar chart ---
        author_names = label_encoder.classes_
        prob_percent = [probs[i]*100 for i in range(len(author_names))]
        prob_df = pd.DataFrame({
            "Author": author_names,
            "Confidence (%)": prob_percent
        }).sort_values("Confidence (%)", ascending=True)

        st.markdown("#### 🔎 Model Confidence for Each Author")
        st.bar_chart(
            data=prob_df.set_index("Author"),
            use_container_width=True
        )

        # Save assistant reply to history
        st.session_state.messages.append({"role": "assistant", "content": reply})
