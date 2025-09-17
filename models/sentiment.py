from transformers import pipeline
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize

# ---------------------------
# Ensure NLTK tokenizer data is available
# ---------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------------------------
# Label mapping for Cardiff NLP RoBERTa
# ---------------------------
label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

# ---------------------------
# Load sentiment pipeline (cached for Streamlit)
# ---------------------------
@st.cache_resource
def load_sentiment():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1,  # CPU only
    )

sentiment_analyzer = load_sentiment()

# ---------------------------
# Helper: Adjust label based on confidence thresholds
# ---------------------------
def map_label_with_threshold(label, score, pos_thresh=0.55, neg_thresh=0.55):
    """
    Adjust sentiment based on confidence score.
    - If neutral but score > pos_thresh → positive
    - If neutral but score > neg_thresh → negative
    """
    label = label.lower()
    if label == "neutral":
        if score > pos_thresh:
            return "positive"
        elif score > neg_thresh:
            return "negative"
    return label

# ---------------------------
# Analyze sentiment for single string or list of strings
# ---------------------------
def analyze_sentiment(text):
    try:
        if isinstance(text, list):
            if not text:
                return []
            results = sentiment_analyzer(text)
            output = []
            for res in results:
                label = res.get("label", "neutral")
                score = float(res.get("score", 0.0))
                sentiment_label = map_label_with_threshold(label_map.get(label, "neutral"), score)
                output.append({"Sentiment": sentiment_label, "Score": round(score, 2)})
            return output
        else:
            res = sentiment_analyzer(text)[0]
            label = res.get("label", "neutral")
            score = float(res.get("score", 0.0))
            sentiment_label = map_label_with_threshold(label_map.get(label, "neutral"), score)
            return {"Sentiment": sentiment_label, "Score": round(score, 2)}
    except Exception:
        return {"Sentiment": "neutral", "Score": 0.0}

# ---------------------------
# Analyze long text by sentences with weighted aggregation
# ---------------------------
def analyze_long_text(text):
    if not text or not isinstance(text, str):
        return {"sentences": [], "overall": {"Sentiment": "neutral", "Score": 0.0}}

    # Tokenize into sentences and ignore very short ones
    sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 3]
    if not sentences:
        return {"sentences": [], "overall": {"Sentiment": "neutral", "Score": 0.0}}

    # Sentence-level sentiment
    sentence_results = analyze_sentiment(sentences)

    # Weighted aggregation
    weighted_scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    total_confidence = 0.0
    for res in sentence_results:
        weighted_scores[res["Sentiment"]] += res["Score"]
        total_confidence += res["Score"]

    overall_sentiment = max(weighted_scores, key=weighted_scores.get)
    avg_score = round(total_confidence / len(sentence_results), 2)

    return {"sentences": sentence_results, "overall": {"Sentiment": overall_sentiment, "Score": avg_score}}

# ---------------------------
# Optional standalone testing
# ---------------------------
if __name__ == "__main__":
    st.title("Sentiment App")
    user_input = st.text_area("Enter text to analyze its sentiment:")
    if user_input:
        results = analyze_long_text(user_input)
        st.write("### Sentence-level analysis")
        st.json(results["sentences"])
        st.write("### Overall sentiment")
        st.json(results["overall"])
