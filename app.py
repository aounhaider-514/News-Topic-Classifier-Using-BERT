# app.py
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. Set up page configuration
st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="centered"
)

# 2. Title and description
st.title("📰 News Topic Classifier")
st.markdown("""
Classify news headlines into 4 categories using BERT:
- **World** 🌍
- **Sports** ⚽
- **Business** 💼
- **Science/Tech** 🔬
""")

# 3. Load model and tokenizer with caching
@st.cache_resource
def load_model():
    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('./bert_news_classifier')
    model = BertForSequenceClassification.from_pretrained('./bert_news_classifier')
    return tokenizer, model

tokenizer, model = load_model()

# 4. Class label mapping
class_names = {
    0: "World News 🌍",
    1: "Sports ⚽",
    2: "Business 💼",
    3: "Science/Tech 🔬"
}

# 5. User input section
st.subheader("Enter a News Headline")
headline = st.text_area("", "Apple launches new AI-powered chip", height=100)

# 6. Prediction function
def predict_topic(text):
    # Tokenize input
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process output
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).numpy()[0]
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class, probabilities

# 7. Make and display prediction
if st.button("Classify Topic"):
    if headline.strip() == "":
        st.warning("⚠️ Please enter a headline")
    else:
        with st.spinner("Analyzing headline..."):
            pred_class, probs = predict_topic(headline)
        
        # Display results
        st.success(f"Predicted Topic: **{class_names[pred_class]}**")
        
        # Show confidence scores
        st.subheader("Confidence Levels")
        for i, (class_id, class_label) in enumerate(class_names.items()):
            confidence = probs[i] * 100
            st.progress(int(confidence), text=f"{class_label}: {confidence:.1f}%")

# 8. Add explanation section
st.markdown("---")
st.subheader("How It Works")
st.markdown("""
1. **Input**: You enter a news headline
2. **Tokenization**: The text is split into BERT-compatible tokens
3. **BERT Processing**: The model analyzes word relationships contextually
4. **Classification**: Final layer predicts topic probabilities
5. **Output**: Results show predicted topic with confidence levels

**Technical Details**:
- Uses `bert-base-uncased` fine-tuned on AG News dataset
- Processes text in 64-token chunks
- Runs inference on CPU (no GPU needed for predictions)
""")

# 9. Add sample headlines
st.sidebar.markdown("## Sample Headlines")
sample_headlines = [
    "Microsoft announces new cloud computing partnership",
    "Olympic gold medalist breaks world record",
    "NASA discovers water on Mars surface",
    "Stock markets hit all-time high amid economic recovery"
]

for sample in sample_headlines:
    if st.sidebar.button(sample):
        headline = sample