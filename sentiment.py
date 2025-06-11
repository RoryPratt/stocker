from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter

# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
model.eval()

def encode(label):
    mapping = {
        "positive": [1, 0, 0],
        "neutral":  [0, 1, 0],
        "negative": [0, 0, 1]
    }
    return mapping.get(label.lower(), [0, 0, 0])

def get_sentiment(text, max_tokens=512):
    """
    Efficient batched FinBERT sentiment using tokenizer.__call__.
    Returns one-hot vector + average confidence for top label.
    """
    # Tokenize the whole text first (no special tokens)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return [0, 1, 0, 0.0]  # neutral fallback

    chunks = [tokens[i:i + max_tokens - 2] for i in range(0, len(tokens), max_tokens - 2)]
    
    # Re-decode chunks into text
    chunk_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    # Efficient tokenization + padding
    encoded = tokenizer(
        chunk_texts,
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        labels = torch.argmax(probs, dim=1)
        scores = probs.max(dim=1).values

    label_names = [model.config.id2label[label.item()].lower() for label in labels]
    results = list(zip(label_names, scores.tolist()))

    # Majority vote with confidence average
    label_counts = Counter(label for label, _ in results)
    top_label = label_counts.most_common(1)[0][0]
    avg_score = sum(score for label, score in results if label == top_label) / label_counts[top_label]

    vector = encode(top_label)
    vector.append(avg_score)
    return vector
