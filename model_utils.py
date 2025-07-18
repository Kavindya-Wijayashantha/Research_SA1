import torch
import numpy as np
import re
import spacy
from transformers import BertTokenizer, BertModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.to(device)
        self.bert_model.eval()

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("Please install spaCy English model: python -m spacy download en_core_web_sm")

        # Initialize VADER
        self.sid = SentimentIntensityAnalyzer()

    def clean_text(self, text):
        """Clean and preprocess text data"""
        text = str(text)
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        return text.lower()

    def extract_bert_features(self, text):
        """Extract BERT embeddings"""
        cleaned_text = self.clean_text(text)
        inputs = self.tokenizer(cleaned_text, return_tensors="pt", padding=True,
                                truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return cls_embedding.flatten()

    def extract_lexical_features(self, text):
        """Extract VADER sentiment features"""
        cleaned_text = self.clean_text(text)
        scores = self.sid.polarity_scores(cleaned_text)
        return np.array([scores['pos'], scores['neu'], scores['neg'], scores['compound']])

    def extract_syntactic_features(self, text):
        """Extract syntactic features using spaCy"""
        cleaned_text = self.clean_text(text)
        doc = self.nlp(cleaned_text)

        negation = int(any(tok.dep_ == "neg" for tok in doc))
        num_nouns = sum(1 for tok in doc if tok.pos_ == "NOUN")
        num_verbs = sum(1 for tok in doc if tok.pos_ == "VERB")

        return np.array([negation, num_nouns, num_verbs])

    def extract_all_features(self, text):
        """Extract all three feature types for a single text"""
        bert_features = self.extract_bert_features(text)
        lexical_features = self.extract_lexical_features(text)
        syntactic_features = self.extract_syntactic_features(text)

        return bert_features, lexical_features, syntactic_features


def load_model(model_path, device='cpu'):
    """Load the trained model"""
    from models.model_architecture import FeatureAttentionFusionModel

    # Initialize model with same parameters as training
    model = FeatureAttentionFusionModel(
        bert_dim=768,
        lex_dim=4,
        syn_dim=3,
        hidden_dim=128,
        num_classes=2
    )

    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model
