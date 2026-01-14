"""
Feature Extractors for XAI Engine - Fake Review Detection
Team C - Advanced Models & XAI Engine

Tier 1: Essential Features (sentiment, word count, adj/noun ratio, first-person)
Tier 2: Important Features (spam keywords, caps, punctuation, redundancy)
"""

import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK data (run once):
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# Initialize VADER
sentiment_analyzer = SentimentIntensityAnalyzer()


# ============================================================================
# TIER 1: ESSENTIAL FEATURES (MUST IMPLEMENT)
# ============================================================================

def extract_sentiment(review_text):
    """
    Get sentiment score from -1.0 (negative) to +1.0 (positive).
    Uses VADER sentiment analysis.
    
    Threshold: > 0.85 or < -0.85 is suspicious
    """
    scores = sentiment_analyzer.polarity_scores(review_text)
    return scores['compound']


def extract_word_count(review_text):
    """
    Count total words in review.
    
    Threshold: < 15 words (too short) or > 200 words (too long)
    """
    return len(review_text.split())


def extract_adjective_noun_ratio(review_text):
    """
    Calculate ratio of adjectives to nouns.
    Fake reviews have too many adjectives without specific nouns.
    
    Threshold: > 2.5 is suspicious
    Returns: {'ratio': float, 'adjectives': int, 'nouns': int}
    """
    tokens = word_tokenize(review_text)
    pos_tags = pos_tag(tokens)
    
    # Count adjectives (JJ, JJR, JJS)
    adjectives = sum(1 for _, pos in pos_tags if pos.startswith('JJ'))
    
    # Count nouns (NN, NNS, NNP, NNPS)
    nouns = sum(1 for _, pos in pos_tags if pos.startswith('NN'))
    
    ratio = adjectives / max(nouns, 1)  # Avoid division by zero
    
    return {
        'ratio': ratio,
        'adjectives': adjectives,
        'nouns': nouns
    }


def extract_first_person_ratio(review_text):
    """
    Calculate ratio of first-person pronouns (I, me, my, etc.).
    Fake reviews overuse first-person to appear personal.
    
    Threshold: > 0.15 (15% of words) is suspicious
    Returns: {'ratio': float, 'count': int, 'total_words': int}
    """
    first_person = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
    
    words = review_text.lower().split()
    total = len(words)
    count = sum(1 for word in words if word in first_person)
    
    return {
        'ratio': count / max(total, 1),
        'count': count,
        'total_words': total
    }


# ============================================================================
# TIER 2: IMPORTANT FEATURES (SHOULD IMPLEMENT)
# ============================================================================

def detect_spam_keywords(review_text):
    """
    Detect spam/promotional keywords.
    
    Threshold: >= 3 keywords is suspicious
    Returns: {'found': list, 'count': int}
    """
    spam_keywords = {
        'extreme_positive': ['amazing', 'perfect', 'best ever', 'incredible', 'outstanding', 'flawless'],
        'extreme_negative': ['worst', 'terrible', 'horrible', 'awful', 'disgusting', 'pathetic'],
        'promotional': ['buy now', 'must have', 'life changing', 'miracle', 'highly recommend']
    }
    
    review_lower = review_text.lower()
    found = []
    
    for category, keywords in spam_keywords.items():
        for keyword in keywords:
            if keyword in review_lower:
                found.append({'keyword': keyword, 'category': category})
    
    return {'found': found, 'count': len(found)}


def detect_excessive_caps(review_text):
    """
    Detect excessive ALL CAPS words.
    
    Threshold: > 0.20 (20% of words) is suspicious
    Returns: {'ratio': float, 'count': int, 'words': list}
    """
    words = [word for word in review_text.split() if word.isalpha()]
    caps_words = [word for word in words if word.isupper() and len(word) > 1]
    
    return {
        'ratio': len(caps_words) / max(len(words), 1),
        'count': len(caps_words),
        'words': caps_words
    }


def detect_excessive_punctuation(review_text):
    """
    Detect repeated punctuation (!!!, ???, ...).
    
    Threshold: >= 3 patterns is suspicious
    Returns: {'exclamations': int, 'questions': int, 'ellipses': int, 'total': int}
    """
    exclamations = len(re.findall(r'!{2,}', review_text))
    questions = len(re.findall(r'\?{2,}', review_text))
    ellipses = len(re.findall(r'\.{3,}', review_text))
    
    return {
        'exclamations': exclamations,
        'questions': questions,
        'ellipses': ellipses,
        'total': exclamations + questions + ellipses
    }


def calculate_text_redundancy(review_text):
    """
    Calculate text redundancy (repetitive words).
    
    Threshold: < 0.60 uniqueness ratio is suspicious
    Returns: {'uniqueness_ratio': float, 'unique_words': int, 'total_words': int}
    """
    words = review_text.lower().split()
    unique_words = set(words)
    
    return {
        'uniqueness_ratio': len(unique_words) / max(len(words), 1),
        'unique_words': len(unique_words),
        'total_words': len(words)
    }


# ============================================================================
# MAIN FUNCTION: EXTRACT ALL FEATURES
# ============================================================================

def extract_all_features(review_text, tier=1):
    """
    Extract all features based on tier level.
    
    Args:
        review_text: The review to analyze
        tier: 1 (essential only), 2 (essential + important)
    
    Returns:
        Dictionary with all extracted features
    """
    features = {}
    
    # TIER 1: Always extract these
    features['sentiment'] = extract_sentiment(review_text)
    features['word_count'] = extract_word_count(review_text)
    
    adj_noun = extract_adjective_noun_ratio(review_text)
    features['adj_noun_ratio'] = adj_noun['ratio']
    features['adjective_count'] = adj_noun['adjectives']
    features['noun_count'] = adj_noun['nouns']
    
    first_person = extract_first_person_ratio(review_text)
    features['first_person_ratio'] = first_person['ratio']
    features['first_person_count'] = first_person['count']
    
    # TIER 2: Add if tier >= 2
    if tier >= 2:
        spam = detect_spam_keywords(review_text)
        features['spam_keywords_count'] = spam['count']
        
        caps = detect_excessive_caps(review_text)
        features['caps_ratio'] = caps['ratio']
        
        punct = detect_excessive_punctuation(review_text)
        features['excessive_punctuation'] = punct['total']
        
        redundancy = calculate_text_redundancy(review_text)
        features['uniqueness_ratio'] = redundancy['uniqueness_ratio']
    
    return features
        
# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING FEATURE EXTRACTORS")
    print("=" * 60)
    
    # Test 1: Fake-looking review
    fake = "This is AMAZING!!! Best EVER!!! I love it! Perfect!!!"
    print("\nTest 1: Suspicious Review")
    print(f"Review: {fake}")
    
    features = extract_all_features(fake, tier=2)
    print(features)
    
    # print(f"\nSentiment: {features['sentiment']:.3f}")
    # print(f"Word Count: {features['word_count']}")
    # print(f"Adj/Noun Ratio: {features['adj_noun_ratio']:.2f}")
    # print(f"First-Person: {features['first_person_ratio']:.2%}")
    # print(f"Spam Keywords: {features['spam_keywords_count']}")
    # print(f"ALL CAPS: {features['caps_ratio']:.2%}")
    
    # Test 2: Genuine-looking review
    genuine = "Bought this laptop 3 weeks ago for school. Battery lasts about 6 hours. Keyboard is comfortable but trackpad could be better. Good value overall."
    print("\n\nTest 2: Genuine Review")
    print(f"Review: {genuine}")
    
    features = extract_all_features(genuine, tier=2)
    print(features)
    
    # print(f"\nSentiment: {features['sentiment']:.3f}")
    # print(f"Word Count: {features['word_count']}")
    # print(f"Adj/Noun Ratio: {features['adj_noun_ratio']:.2f}")
    # print(f"First-Person: {features['first_person_ratio']:.2%}")
    # print(f"Uniqueness: {features['uniqueness_ratio']:.2%}")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)