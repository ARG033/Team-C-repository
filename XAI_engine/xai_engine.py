"""
XAI Engine - Main Orchestrator Class
=====================================
This class coordinates all XAI components to analyze reviews and generate explanations.

OOP Concepts Used:
- Class: Blueprint for creating objects
- __init__: Constructor that runs when you create an object
- self: Reference to the current object instance
- Methods: Functions that belong to a class
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from feature_extractors import (
    extract_sentiment,
    extract_word_count,
    extract_adjective_noun_ratio,
    extract_first_person_ratio,
    detect_spam_keywords,
    detect_excessive_caps,
    detect_excessive_punctuation,
    calculate_text_redundancy
)
from reason_generator import generate_explanation, format_explanation_text
from config import THRESHOLDS

# ============================================================================
# CLASS DEFINITION
# ============================================================================
# Think of a class as a BLUEPRINT for creating objects

class FakeReviewXAI:
    """
    Main XAI Engine that analyzes reviews for fake patterns.
    
    Think of this as a "Review Analyzer Machine":
    - You give it a review (text)
    - It extracts features (sentiment, length, etc.)
    - It checks against thresholds
    - It returns a verdict with explanations
    
    Example usage:
        xai = FakeReviewXAI()  # Create the machine
        result = xai.analyze_review("Great product!")  # Use it
        print(result['verdict'])  # See the result
    """
    
    # ========================================================================
    # __init__ METHOD (CONSTRUCTOR)
    # ========================================================================
    # Think of it as "setting up" or "initializing" the machine
    
    def __init__(self):
        """
        Constructor - Sets up the XAI engine.
        
        OOP EXPLANATION:
        ----------------
        When you write: xai = FakeReviewXAI()
        Python automatically calls this __init__ method
        
        'self' refers to the specific object being created
        Think of 'self' as "this specific machine"
        
        What happens here:
        1. Creates a sentiment analyzer (VADER tool)
        2. Loads threshold values from config
        3. Stores them as INSTANCE VARIABLES (belong to this object)
        """
        
        # INSTANCE VARIABLE: This machine's thresholds
        # We load from config so they can be easily changed later
        self.thresholds = THRESHOLDS.copy()
        
        # ========================================================================
    # METHODS (FUNCTIONS THAT BELONG TO THE CLASS)
    # ========================================================================
    # Methods are functions defined inside a class
    # They ALWAYS have 'self' as first parameter (refers to the object)
    
    def extract_all_features(self, review_text):
        """
        Extract all features from a review.
        
        OOP EXPLANATION:
        ----------------
        This is a METHOD (function inside a class)
        
        Parameters:
        - self: Reference to THIS object (automatically passed)
        - review_text: The actual parameter you provide
        
        Args:
            review_text (str): The review to analyze
            
        Returns:
            dict: Dictionary with all extracted features
        """
        features = {}
        
        # ===== TIER 1: ESSENTIAL FEATURES =====
        
        # Feature 1: Sentiment Analysis
        features['sentiment'] = extract_sentiment(review_text)
        
        # Feature 2: Word Count
        features['word_count'] = extract_word_count(review_text)
        
        # Feature 3: Adjective-to-Noun Ratio
        adj_noun_result = extract_adjective_noun_ratio(review_text)
        features['adj_noun_ratio'] = adj_noun_result['ratio']
        features['adjective_count'] = adj_noun_result['adjectives']
        features['noun_count'] = adj_noun_result['nouns']
        
        # Feature 4: First-Person Pronoun Usage
        first_person_result = extract_first_person_ratio(review_text)
        features['first_person_ratio'] = first_person_result['ratio']
        features['first_person_count'] = first_person_result['count']
        
        # ===== TIER 2: IMPORTANT FEATURES =====
        
        # Feature 5: Spam Keywords
        spam_result = detect_spam_keywords(review_text)
        features['spam_keyword_count'] = spam_result['count']
        features['spam_keywords_found'] = spam_result['found']
        
        # Feature 6: Excessive Capitalization
        caps_result = detect_excessive_caps(review_text)
        features['caps_ratio'] = caps_result['ratio']
        features['caps_count'] = caps_result['count']
        
        # Feature 7: Excessive Punctuation
        punct_result = detect_excessive_punctuation(review_text)
        features['excessive_punct_count'] = punct_result['total']
        
        # Feature 8: Text Redundancy
        redundancy_result = calculate_text_redundancy(review_text)
        features['uniqueness_ratio'] = redundancy_result['uniqueness_ratio']
        
        return features
    
    def analyze_review(self, review_text):
        """
        Complete analysis: extract features and generate explanation.
        
        Args:
            review_text (str): The review to analyze
            
        Returns:
            dict: Complete analysis with features, verdict, and explanation
        """
        # Extract all features
        features = self.extract_all_features(review_text)
        
        # Generate explanation using reason_generator
        explanation_data = generate_explanation(features, self.thresholds)
        
        # Combine everything
        return {
            'review_text': review_text,
            'features': features,
            'verdict': explanation_data['verdict'],
            'confidence': explanation_data['confidence'],
            'reasons': explanation_data['reasons'],
            'flag_count': explanation_data['flag_count']
        }
        
    def format_explanation(self, analysis_result):
        """
        Format analysis result into readable text.
        
        Args:
            analysis_result (dict): Result from analyze_review()
            
        Returns:
            str: Formatted text explanation
        """
        return format_explanation_text(analysis_result)   
        