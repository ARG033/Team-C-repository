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
from .feature_extractors import extract_all_features
from .reason_generator import generate_explanation, format_explanation_text
from .config import THRESHOLDS

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
        # Use the enhanced extract_all_features with tier 3 (advanced features)
        return extract_all_features(review_text, tier=3)
    
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
        