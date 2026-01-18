"""
api_interface.py - Production-Ready API for Backend Integration

This is the ONLY file the backend team needs to import.
All complexity is hidden - simple, clean interface.

Usage Example (Backend):
    from integration.api_interface import ReviewAnalyzer
    
    analyzer = ReviewAnalyzer()
    result = analyzer.predict("Great product!")
    
    # result is JSON-ready dict
    print(result['prediction'])  # "FAKE" or "REAL"
    print(result['confidence'])  # 0.87
    print(result['explanation']) # Human-readable reasons
"""

from Integration.hybrid_analyzer import HybridAnalyzer


class ReviewAnalyzer:
    """
    Production-ready interface for fake review detection.
    
    This class provides a simple API for the backend team.
    All ML complexity is hidden inside.
    
    Example:
        analyzer = ReviewAnalyzer()
        result = analyzer.predict("This product is amazing!")
        
        # Use in your API
        return jsonify(result)
    """
    
    def __init__(self):
        """
        Initialize the review analyzer.
        
        Loads DistilBERT and XAI engine automatically.
        This happens once when the server starts.
        """
        
        print("Initializing Review Analyzer...")
        self._analyzer = HybridAnalyzer()
        print("Review Analyzer ready for predictions")
    
    
    def predict(self, review_text):
        """
        Analyze a single review for fake patterns.
        
        This is the MAIN method the backend will use.
        
        Args:
            review_text (str): The review text to analyze
            
        Returns:
            dict: {
                'success': bool,
                'prediction': 'FAKE' or 'REAL',
                'confidence': float (0.0-1.0),
                'explanation': str (human-readable),
                'details': {
                    'model_verdict': str,
                    'xai_verdict': str,
                    'flags': int,
                    'agreement': bool
                },
                'error': str (only if success=False)
            }
            
        Example:
            result = analyzer.predict("Great product!")
            
            if result['success']:
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.0%}")
                print(f"Why: {result['explanation']}")
        """
        # Validate input
        if not review_text:
            return {
                'success': False,
                'error': 'Review text cannot be empty',
                'prediction': None,
                'confidence': None,
                'explanation': None
            }
        
        if not isinstance(review_text, str):
            return {
                'success': False,
                'error': 'Review text must be a string',
                'prediction': None,
                'confidence': None,
                'explanation': None
            }
        
        # Strip whitespace
        review_text = review_text.strip()
        
        if len(review_text) == 0:
            return {
                'success': False,
                'error': 'Review text cannot be empty or whitespace only',
                'prediction': None,
                'confidence': None,
                'explanation': None
            }
        
        try:
            # Get analysis from hybrid analyzer
            analysis = self._analyzer.analyze(review_text)
            
            # Format explanation into single readable string
            explanation_text = self._format_explanation(analysis)
            
            # Return clean, JSON-ready response
            return {
                'success': True,
                'prediction': analysis['prediction'],
                'confidence': round(analysis['confidence'], 4),
                'explanation': explanation_text,
                'details': {
                    'model_verdict': analysis['prediction'],
                    'xai_verdict': analysis['explanation']['verdict'],
                    'flags': analysis['explanation']['flag_count'],
                    'agreement': analysis['agreement']
                }
            }
            
        except Exception as e:
            # Handle any errors 
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}',
                'prediction': None,
                'confidence': None,
                'explanation': None
            }
    
    
    def _format_explanation(self, analysis):
        """
        Convert XAI reasons into single readable string.
        
        Internal method - backend doesn't call this directly.
        
        Args:
            analysis (dict): Full analysis result from hybrid_analyzer
            
        Returns:
            str: Human-readable explanation
        """
        reasons = analysis['explanation']['reasons']
        
        if not reasons:
            return "No suspicious patterns detected."
        
        # Build explanation string
        explanation_parts = []
        
        for reason in reasons:
            # Skip the "Overall" reason (for genuine reviews)
            if reason['feature'] == 'Overall':
                continue
            
            # Format: "Feature: message"
            explanation_parts.append(f"{reason['feature']}: {reason['detail']}")
        
        if not explanation_parts:
            return "No suspicious patterns detected."
        
        # Join with semicolons
        return "; ".join(explanation_parts)
    
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Useful for API /info or /health endpoints.
        
        Returns:
            dict: {
                'model_name': str,
                'model_type': str,
                'features': list,
                'labels': dict
            }
            
        Example:
            info = analyzer.get_model_info()
            
            # Use in /info endpoint
            @app.get("/info")
            def info():
                return jsonify(analyzer.get_model_info())
        """
        return {
            'model_name': 'DistilBERT Fine-tuned for Fake Review Detection',
            'model_type': 'Hybrid (DistilBERT + Rule-based XAI)',
            'model_checkpoint': 'ARG33/DistilBERT-finetuned',
            'features': [
                'Sentiment Analysis',
                'Word Count',
                'Adjective-to-Noun Ratio',
                'First-Person Pronoun Usage',
                'Spam Keywords',
                'Excessive Capitalization',
                'Excessive Punctuation',
                'Text Redundancy'
            ],
            'labels': {
                'FAKE': 'Computer Generated (CG) - Likely fake review',
                'REAL': 'Original (OR) - Likely genuine review'
            },
            'output_format': {
                'prediction': 'FAKE or REAL',
                'confidence': 'float 0.0-1.0',
                'explanation': 'human-readable string',
                'details': 'additional metadata'
            }
        }
    
    
    def health_check(self):
        """
        Quick health check to verify analyzer is working.
        
        Useful for API /health endpoints.
        
        Returns:
            dict: {
                'status': 'healthy' or 'unhealthy',
                'model_loaded': bool,
                'test_prediction': dict (if healthy)
            }
            
        Example:
            @app.get("/health")
            def health():
                return jsonify(analyzer.health_check())
        """
        try:
            # Try a simple prediction
            test_result = self.predict("Test review")
            
            return {
                'status': 'healthy',
                'model_loaded': True,
                'test_prediction': test_result['success']
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model_loaded': False,
                'error': str(e)
            }
# ============================================================================
# USAGE EXAMPLES FOR BACKEND TEAM
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("API INTERFACE - Usage Examples for Backend Team")
    print("="*70)
    
    # Initialize analyzer (do this once when server starts)
    print("\n1. Initialize Analyzer")
    print("-" * 70)
    analyzer = ReviewAnalyzer()
    
    # Example 1: Analyze a review
    print("\n2. Analyze Single Review")
    print("-" * 70)
    result = analyzer.predict("AMAZING!!! Best product ever!!!")
    
    print(f"Success: {result['success']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Explanation: {result['explanation']}")
    print(f"Details: {result['details']}")
    
    # Example 2: Handle empty input
    print("\n3. Handle Invalid Input")
    print("-" * 70)
    result = analyzer.predict("")
    print(f"Success: {result['success']}")
    print(f"Error: {result['error']}")
    
    # Example 3: Get model info
    print("\n4. Get Model Information")
    print("-" * 70)
    info = analyzer.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Type: {info['model_type']}")
    print(f"Features: {len(info['features'])} features")
    
    # Example 4: Health check
    print("\n5. Health Check")
    print("-" * 70)
    health = analyzer.health_check()
    print(f"Status: {health['status']}")
    print(f"Model Loaded: {health['model_loaded']}")
    
    # Example 5: Genuine review
    print("\n6. Genuine Review Example")
    print("-" * 70)
    result = analyzer.predict("Bought this laptop last month. Battery lasts 6 hours. Good value.")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Agreement: {result['details']['agreement']}")
    
    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("="*70)