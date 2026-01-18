
"""
hybrid_analyzer.py - Bridge Between DistilBERT and XAI Engine

Combines:
1. DistilBERT prediction (FAKE/REAL with confidence)
2. XAI explanation (WHY it's fake/real)

Usage:
    analyzer = HybridAnalyzer()
    result = analyzer.analyze("Great product!")
    print(result['prediction'])  # "FAKE" or "REAL"
    print(result['explanation'])  # Human-readable reasons
"""

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from XAI_engine.xai_engine import FakeReviewXAI


class HybridAnalyzer:
    """
    Combines DistilBERT prediction with XAI explanation.
    
    Attributes:
        tokenizer: DistilBERT tokenizer
        model: Fine-tuned DistilBERT model
        xai_engine: XAI engine for explanations
        device: CPU or GPU
    """
    
    def __init__(self, model_path="ARG33/DistilBERT-finetuned", xai_engine=None):
        """
        Initialize the hybrid analyzer.
        
        Args:
            model_path: Hugging Face model checkpoint
            xai_engine: Optional pre-configured XAI engine
        """
        print("Loading DistilBERT model from Hugging Face Hub...")
        
        # Load tokenizer and model from Hugging Face
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        
        # Set device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Label mapping (from your training)
        self.id2label = {0: "CG", 1: "OR"}  # 0=Fake, 1=Genuine
        self.label2prediction = {"CG": "FAKE", "OR": "REAL"}
        
        # Initialize XAI engine
        self.xai_engine = xai_engine if xai_engine else FakeReviewXAI()
        
        print(f"Model loaded on {self.device}")
        print(f"XAI engine initialized")
        
    #================ DistilBERT Prediction Method =================#
    def predict_distilbert(self, review_text):
        """
        Get DistilBERT prediction.
        
        Args:
            review_text (str): Review to classify
            
        Returns:
            dict: {
                'label': 'CG' or 'OR',
                'prediction': 'FAKE' or 'REAL',
                'confidence': 0.0-1.0,
                'probabilities': [prob_fake, prob_real]
            }
        """
        # Tokenize (same as training - no preprocessing)
        inputs = self.tokenizer(
            review_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map to labels
        label = self.id2label[predicted_class]
        prediction = self.label2prediction[label]
        
        return {
            'label': label,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().tolist()
        }
    
    #================ Hybrid Analysis Method =================#
    def analyze(self, review_text):
        """
        Complete analysis: DistilBERT prediction + XAI explanation.
        
        Args:
            review_text (str): Review to analyze
            
        Returns:
            dict: {
                'review_text': str,
                'prediction': 'FAKE' or 'REAL',
                'confidence': float,
                'model_label': 'CG' or 'OR',
                'explanation': {
                    'verdict': str,
                    'reasons': list,
                    'features': dict
                },
                'agreement': bool  # Do DistilBERT and XAI agree?
            }
        """
        # Step 1: Get DistilBERT prediction
        distilbert_result = self.predict_distilbert(review_text)
        
        # Step 2: Get XAI explanation
        xai_result = self.xai_engine.analyze_review(review_text)
        
        # Step 3: Check agreement
        # DistilBERT says FAKE → XAI should say LIKELY FAKE or SUSPICIOUS
        # DistilBERT says REAL → XAI should say LIKELY GENUINE
        distilbert_says_fake = distilbert_result['prediction'] == 'FAKE'
        xai_says_fake = xai_result['verdict'] in ['LIKELY FAKE', 'SUSPICIOUS']
        agreement = (distilbert_says_fake == xai_says_fake)
        
        # Step 4: Combine results
        return {
            'review_text': review_text,
            
            # Primary prediction (from DistilBERT - most accurate)
            'prediction': distilbert_result['prediction'],
            'confidence': distilbert_result['confidence'],
            'model_label': distilbert_result['label'],
            'probabilities': distilbert_result['probabilities'],
            
            # Explanation (from XAI - most interpretable)
            'explanation': {
                'verdict': xai_result['verdict'],
                'xai_confidence': xai_result['confidence'],
                'reasons': xai_result['reasons'],
                'features': xai_result['features'],
                'flag_count': xai_result['flag_count']
            },
            
            # Agreement check
            'agreement': agreement
        }
        
    #================ Explanation Formatting Method =================#
    def format_result(self, analysis_result):
        """
        Format analysis result into readable text.
        
        Args:
            analysis_result (dict): Result from analyze()
            
        Returns:
            str: Formatted output
        """
        lines = []
        lines.append("="*70)
        lines.append("HYBRID ANALYSIS: DistilBERT + XAI")
        lines.append("="*70)
        
        # Review
        lines.append(f"\nReview: \"{analysis_result['review_text'][:100]}...\"")
        
        # DistilBERT Prediction
        lines.append("\n" + "-"*70)
        lines.append("DISTILBERT PREDICTION (Primary)")
        lines.append("-"*70)
        lines.append(f"Prediction:  {analysis_result['prediction']}")
        lines.append(f"Confidence:  {analysis_result['confidence']:.1%}")
        lines.append(f"Probabilities:")
        lines.append(f"  FAKE (CG): {analysis_result['probabilities'][0]:.1%}")
        lines.append(f"  REAL (OR): {analysis_result['probabilities'][1]:.1%}")
        
        # XAI Explanation
        lines.append("\n" + "-"*70)
        lines.append("XAI EXPLANATION (Why?)")
        lines.append("-"*70)
        lines.append(f"XAI Verdict:    {analysis_result['explanation']['verdict']}")
        lines.append(f"XAI Confidence: {analysis_result['explanation']['xai_confidence']}%")
        lines.append(f"Flags Detected: {analysis_result['explanation']['flag_count']}")
        
        if analysis_result['explanation']['reasons']:
            lines.append("\nReasons:")
            for i, reason in enumerate(analysis_result['explanation']['reasons'], 1):
                lines.append(f"\n  {i}. {reason['feature']}")
                lines.append(f"     {reason['message']}")
                lines.append(f"     → {reason['detail']}")
        
        # Agreement
        lines.append("\n" + "-"*70)
        if analysis_result['agreement']:
            lines.append(" DistilBERT and XAI agree on classification")
        else:
            lines.append("  DistilBERT and XAI disagree - may need manual review")
        
        lines.append("="*70)
        
        return "\n".join(lines)
    
