"""
Reason Generator - Converts features into human-readable explanations
"""

def generate_explanation(features, thresholds):
    """
    Generate human-readable explanation from extracted features.
    
    Args:
        features (dict): Extracted features from feature_extractors
        thresholds (dict): Threshold values from config
        
    Returns:
        dict: Contains verdict, confidence, and list of reasons
    """
    reasons = []
    confidence_score = 0
    
    # ===== TIER 1: ESSENTIAL FEATURES =====
    
    # 1. Sentiment Check
    sentiment = features['sentiment']
    if abs(sentiment) > thresholds['SENTIMENT_EXTREME']:
        if sentiment > 0:
            reasons.append({
                'feature': 'Sentiment',
                'message': f"Extremely positive sentiment (score: {sentiment:.2f}/1.0)",
                'detail': "Genuine reviews typically show more balanced emotions"
            })
        else:
            reasons.append({
                'feature': 'Sentiment',
                'message': f" Extremely negative sentiment (score: {sentiment:.2f}/1.0)",
                'detail': "Genuine reviews typically show more balanced emotions"
            })
        confidence_score += 25
    
    # 2. Word Count Check
    word_count = features['word_count']
    if word_count < thresholds['WORD_COUNT_MIN']:
        reasons.append({
            'feature': 'Length',
            'message': f" Suspiciously short review ({word_count} words)",
            'detail': "Genuine reviews typically provide more detail"
        })
        confidence_score += 15
    elif word_count >= thresholds['WORD_COUNT_MAX']:
        reasons.append({
            'feature': 'Length',
            'message': f" Unusually long review ({word_count} words)",
            'detail': "May contain padding or excessive fluff"
        })
        confidence_score += 10
    
    # 3. Adjective-to-Noun Ratio Check
    adj_noun_ratio = features['adj_noun_ratio']
    if adj_noun_ratio > thresholds['ADJ_NOUN_RATIO']:
        reasons.append({
            'feature': 'Specificity',
            'message': f" Excessive descriptive language (adj/noun ratio: {adj_noun_ratio:.1f}x)",
            'detail': f"Uses {features['adjective_count']} adjectives but only {features['noun_count']} nouns - lacks specific details"
        })
        confidence_score += 25
    
    # 4. First-Person Pronouns Check
    first_person_ratio = features['first_person_ratio']
    if first_person_ratio > thresholds['FIRST_PERSON_RATIO']:
        reasons.append({
            'feature': 'Self-Reference',
            'message': f" Excessive self-referencing ({first_person_ratio:.1%} of words)",
            'detail': f"Uses first-person pronouns {features['first_person_count']} times - appears overly personal"
        })
        confidence_score += 20
    
    # ===== TIER 2: IMPORTANT FEATURES =====
    
    # 5. Spam Keywords Check
    if features['spam_keyword_count'] >= thresholds['SPAM_KEYWORD_MIN']:
        reasons.append({
            'feature': 'Spam Language',
            'message': f" Contains {features['spam_keyword_count']} spam/promotional keywords",
            'detail': "High use of marketing language"
        })
        confidence_score += 15
    
    # 6. Excessive Caps Check
    if features['caps_ratio'] > thresholds['CAPS_RATIO_MAX']:
        reasons.append({
            'feature': 'Capitalization',
            'message': f" Excessive capitalization ({features['caps_ratio']:.1%} of words)",
            'detail': "Indicates emotional exaggeration"
        })
        confidence_score += 10
    
    # 7. Excessive Punctuation Check
    if features['excessive_punct_count'] >= thresholds['EXCESSIVE_PUNCT_MIN']:
        reasons.append({
            'feature': 'Punctuation',
            'message': f" Excessive punctuation patterns ({features['excessive_punct_count']} instances)",
            'detail': "Indicates emotional manipulation"
        })
        confidence_score += 10
    
    # 8. Text Redundancy Check
    if features['uniqueness_ratio'] < thresholds['UNIQUENESS_RATIO_MIN']:
        reasons.append({
            'feature': 'Redundancy',
            'message': f" High text redundancy ({features['uniqueness_ratio']:.1%} unique words)",
            'detail': "Repetitive language without substance"
        })
        confidence_score += 15
    
    # ===== DETERMINE VERDICT =====
    
    if confidence_score >= 50:
        verdict = "LIKELY FAKE"
        confidence = min(confidence_score, 95)  # Cap at 95%
    elif confidence_score >= 25:
        verdict = "SUSPICIOUS"
        confidence = confidence_score + 30
    else:
        verdict = "LIKELY GENUINE"
        confidence = 85
        reasons.append({
            'feature': 'Overall',
            'message': "No significant suspicious patterns detected",
            'detail': "Review appears to have natural linguistic characteristics"
        })
    
    return {
        'verdict': verdict,
        'confidence': confidence,
        'reasons': reasons,
        'flag_count': len([r for r in reasons if '' in r['message']])
    }
    

def format_explanation_text(analysis_result):
    """
    Format analysis result into readable text.
    
    Args:
        analysis_result (dict): Result from analyze_review() ==> from generate_explanation() ==> from class methods
        
    Returns:
        str: Formatted text explanation
    """
    lines = []
    lines.append("="*60)
    lines.append("FAKE REVIEW DETECTION - XAI ANALYSIS")
    lines.append("="*60)
    lines.append(f"\nReview: \"{analysis_result['review_text'][:100]}...\"")
    lines.append(f"\nVERDICT: {analysis_result['verdict']}")
    lines.append(f"CONFIDENCE: {analysis_result['confidence']}%")
    lines.append(f"\n{'SUSPICIOUS INDICATORS:' if analysis_result['flag_count'] > 0 else 'ANALYSIS:'}")
    lines.append("-"*60)
    
    for i, reason in enumerate(analysis_result['reasons'], 1):
        lines.append(f"\n{i}. {reason['feature']}")
        lines.append(f"   {reason['message']}")
        lines.append(f"   → {reason['detail']}")
    
    lines.append("\n" + "="*60)
    
    return "\n".join(lines)
    
