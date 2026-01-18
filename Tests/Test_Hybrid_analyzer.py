"""
hybrid_test.py - Tests for Hybrid Analyzer

Tests the integration of DistilBERT + XAI Engine

Run with: python hybrid_test.py
"""

from Integration.hybrid_analyzer import HybridAnalyzer


# ============================================================================
# SETUP
# ============================================================================

print("="*70)
print("INITIALIZING HYBRID ANALYZER")
print("="*70)
print("Loading models (this may take a minute)...\n")

analyzer = HybridAnalyzer()

print("\n✅ Analyzer ready!\n")


# ============================================================================
# TEST CASES
# ============================================================================

def test_obvious_fake():
    """Test on obviously fake review"""
    print("="*70)
    print("TEST 1: Obvious Fake Review")
    print("="*70)
    
    review = "AMAZING!!! BEST PRODUCT EVER!!! I LOVE IT SO MUCH!!!"
    result = analyzer.analyze(review)
    
    # Verify DistilBERT prediction
    assert result['prediction'] == 'FAKE', \
        f"Expected FAKE, got {result['prediction']}"
    
    assert result['confidence'] > 0.5, \
        f"Expected confidence > 50%, got {result['confidence']:.1%}"
    
    # Verify XAI explanation exists
    assert len(result['explanation']['reasons']) > 0, \
        "No explanation reasons generated"
    
    # Display result
    print(analyzer.format_result(result))
    
    print("\n✅ Test passed: Correctly identified as FAKE")
    return True


def test_obvious_genuine():
    """Test on obviously genuine review"""
    print("\n" + "="*70)
    print("TEST 2: Obvious Genuine Review")
    print("="*70)
    
    review = "Bought this laptop 3 weeks ago for work. Battery lasts about 6 hours with normal use. Keyboard is comfortable for typing. Screen could be brighter for outdoor use. Overall good value for the price."
    result = analyzer.analyze(review)
    
    # Verify DistilBERT prediction
    assert result['prediction'] == 'REAL', \
        f"Expected REAL, got {result['prediction']}"
    
    assert result['confidence'] > 0.5, \
        f"Expected confidence > 50%, got {result['confidence']:.1%}"
    
    # Display result
    print(analyzer.format_result(result))
    
    print("\n✅ Test passed: Correctly identified as GENUINE")
    return True


def test_result_structure():
    """Test that result has correct structure"""
    print("\n" + "="*70)
    print("TEST 3: Result Structure")
    print("="*70)
    
    review = "Good product works well"
    result = analyzer.analyze(review)
    
    # Check required keys
    required_keys = [
        'review_text', 'prediction', 'confidence', 'model_label',
        'probabilities', 'explanation', 'agreement'
    ]
    
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    
    # Check explanation structure
    explanation_keys = ['verdict', 'xai_confidence', 'reasons', 'features', 'flag_count']
    for key in explanation_keys:
        assert key in result['explanation'], f"Missing explanation key: {key}"
    
    print("\nResult Structure:")
    print(f"  review_text: {type(result['review_text'])}")
    print(f"  prediction: {result['prediction']}")
    print(f"  confidence: {result['confidence']:.2%}")
    print(f"  model_label: {result['model_label']}")
    print(f"  probabilities: {result['probabilities']}")
    print(f"  explanation: {len(result['explanation'])} keys")
    print(f"  agreement: {result['agreement']}")
    
    print("\n Test passed: Result structure is correct")
    return True


def test_distilbert_prediction():
    """Test DistilBERT prediction separately"""
    print("\n" + "="*70)
    print("TEST 4: DistilBERT Prediction")
    print("="*70)
    
    review = "This is a test review"
    result = analyzer.predict_distilbert(review)
    
    # Check result structure
    assert 'label' in result, "Missing label"
    assert 'prediction' in result, "Missing prediction"
    assert 'confidence' in result, "Missing confidence"
    assert 'probabilities' in result, "Missing probabilities"
    
    # Check label values
    assert result['label'] in ['CG', 'OR'], \
        f"Invalid label: {result['label']}"
    
    assert result['prediction'] in ['FAKE', 'REAL'], \
        f"Invalid prediction: {result['prediction']}"
    
    # Check confidence range
    assert 0 <= result['confidence'] <= 1, \
        f"Confidence out of range: {result['confidence']}"
    
    # Check probabilities sum to 1
    prob_sum = sum(result['probabilities'])
    assert 0.99 <= prob_sum <= 1.01, \
        f"Probabilities don't sum to 1: {prob_sum}"
    
    print("\nDistilBERT Prediction:")
    print(f"  Label: {result['label']}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities: FAKE={result['probabilities'][0]:.2%}, REAL={result['probabilities'][1]:.2%}")
    
    print("\nTest passed: DistilBERT prediction works correctly")
    return True


def test_agreement_check():
    """Test agreement between DistilBERT and XAI"""
    print("\n" + "="*70)
    print("TEST 5: Agreement Check")
    print("="*70)
    
    # Test case where they should agree (obvious fake)
    fake_review = "BEST EVER!!! AMAZING!!! PERFECT!!!"
    result1 = analyzer.analyze(fake_review)
    
    print(f"\nObvious Fake Review:")
    print(f"  DistilBERT: {result1['prediction']}")
    print(f"  XAI: {result1['explanation']['verdict']}")
    print(f"  Agreement: {result1['agreement']}")
    
    # Test case where they might disagree (borderline)
    borderline_review = "Good product"
    result2 = analyzer.analyze(borderline_review)
    
    print(f"\nBorderline Review:")
    print(f"  DistilBERT: {result2['prediction']}")
    print(f"  XAI: {result2['explanation']['verdict']}")
    print(f"  Agreement: {result2['agreement']}")
    
    print("\nTest passed: Agreement check working")
    return True


def test_format_output():
    """Test formatted output"""
    print("\n" + "="*70)
    print("TEST 7: Format Output")
    print("="*70)
    
    review = "This product is amazing!"
    result = analyzer.analyze(review)
    formatted = analyzer.format_result(result)
    
    # Check that formatted output is a string
    assert isinstance(formatted, str), "Formatted output should be string"
    
    # Check that it contains key sections
    assert "DISTILBERT PREDICTION" in formatted, "Missing DistilBERT section"
    assert "XAI EXPLANATION" in formatted, "Missing XAI section"
    assert "Prediction:" in formatted, "Missing prediction"
    assert "Confidence:" in formatted, "Missing confidence"
    
    print("\nFormatted Output Preview:")
    print(formatted[:500] + "...")
    
    print("\nTest passed: Output formatting works")
    return True


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "="*70)
    print("TEST 8: Edge Cases")
    print("="*70)
    
    # Very short review
    short = "Good"
    result1 = analyzer.analyze(short)
    print(f"\nVery short review: {result1['prediction']} ({result1['confidence']:.1%})")
    
    # Very long review
    long = "Great product. " * 50
    result2 = analyzer.analyze(long)
    print(f"Very long review: {result2['prediction']} ({result2['confidence']:.1%})")
    
    # Review with special characters
    special = "Good! @#$ product :) 100%"
    result3 = analyzer.analyze(special)
    print(f"Special characters: {result3['prediction']} ({result3['confidence']:.1%})")
    
    # Empty-ish review
    minimal = "ok"
    result4 = analyzer.analyze(minimal)
    print(f"Minimal review: {result4['prediction']} ({result4['confidence']:.1%})")
    
    print("\n✅ Test passed: Edge cases handled")
    return True


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING HYBRID ANALYZER TESTS")
    print("="*70)
    
    tests = [
        test_obvious_fake,
        test_obvious_genuine,
        test_result_structure,
        test_distilbert_prediction,
        test_agreement_check,
        test_format_output,
        test_edge_cases
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\nFAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\nERROR: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n All hybrid analyzer tests passed!")
        print("Integration working correctly!")
    else:
        print(f"\n{failed} test(s) need attention")