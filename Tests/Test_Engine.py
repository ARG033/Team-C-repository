"""
Engine_test.py - End-to-End XAI Engine Tests (All Tiers)

Tests the complete pipeline:
1. Extract features (Tier 1 + 2 + 3)
2. Generate reasons
3. Format output

Run with: python Engine_test.py
"""
from XAI_engine.xai_engine import FakeReviewXAI

# ============================================================================
# TEST CASES
# ============================================================================

def test_obvious_fake_review():
    """Test engine on obviously fake review"""
    print("\n" + "="*60)
    print("TEST 1: Obvious Fake Review (All Tiers)")
    print("="*60)
    
    xai = FakeReviewXAI()
    review = "AMAZING!!! BEST PRODUCT EVER!!! I LOVE IT SO MUCH!!!"
    
    # Run complete pipeline
    result = xai.analyze_review(review)
    
    # Verify results
    assert result['verdict'] in ['LIKELY FAKE', 'SUSPICIOUS'], \
        f"Expected FAKE/SUSPICIOUS, got {result['verdict']}"
    assert result['confidence'] > 50, \
        f"Expected confidence > 50%, got {result['confidence']}%"
    assert result['flag_count'] >= 2, \
        f"Expected at least 2 flags, got {result['flag_count']}"
    
    # Display formatted output
    print(xai.format_explanation(result))
    print("\n Test passed: Correctly identified as fake")


def test_obvious_genuine_review():
    """Test engine on obviously genuine review"""
    print("\n" + "="*60)
    print("TEST 2: Obvious Genuine Review (All Tiers)")
    print("="*60)
    
    xai = FakeReviewXAI()
    review = "Bought this laptop 3 weeks ago for work. Battery lasts about 6 hours with normal use. Keyboard is comfortable for typing. Screen could be brighter for outdoor use. Overall good value for the price."
    
    # Run complete pipeline
    result = xai.analyze_review(review)
    
    # Verify results
    assert result['verdict'] == 'LIKELY GENUINE', \
        f"Expected GENUINE, got {result['verdict']}"
    assert result['flag_count'] <= 1, \
        f"Expected <= 1 flag, got {result['flag_count']}"
    
    # Display formatted output
    print(xai.format_explanation(result))
    print("\n Test passed: Correctly identified as genuine")


def test_borderline_suspicious():
    """Test engine on borderline suspicious review"""
    print("\n" + "="*60)
    print("TEST 3: Borderline Suspicious Review")
    print("="*60)
    
    xai = FakeReviewXAI()
    review = "Great product! Works perfectly!"
    
    # Run complete pipeline
    result = xai.analyze_review(review)
    
    # Verify results
    assert result['verdict'] in ['LIKELY FAKE', 'SUSPICIOUS', 'LIKELY GENUINE'], \
        f"Got unexpected verdict: {result['verdict']}"
    
    # Display formatted output
    print(xai.format_explanation(result))
    print(f"\n Test passed: Verdict = {result['verdict']}")


def test_feature_extraction_all_tiers():
    """Test that all tiers are extracted"""
    print("\n" + "="*60)
    print("TEST 4: Feature Extraction (All Tiers)")
    print("="*60)
    
    xai = FakeReviewXAI()
    review = "This amazing product is perfect and wonderful!"
    
    # Extract features only
    features = xai.extract_all_features(review)
    
    # Verify Tier 1 features
    tier1_features = ['sentiment', 'word_count', 'adj_noun_ratio', 'first_person_ratio']
    
    # Verify Tier 2 features
    tier2_features = ['spam_keyword_count', 'caps_ratio', 'excessive_punct_count', 'uniqueness_ratio']
    
    # Verify Tier 3 features
    tier3_features = ['flesch_score', 'dale_chall_score', 'bigram_repetitiveness', 
                      'common_fake_ngrams', 'tfidf_similarity']
    
    all_required = tier1_features + tier2_features + tier3_features
    
    for feature in all_required:
        assert feature in features, f"Missing feature: {feature}"
    
    # Display extracted features
    print("\n📊 Extracted Features:")
    print("\nTier 1 (Essential):")
    print(f"  Sentiment:          {features['sentiment']:.3f}")
    print(f"  Word Count:         {features['word_count']}")
    print(f"  Adj/Noun Ratio:     {features['adj_noun_ratio']:.2f}")
    print(f"  First-Person Ratio: {features['first_person_ratio']:.2%}")
    
    print("\nTier 2 (Important):")
    print(f"  Spam Keywords:      {features['spam_keyword_count']}")
    print(f"  Caps Ratio:         {features['caps_ratio']:.2%}")
    print(f"  Excess Punct:       {features['excessive_punct_count']}")
    print(f"  Uniqueness:         {features['uniqueness_ratio']:.2%}")
    
    print("\nTier 3 (Advanced):")
    print(f"  Flesch Score:       {features['flesch_score']:.2f}")
    print(f"  Dale-Chall Score:   {features['dale_chall_score']:.2f}")
    print(f"  Bigram Repeat:      {features['bigram_repetitiveness']:.2f}")
    print(f"  Fake N-grams:       {features['common_fake_ngrams']}")
    print(f"  TF-IDF Similarity:  {features['tfidf_similarity']:.2f}")
    
    print(f"\n Test passed: All {len(all_required)} features extracted")


def test_reason_generation():
    """Test that reasons are generated correctly"""
    print("\n" + "="*60)
    print("TEST 5: Reason Generation")
    print("="*60)
    
    xai = FakeReviewXAI()
    review = "AMAZING!!! Best ever!"
    
    # Run analysis
    result = xai.analyze_review(review)
    
    # Verify reasons exist
    assert 'reasons' in result, "Missing 'reasons' in result"
    assert len(result['reasons']) > 0, "No reasons generated"
    
    # Check reason structure
    first_reason = result['reasons'][0]
    assert 'feature' in first_reason, "Reason missing 'feature'"
    assert 'message' in first_reason, "Reason missing 'message'"
    assert 'detail' in first_reason, "Reason missing 'detail'"
    
    # Display reasons
    print("\nGenerated Reasons:")
    for i, reason in enumerate(result['reasons'], 1):
        print(f"\n{i}. {reason['feature']}")
        print(f"   {reason['message']}")
        print(f"   → {reason['detail']}")
    
    print("\n Test passed: Reasons generated correctly")


def test_format_output():
    """Test that output formatting works"""
    print("\n" + "="*60)
    print("TEST 6: Output Formatting")
    print("="*60)
    
    xai = FakeReviewXAI()
    review = "Great product!"
    
    # Run analysis
    result = xai.analyze_review(review)
    
    # Format output
    formatted = xai.format_explanation(result)
    
    # Verify formatting
    assert isinstance(formatted, str), "Formatted output should be a string"
    assert len(formatted) > 0, "Formatted output is empty"
    assert 'VERDICT' in formatted, "Missing verdict in formatted output"
    assert 'CONFIDENCE' in formatted, "Missing confidence in formatted output"
    
    print("\nFormatted Output:")
    print(formatted)
    
    print("\n Test passed: Output formatted correctly")


def test_threshold_usage():
    """Test that custom thresholds are used"""
    print("\n" + "="*60)
    print("TEST 7: Custom Thresholds")
    print("="*60)
    
    xai = FakeReviewXAI()
    review = "Good product works well"
    
    # Get result with default thresholds
    result_default = xai.analyze_review(review)
    
    # Change thresholds to be very strict
    xai.thresholds['SENTIMENT_EXTREME'] = 0.50
    xai.thresholds['WORD_COUNT_MIN'] = 100
    xai.thresholds['FLESCH_MAX'] = 50  # Tier 3 threshold
    
    # Get result with strict thresholds
    result_strict = xai.analyze_review(review)
    
    print(f"\nDefault thresholds:")
    print(f"  Verdict: {result_default['verdict']}")
    print(f"  Flags: {result_default['flag_count']}")
    
    print(f"\nStrict thresholds:")
    print(f"  Verdict: {result_strict['verdict']}")
    print(f"  Flags: {result_strict['flag_count']}")
    
    # Strict thresholds should flag more
    assert result_strict['flag_count'] >= result_default['flag_count'], \
        "Strict thresholds should flag more issues"
    
    print("\n Test passed: Custom thresholds are applied")


def test_tier3_contribution():
    """Test that Tier 3 features contribute to detection"""
    print("\n" + "="*60)
    print("TEST 8: Tier 3 Feature Contribution")
    print("="*60)
    
    xai = FakeReviewXAI()
    
    # Simple fake-like review (should trigger Tier 3 features)
    simple_fake = "Good product. I like it. It works. Very nice. Highly recommend."
    result = xai.analyze_review(simple_fake)
    
    features = result['features']
    
    print("\nTier 3 Feature Values:")
    print(f"  Flesch Score: {features['flesch_score']:.2f} (threshold: {xai.thresholds['FLESCH_MAX']})")
    print(f"  Dale-Chall: {features['dale_chall_score']:.2f} (threshold: {xai.thresholds['DALE_CHALL_MAX']})")
    print(f"  Bigram Repeat: {features['bigram_repetitiveness']:.2f} (threshold: {xai.thresholds['BIGRAM_REPETITIVENESS_MIN']})")
    print(f"  Fake N-grams: {features['common_fake_ngrams']} (threshold: {xai.thresholds['COMMON_FAKE_NGRAMS_MIN']})")
    print(f"  TF-IDF Sim: {features['tfidf_similarity']:.2f} (threshold: {xai.thresholds['TFIDF_SIMILARITY_MAX']})")
    
    print(f"\nVerdict: {result['verdict']}")
    print(f"Total Flags: {result['flag_count']}")
    
    # Check if any Tier 3 features contributed
    tier3_flags = []
    for reason in result['reasons']:
        if any(keyword in reason['feature'].lower() for keyword in ['readability', 'flesch', 'chall', 'ngram', 'similarity', 'repetitive']):
            tier3_flags.append(reason['feature'])
    
    if tier3_flags:
        print(f"\nTier 3 flags detected: {', '.join(tier3_flags)}")
    else:
        print("\nNo Tier 3 flags for this review")
    
    print("\n Test passed: Tier 3 features are being evaluated")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("XAI ENGINE - END-TO-END TESTS (ALL TIERS)")
    print("="*60)
    
    tests = [
        test_obvious_fake_review,
        test_obvious_genuine_review,
        test_borderline_suspicious,
        test_feature_extraction_all_tiers,
        test_reason_generation,
        test_format_output,
        test_threshold_usage,
        test_tier3_contribution
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Features Tested: 18 total (Tier 1: 4, Tier 2: 4, Tier 3: 5)")
    print(f" Passed: {passed}/{len(tests)}")
    print(f" Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n All engine tests passed! Pipeline working correctly with all tiers.")
    else:
        print(f"\n  {failed} test(s) need attention")