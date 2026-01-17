"""
test_features.py - Unit Tests for Feature Extractors

Tests each feature extraction function independently to ensure
they work correctly before integration.

Run with: pytest test_features.py
Or simply: python test_features.py
"""

import sys

from XAI_engine.xai_engine import FakeReviewXAI


# ============================================================================
# TIER 1 FEATURE TESTS
# ============================================================================

def test_sentiment_extreme_positive():
    """Should detect extremely positive sentiment"""
    xai = FakeReviewXAI()
    result = xai.extract_all_features("AMAZING!!! BEST EVER!!! PERFECT!!!")
    
    assert result['sentiment'] > 0.85, f"Expected sentiment > 0.85, got {result['sentiment']}"
    print(" Extreme positive sentiment detected")


def test_sentiment_extreme_negative():
    """Should detect extremely negative sentiment"""
    xai = FakeReviewXAI()
    result = xai.extract_all_features("TERRIBLE!!! WORST EVER!!! HORRIBLE!!!")
    
    assert result['sentiment'] < -0.85, f"Expected sentiment < -0.85, got {result['sentiment']}"
    print(" Extreme negative sentiment detected")


def test_sentiment_neutral():
    """Should detect neutral sentiment"""
    xai = FakeReviewXAI()
    result = xai.extract_all_features("Product works as expected. Does the job.")
    
    assert -0.1 < result['sentiment'] < 0.1, f"Expected neutral sentiment, got {result['sentiment']}"
    print(" Neutral sentiment detected")


def test_word_count_short():
    """Should detect short reviews"""
    xai = FakeReviewXAI()
    result = xai.extract_all_features("Great product")
    
    assert result['word_count'] == 2, f"Expected 2 words, got {result['word_count']}"
    assert result['word_count'] < 15, "Should be flagged as too short"
    print(" Short review detected")


def test_word_count_normal():
    """Should accept normal length reviews"""
    xai = FakeReviewXAI()
    result = xai.extract_all_features("I bought this product last week and it works well for basic tasks. Good value for the price.")
    
    assert 15 <= result['word_count'] <= 200, f"Expected normal length, got {result['word_count']}"
    print(" Normal length review detected")


def test_excessive_adjectives():
    """Should detect excessive adjectives without nouns"""
    xai = FakeReviewXAI()
    result = xai.extract_all_features("Amazing wonderful perfect great product")
    
    # 4 adjectives, 1 noun = ratio of 4.0
    assert result['adj_noun_ratio'] > 2.5, f"Expected ratio > 2.5, got {result['adj_noun_ratio']}"
    print(f" Excessive adjectives detected (ratio: {result['adj_noun_ratio']:.2f})")


def test_balanced_adjectives():
    """Should accept balanced adjective/noun usage"""
    xai = FakeReviewXAI()
    result = xai.extract_all_features("Good laptop with decent battery and comfortable keyboard")
    
    # More balanced: nouns like laptop, battery, keyboard
    assert result['adj_noun_ratio'] <= 2.5, f"Expected balanced ratio, got {result['adj_noun_ratio']}"
    print(f" Balanced language detected (ratio: {result['adj_noun_ratio']:.2f})")


def test_excessive_first_person():
    """Should detect excessive self-referencing"""
    xai = FakeReviewXAI()
    result = xai.extract_all_features("I love this! I bought it for my family and I think it's perfect for me!")
    
    assert result['first_person_ratio'] > 0.15, f"Expected ratio > 0.15, got {result['first_person_ratio']:.2%}"
    print(f" Excessive first-person usage detected ({result['first_person_ratio']:.2%})")


def test_normal_first_person():
    """Should accept normal first-person usage"""
    xai = FakeReviewXAI()
    result = xai.extract_all_features("Bought this last month. Battery life is good. Screen is bright.")
    
    assert result['first_person_ratio'] <= 0.15, f"Expected normal ratio, got {result['first_person_ratio']:.2%}"
    print(f" Normal first-person usage detected ({result['first_person_ratio']:.2%})")


# ============================================================================
# TIER 2 FEATURE TESTS
# ============================================================================
def test_spam_keywords():
    """Should detect spam/promotional keywords"""
    xai = FakeReviewXAI()
    result = xai.extract_all_features("Amazing product! Best ever! Must have!")
    
    assert result['spam_keyword_count'] >= 3, f"Expected >= 3 spam keywords, got {result['spam_keyword_count']}"
    print(f" Spam keywords detected ({result['spam_keyword_count']} found)")

def test_excessive_caps():
    """Should detect excessive capitalization"""
    xai = FakeReviewXAI()
    result = xai.extract_all_features("This is AMAZING! BEST product EVER!")
    
    assert result['caps_ratio'] > 0.20, f"Expected caps ratio > 0.20, got {result['caps_ratio']:.2%}"
    print(f" Excessive caps detected ({result['caps_ratio']:.2%})")


def test_excessive_punctuation():
    """Should detect excessive punctuation patterns"""
    xai = FakeReviewXAI()
    result = xai.extract_all_features("Great!!! Really??? I can't believe it... Best ever!!!")
    
    assert result['excessive_punct_count'] >= 3, f"Expected >= 3 patterns, got {result['excessive_punct_count']}"
    print(f" Excessive punctuation detected ({result['excessive_punct_count']} patterns)")


def test_text_redundancy():
    """Should detect repetitive text"""
    xai = FakeReviewXAI()
    result = xai.extract_all_features("Great great great product great quality great price great")
    
    assert result['uniqueness_ratio'] < 0.60, f"Expected uniqueness < 0.60, got {result['uniqueness_ratio']:.2%}"
    print(f" Text redundancy detected ({result['uniqueness_ratio']:.2%} unique)")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("RUNNING FEATURE EXTRACTION TESTS")
    print("="*60)
    
    tests = [
        # Tier 1
        test_sentiment_extreme_positive,
        test_sentiment_extreme_negative,
        test_sentiment_neutral,
        test_word_count_short,
        test_word_count_normal,
        test_excessive_adjectives,
        test_balanced_adjectives,
        test_excessive_first_person,
        test_normal_first_person,
        # Tier 2
        test_spam_keywords,
        test_excessive_caps,
        test_excessive_punctuation,
        test_text_redundancy
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\nRunning: {test.__name__}")
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f" Passed: {passed}/{len(tests)}")
    print(f" Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\nAll tests passed!")
    else:
        print(f"\n {failed} test(s) need attention")