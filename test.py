# Run this quick test:
from XAI_engine.feature_extractors import detect_spam_keywords, detect_excessive_punctuation

spam_result = detect_spam_keywords("Amazing best perfect")
print("Spam result:", spam_result)

punct_result = detect_excessive_punctuation("Great!!! Really???")
print("Punct result:", punct_result)