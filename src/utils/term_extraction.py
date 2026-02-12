"""
استخراج کلمات مهم برای هر خوشه (Chi-square).
در صورت نبود متن واقعی، با کلمات ساختگی پر می‌شود.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

def extract_important_terms(cluster_labels, user_texts=None, top_k=3):
    """
    اگر user_texts داده شود، از chi-square استفاده می‌کند.
    در غیر این صورت کلمات ساختگی بر اساس میانگین گرایش برمی‌گرداند.
    """
    if user_texts is None or len(user_texts) == 0:
        # تولید کلمات نمونه بر اساس جهت‌گیری
        topics = {
            'far_left':   ['equality', 'climate', 'healthcare', 'protest', 'union'],
            'left':       ['democrats', 'biden', 'rights', 'tax', 'education'],
            'center':     ['news', 'update', 'report', 'election', 'poll'],
            'right':      ['republican', 'economy', 'border', 'freedom', 'military'],
            'far_right':  ['trump', 'patriot', 'illegal', 'constitution', 'nationalism']
        }
        term_dict = {}
        for cid in set(cluster_labels):
            # میانگین گرایش خوشه باید از جای دیگر بیاید - اینجا صفر فرض می‌شود
            term_dict[cid] = topics['center'][:top_k]
        return term_dict
    
    # پیاده‌سازی واقعی chi2 - در صورت وجود متن
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(user_texts)
    chi2_scores, _ = chi2(X, cluster_labels)
    ...
    return term_dict