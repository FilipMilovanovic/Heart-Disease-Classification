# 🔍 Model Comparison Summary

This document provides a quick comparison of the three classification models used to predict heart disease: Naive Bayes, K-Nearest Neighbors (KNN), and Decision Tree. The evaluation is based on four key performance metrics: Accuracy, Precision, Recall, and F1-score.

## 📊 Results Overview

| Model         | Accuracy | Precision | Recall | F1-score |
|---------------|----------|-----------|--------|----------|
| Naive Bayes   | 87.98%   | 90.72%    | 87.13% | 88.89%   |
| KNN           | 86.89%   | 88.12%    | 88.12% | 88.12%   |
| Decision Tree | 86.89%   | 88.12%    | 88.12% | 88.12%   |

## 🧾 Interpretation

All three models performed fairly well, with similar accuracy and balanced precision/recall scores. However, the **Naive Bayes** classifier stood out slightly, achieving the highest precision and F1-score. This suggests it handled the trade-off between false positives and false negatives a bit better than the others.

## ✅ Conclusion

While KNN and Decision Tree yielded identical results, Naive Bayes emerged as the best overall performer on this dataset. For practical purposes, especially when prioritizing precision and overall balance, Naive Bayes would be the preferred choice among the three.
