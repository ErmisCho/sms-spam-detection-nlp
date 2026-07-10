# TF-IDF Classifier Evaluation

## Summary

- Row-stratified accuracy: 0.9883
- Row-stratified SPAM precision: 0.9595
- Row-stratified SPAM recall: 0.9530
- Row-stratified SPAM F1: 0.9562
- Duplicate-safe SPAM F1: 0.9416
- Row-stratified false positives: 6
- Row-stratified false negatives: 7

## Duplicate-Safe Check

- Duplicate message rows retained in validated data: 703
- Duplicate message groups: 289
- Row-stratified split text overlap count: 117
- Row-stratified test rows with text also present in train: 130
- Duplicate-safe strategy: group_shuffle_split_by_text
- Duplicate-safe train/test rows: 4433 / 1139
- Duplicate-safe train/test text overlap count: 0
- Duplicate-safe accuracy: 0.9860
- Duplicate-safe SPAM precision: 0.9348
- Duplicate-safe SPAM recall: 0.9485
- Duplicate-safe SPAM F1: 0.9416

## Row-Stratified Classification Report

```text
              precision    recall  f1-score   support

         ham       0.99      0.99      0.99       966
        spam       0.96      0.95      0.96       149

    accuracy                           0.99      1115
   macro avg       0.98      0.97      0.97      1115
weighted avg       0.99      0.99      0.99      1115

```

## Duplicate-Safe Classification Report

```text
              precision    recall  f1-score   support

         ham       0.99      0.99      0.99      1003
        spam       0.93      0.95      0.94       136

    accuracy                           0.99      1139
   macro avg       0.96      0.97      0.97      1139
weighted avg       0.99      0.99      0.99      1139

```

## Error Analysis Notes

- False positives are HAM messages predicted as SPAM; these matter because they could block legitimate messages.
- False negatives are SPAM messages predicted as HAM; these matter because spam reaches the inbox.
- Accuracy should be interpreted alongside SPAM recall and precision because the dataset is imbalanced.
- The duplicate-safe grouped split is the stricter score to cite when discussing potential duplicate leakage.
