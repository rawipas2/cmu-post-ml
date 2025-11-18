# Version Template - README.md

## ตัวอย่าง README ที่จะถูกสร้างอัตโนมัติสำหรับแต่ละ version

หลังจากการ training เสร็จ ระบบจะสร้าง README.md ใน `versions/<version_name>/README.md` โดยอัตโนมัติ

### เนื้อหาที่จะมี:

1. **Overview**

   - วันที่ train
   - Version name
   - เป้าหมาย

2. **Performance Summary**

   - Target accuracy
   - Best model
   - Average accuracy
   - Ensemble performance

3. **Individual Model Performance**

   - ตารางแสดง metrics ทุก model
   - Accuracy, Precision, Recall, F1, AUC-ROC

4. **Models Used**

   - รายละเอียดแต่ละ model
   - Architecture
   - Hyperparameters

5. **Ensemble Method**

   - วิธีการ ensemble
   - Meta-learner architecture

6. **Strengths ✅**

   - จุดแข็งของ version นี้
   - Models ที่ได้คะแนนดี
   - Features ที่โดดเด่น

7. **Weaknesses ❌**

   - จุดอ่อนที่พบ
   - Models ที่ต้องปรับปรุง
   - ข้อจำกัด

8. **Improvements for Next Version**

   - แนวทางการปรับปรุง
   - Suggestions
   - Ideas

9. **Files**

   - โครงสร้างไฟล์
   - Models, plots, metrics

10. **Notes**

    - บันทึกเพิ่มเติม
    - Observations

11. **Configuration**
    - Hyperparameters ที่ใช้
    - Settings

## ตัวอย่าง README ที่ได้จริง

```markdown
# Thai Depression Classification - v1.0

## Overview

Training run completed on: 2025-11-18 15:30:00

## Performance Summary

### Target

- **Target Accuracy**: 80.0%

### Results

- **Best Single Model**: Deep_Learning (0.8234)
- **Average Accuracy**: 0.7856
- **Ensemble Accuracy**: 0.8512 ✅ TARGET MET!

## Individual Model Performance

| Model             | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
| ----------------- | -------- | --------- | ------ | -------- | ------- |
| Ensemble_Stacking | 0.8512   | 0.8489    | 0.8512 | 0.8500   | 0.9123  |
| Deep_Learning     | 0.8234   | 0.8156    | 0.8234 | 0.8195   | 0.8876  |
| Neural_Network    | 0.8012   | 0.7956    | 0.8012 | 0.7984   | 0.8654  |
| Maximum_Entropy   | 0.7845   | 0.7798    | 0.7845 | 0.7821   | 0.8432  |
| SVM               | 0.7723   | 0.7689    | 0.7723 | 0.7706   | 0.8312  |
| Bayesian_Network  | 0.7656   | 0.7612    | 0.7656 | 0.7634   | 0.8245  |
| Naive_Bayes       | 0.7267   | 0.7234    | 0.7267 | 0.7250   | 0.7891  |

## Strengths ✅

- 1 model(s) achieved target accuracy
- Ensemble model successfully met target accuracy
- All models trained on GPU for optimal performance
- Comprehensive evaluation metrics and visualizations

## Weaknesses ❌

- 6 model(s) below target accuracy
  - Naive_Bayes: 0.7267
  - Bayesian_Network: 0.7656
  - SVM: 0.7723

## Improvements for Next Version

- Fine-tune hyperparameters for underperforming models
- Experiment with different ensemble techniques (voting, boosting)
- Try advanced Thai text preprocessing (subword tokenization)
- Increase model complexity or add more features
- Use pre-trained Thai language models (WangchanBERTa)
- Augment training data
```

## การใช้งาน

README จะถูกสร้างอัตโนมัติโดย:

```python
from utils import generate_readme

generate_readme(version_dir, all_metrics, notes="Custom notes here")
```

ไม่จำเป็นต้องสร้างเอง!
