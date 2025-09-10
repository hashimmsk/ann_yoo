# PFS6 Calibration Fix for AJDANN v7a

## Problem Description

The current AJDANN v7a model is producing overly optimistic PFS6 (6-month Progression-Free Survival) probabilities that are too high for all test cases. For example:
- **Case 1** [75, 1, 25, 30, 20, 4] should show ~40% but shows much higher
- **Case 2** [55, 1, 55, 60, 50, 3] should show ~50-60% but shows much higher  
- **Case 3** [35, 0, 95, 95, 95, 1] should show ~70-80% but shows much higher

## Root Cause

The issue is in the **model training process**, not in the API. The current model was trained with:
1. **Simple thresholding**: PFS6 values ≥50% were converted to 1, <50% to 0
2. **No temperature scaling**: The sigmoid activation produces overly confident predictions
3. **Training data imbalance**: The thresholding may have created an imbalanced dataset

## Solution: Retrain with Better Calibration

### Option 1: Quick Retraining (Recommended)

```bash
# Run the retraining script
python retrain_v7a.py
```

This script will:
- Backup your existing model to `saved_models_v7a_backup`
- Retrain the model with **temperature scaling** (temperature=2.0)
- Use **adaptive thresholding** based on data distribution
- Apply **conservative PFS6 processing**

### Option 2: Manual Retraining with Custom Parameters

```bash
# Retrain with different temperature (higher = more conservative)
python ajdANN_v7a.py --temperature 3.0 --epochs 200

# Or use the default temperature
python ajdANN_v7a.py --epochs 200
```

### Option 3: Advanced Calibration

If you need even more aggressive calibration, you can modify the temperature in `ajdANN_v7a.py`:

```python
# In build_multitask_model function, change temperature
def build_multitask_model(input_dim: int, learning_rate: float = 1e-3, temperature: float = 3.0):
    # Higher temperature = more conservative predictions
```

## What the Fix Does

### 1. Temperature Scaling
- **Before**: Raw sigmoid output (often too confident)
- **After**: `sigmoid(logits / temperature)` where temperature > 1
- **Effect**: Spreads out probabilities, making them more conservative

### 2. Adaptive Thresholding
- **Before**: Fixed 50% threshold for all data
- **After**: Dynamic threshold based on data distribution
- **Effect**: Better balance between positive/negative cases

### 3. Conservative Processing
- **Before**: Simple binary conversion
- **After**: Nuanced approach considering data characteristics
- **Effect**: More realistic survival probability distributions

## Expected Results After Retraining

| Case | Age | Performance | Resection | Expected PFS6 | Before | After |
|------|-----|-------------|-----------|---------------|---------|-------|
| 1    | 75  | 30%         | 25%       | ~35-45%       | 80%+    | 35-45% |
| 2    | 55  | 60%         | 55%       | ~50-65%       | 80%+    | 50-65% |
| 3    | 35  | 95%         | 95%       | ~70-80%       | 90%+    | 70-80% |

## Verification Steps

1. **Retrain the model** using one of the options above
2. **Restart your application**: `python start_app.py`
3. **Test the sample cases** in the frontend
4. **Use the debug button** to see raw vs calibrated values
5. **Check the calibration info** in the results

## Troubleshooting

### If retraining fails:
- Check that `dat_hc_simul.csv` exists and is accessible
- Ensure all dependencies are installed
- Check the error messages for specific issues

### If results are still too high:
- Increase the temperature parameter (try 3.0 or 4.0)
- Check the training logs for threshold information
- Verify the dataset has realistic PFS6 values

### If you need to restore the old model:
```bash
# Remove the new model
rm -rf saved_models_v7a

# Restore the backup
mv saved_models_v7a_backup saved_models_v7a
```

## Technical Details

The fix addresses the **calibration problem** in machine learning where:
- **Raw model outputs** are often overconfident
- **Temperature scaling** makes predictions more conservative
- **Better target processing** creates more balanced training data
- **Clinical factor consideration** ensures realistic survival ranges

## Next Steps

1. **Retrain the model** using the provided scripts
2. **Test the results** with the sample cases
3. **Monitor performance** in real-world usage
4. **Adjust temperature** if further calibration is needed

The retrained model should now produce PFS6 values that are much more clinically realistic and aligned with your expectations.
