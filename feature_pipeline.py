import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')
 
print("=" * 60)
print("  MODULE 1 | LAB 1.3")
print("  Feature Engineering Pipeline")
print("  Measuring lift over Lab 1.2 baseline")
print("=" * 60)
 
# ---------------------------------------------------------------------------
# SECTION 1: Load Events
# ---------------------------------------------------------------------------
print("\n[1] Loading events...")
 
events = pd.read_csv("data/events.csv")
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')
 
purchaser_set = set(
    events[events['event'] == 'transaction']['visitorid'].unique()
)
 
pre_purchase = events[events['event'].isin(['view', 'addtocart'])].copy()
 
print("    Loading baseline feature matrix from Lab 1.2...")
# Load the baseline data generated in Lab 1.2 to build your new pipeline onto
user_df = pd.read_csv("data/user_features_baseline.csv")
print(f"    Baseline columns: {list(user_df.columns)}")
 
# ---------------------------------------------------------------------------
# SECTION 2: HELPER — Train XGBoost and Return AUC
# ---------------------------------------------------------------------------
# This function isolates tracking parameters to let us measure wave lifts cleanly.

def train_and_evaluate(df, feature_cols, label="Model"):
    """
    Trains an XGBoost model given specific feature columns and tracks classification results.
    Returns: (auc_score, average_precision, fitted_model)
    """
    # TODO: Isolate features (X) using feature_cols, and target labels (y) using 'purchased'
    X = None
    y = None
 
    # TODO: Create train/test splits (80/20 ratio, stratified across y, random_state=42)
    X_tr, X_te, y_tr, y_te = None
    
    # TODO: Compute the negative-to-positive class weight ratio to handle background label imbalance
    spw = None
 
    # TODO: Build an XGBClassifier instance. 
    # Params: n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    # scale_pos_weight=spw, eval_metric='auc', early_stopping_rounds=20, random_state=42, verbosity=0
    model = None
    
    # TODO: Fit the configuration using X_tr and y_tr, setting eval_set to monitor [(X_te, y_te)] silently (verbose=False)


    # TODO: Gather positive class prediction probabilities on test partitions
    proba = None
    
    # TODO: Evaluate performance metrics via roc_auc_score and average_precision_score
    auc = None
    ap = None
 
    print(f"    {label:<45} AUC={auc:.4f}  AP={ap:.4f}")
    return auc, ap, model
 
# ---------------------------------------------------------------------------
# SECTION 3: ESTABLISH BASELINE from Lab 1.2
# ---------------------------------------------------------------------------
print("\n[2] Reproducing Lab 1.2 baseline (4 raw features)...")
 
RAW_FEATURES = [
    'total_views',
    'total_addtocart',
    'unique_items_viewed',
    'active_days'
]
 
# TODO: Use the helper function above to execute training on RAW_FEATURES and establish a benchmark
base_auc, base_ap, _ = None
 
results = [("Baseline — Raw Features (Lab 1.2)", base_auc, base_ap, len(RAW_FEATURES))]
 
# ---------------------------------------------------------------------------
# SECTION 4: WAVE 1 — Ratio Features
# ---------------------------------------------------------------------------
# Ratios normalize raw counts across users with different session lengths.
print("\n[3] WAVE 1 — Adding ratio features...")
 
# TODO: Construct 'cart_to_view_ratio' representing intent density per user action
# Formula Hint: user_df['total_addtocart'] divided by (user_df['total_views'] + 1) to prevent zero-division errors
user_df['cart_to_view_ratio'] = None
 
# TODO: Construct 'views_per_day' measuring individual user engagement intensity over time
# Formula Hint: user_df['total_views'] divided by (user_df['active_days'] + 1)
user_df['views_per_day'] = None
 
WAVE1_FEATURES = RAW_FEATURES + ['cart_to_view_ratio', 'views_per_day']
 
# TODO: Evaluate performance across WAVE1_FEATURES using the training helper
w1_auc, w1_ap, _ = None
results.append(("Wave 1 — + Ratio Features", w1_auc, w1_ap, len(WAVE1_FEATURES)))
 
# ---------------------------------------------------------------------------
# SECTION 5: WAVE 2 — Datetime Features
# ---------------------------------------------------------------------------
# Extracts temporal contexts to check if shopping times correlate with purchase intent.
print("\n[4] WAVE 2 — Adding datetime features...")
 
# TODO: Find the absolute minimum timestamp for each visitor, and extract its 'hour' parameter
# Hint: Group pre_purchase by 'visitorid' over 'datetime', fetch .min(), then use the .dt.hour accessor
first_hour = None
 
# TODO: Find the absolute minimum timestamp for each visitor, and extract its day-of-week index (0=Monday, 6=Sunday)
# Hint: Use the .dt.dayofweek accessor on the minimum grouped timestamp
first_dow = None
 
# TODO: Left-merge both first_hour and first_dow onto user_df using 'visitorid' as your join key
user_df = None

# TODO: Fill any resulting unmapped NaN times or missing records with 0
user_df[['first_active_hour', 'first_active_dow']] = None

# TODO: Construct a binary flag 'is_weekend_shopper' (1 if first_active_dow is greater than or equal to 5, else 0)
user_df['is_weekend_shopper'] = None
 
WAVE2_FEATURES = WAVE1_FEATURES + [
    'first_active_hour',
    'first_active_dow',
    'is_weekend_shopper'
]
 
# TODO: Evaluate cumulative wave metrics using WAVE2_FEATURES
w2_auc, w2_ap, _ = None
results.append(("Wave 2 — + Datetime Features", w2_auc, w2_ap, len(WAVE2_FEATURES)))
 
# ---------------------------------------------------------------------------
# SECTION 6: WAVE 3 — Interaction Terms
# ---------------------------------------------------------------------------
# Interaction terms capture multi-variable relationships that tree nodes can use directly.
print("\n[5] WAVE 3 — Adding interaction terms...")
 
# TODO: Model a combined activity product tracking total views against items added to cart
user_df['views_x_cart'] = None
 
# TODO: Model a product tracking cross-session shopping breadth across loyalty spans
user_df['days_x_unique'] = None
 
# TODO: Model a product capturing intent rates tracking alongside total active sessions
user_df['ratio_x_days'] = None
 
WAVE3_FEATURES = WAVE2_FEATURES + [
    'views_x_cart',
    'days_x_unique',
    'ratio_x_days'
]
 
# TODO: Run the final training iteration across WAVE3_FEATURES using the helper function
w3_auc, w3_ap, final_model = None
results.append(("Wave 3 — + Interaction Terms (Final)", w3_auc, w3_ap, len(WAVE3_FEATURES)))
 
# ---------------------------------------------------------------------------
# SECTION 7: Full Lift Summary Table
# ---------------------------------------------------------------------------
print("\n[6] FULL LIFT SUMMARY — Lab 1.2 to Lab 1.3:")
print(f"\n    {'Stage':<45} {'Features':>9} {'AUC':>8} {'AP':>8} {'AUC Lift':>10}")
print(f"    {'-'*82}")
 
# TODO: Run a validation check parsing progression updates relative to index 0 results
# Print metrics tracking cumulative performance changes across each added wave


# ---------------------------------------------------------------------------
# SECTION 8: Visualise Lift Progression
# ---------------------------------------------------------------------------
print("\n[7] Plotting lift progression...")
 
stages   = [r[0].split("—")[0].strip() for r in results]
aucs     = [r[1] for r in results]
n_feats  = [r[3] for r in results]
colors   = ['#9E9E9E', '#4C72B0', '#2E75B6', '#1B3A6B']
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lab 1.3: Feature Engineering Lift over Baseline\n(Each bar = one engineering wave added on top of previous)", fontsize=12, fontweight='bold')
 
# --- Left Subplot: AUC Progression ---
# TODO: Create a bar chart on axes[0] visualizing the change in AUC score over stages
bars = None

# TODO: Overlay a target benchmark line using axes[0].axhline(y=0.80, color='red', linestyle='--')
# Set titles, labels, x-ticks, and add values above each bar for context


# --- Right Subplot: Feature Growth ---
# TODO: Create a bar chart on axes[1] tracing feature counts (n_feats) across stages
# Apply stylistic ticks, titles, and bar labels mapping configuration growth


plt.tight_layout()
plt.savefig("output/03_feature_engineering_lift.png", dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# SECTION 9: Final Feature Importance
# ---------------------------------------------------------------------------
print("\n[8] Final feature importance after all engineering waves...")
 
# TODO: Extract structural gains by compiling an execution matrix based on the final_model
# Construct a DataFrame matching columns 'feature' (WAVE3_FEATURES) and 'importance' (final_model.feature_importances_)
# Sort the entries descending by feature impact
imp_df = None

print(imp_df.to_string(index=False) if imp_df is not None else "    Not Implemented")
 
# --- Horizontal Importance Visualization ---
fig, ax = plt.subplots(figsize=(9, 6))
# TODO: Trace an ax.barh metric representation detailing engineering output metrics
# Optional Hint: slicing with [::-1] mirrors structures to present the most significant components at the top


ax.set_title("Feature Importance — Full Engineered Set\n(Gold = top 3 features)", fontweight='bold')
ax.set_xlabel("Importance (Gain)")
plt.tight_layout()
plt.savefig("output/03_final_feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# SECTION 10: Save Full Engineered Feature Matrix for Lab 1.4
# ---------------------------------------------------------------------------
# TODO: Save your new user_df matrix containing 'visitorid', your engineered features (WAVE3_FEATURES), and 'purchased' to a CSV file
# Target Path: "data/user_features_engineered.csv" (Set index=False)


print("\n    Saved -> data/user_features_engineered.csv")
print("    (Lab 1.4 loads this for LightGBM + Optuna tuning)")
