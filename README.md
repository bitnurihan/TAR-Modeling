# TAR-Predictive Modeling

1. Purpose
- Developed predictive modeling for 'grp' and 'reach%' based on raw data, and proceed with budget allocation by screen to acheive the highest reach% based on total advertising costs.

2. Methodology
- Calculated Modeling based on campaign targets(sex&age).
- Limits the media to up to three for calculating union of total media reach%.
- The curve fitting is used to modeling for non-linear regression.
- Used SLSQP method for budget optimization.
- Calculated the total reach% by applying cross-site duplicate rate due to duplication between sites.
- Calculated cross-site duplication through the panel's media usage data.

3. Description of files
 - modeling_by_python.py
 : extracts modeling constants and slop from raw data.
 - compare_r_py_modeling.py
 : compares R modeling curves with Python to verify which curves are better representative of the actual raw data.
 - cost_optimization.py
 : Budget allocation for media-mix to maximize the advertising effect(reach%) within the same budget.
 - modeling_visualization.py
 : compared raw data and modeling curves with determined coefficients and constants before client review.

** Raw data and modeling results can't be uploaded because it's confidential.
