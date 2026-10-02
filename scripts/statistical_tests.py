"""
Statistical Testing for Benchmark Results
=========================================
Runs statistical tests (e.g., Wilcoxon signed-rank test, Student's t-test)
to compute p-values and 95% Confidence Intervals for benchmark scores,
verifying that differences between model alignment methods are statistically significant.
"""

import json
import numpy as np
from collections import defaultdict
from scipy import stats

def compute_confidence_interval(data, confidence=0.95):
    """Computes mean and 95% CI for a 1D array of data."""
    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2:
        return np.mean(a), 0.0
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def run_significance_test(model_a_scores, model_b_scores, test_type='wilcoxon'):
    """
    Runs a statistical test to check if Model B is significantly better than Model A.
    Returns the p-value.
    """
    if len(model_a_scores) != len(model_b_scores) or len(model_a_scores) == 0:
        return 1.0
        
    if test_type == 'wilcoxon':
        try:
            _, p_value = stats.wilcoxon(model_a_scores, model_b_scores, alternative='two-sided')
            return p_value
        except ValueError:
            return 1.0 # zero diff
    elif test_type == 'ttest':
        _, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
        return p_value
    return 1.0

def analyze_statistical_significance(results_file: str):
    """Parses results and computes CIs and significance between models."""
    with open(results_file, 'r') as f:
        results = json.load(f)
        
    # Group executability by model and problem
    scores = defaultdict(lambda: defaultdict(list))
    for r in results:
        scores[r['model']][r['problem_id']].append(r['metrics']['executability'])
        
    # Average across trials to get 1 score per problem per model
    model_problem_scores = defaultdict(list)
    for model, probs in scores.items():
        for prob_id, s_list in probs.items():
            model_problem_scores[model].append(np.mean(s_list))
            
    print("Statistical Analysis Report:")
    print("============================")
    
    models = list(model_problem_scores.keys())
    for m in models:
        m_scores = model_problem_scores[m]
        mean, err = compute_confidence_interval(m_scores)
        print(f"{m}: {mean*100:.1f}% ± {err*100:.1f}% (95% CI)")
        
    print("\nPairwise Significance (Wilcoxon Signed-Rank Test):")
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            m1, m2 = models[i], models[j]
            p = run_significance_test(model_problem_scores[m1], model_problem_scores[m2])
            sig = "*" if p < 0.05 else " "
            print(f"{m1} vs {m2}: p={p:.4f} {sig}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_statistical_significance(sys.argv[1])
    else:
        print("Usage: python statistical_tests.py <results_json_path>")
