# External Data Fusion

> The single highest-ROI step in ML competitions.

## The Rule

**External data consistently provides the largest single improvement — often 7x the impact of hyperparameter tuning.**

Before tuning any hyperparameters, ask: "What external data could I add?"

## When It Works Best

✅ Same target definition (same metric, same prediction target)
✅ Low noise in the external source
✅ Sufficient overlap with your training distribution
✅ Adds new information not in current features

## When It Fails

❌ Different target definitions (e.g., different scoring)
❌ High noise / sparse coverage
❌ Distribution mismatch (e.g., US data for a UK competition)
❌ Redundant information already in features

## Empirical Evidence

| Competition | External Data | Improvement |
|-------------|---------------|-------------|
| Jigsaw Toxic | XLM-RoBERTa pretrained on toxic data | +0.003 AUC |
| Text Normalization | Training class distribution priors | +0.5% accuracy |
| Denoising | Synthetic clean→dirty pairs | +0.005 RMSE |
| TPS May 2022 | f_27 character features (in-domain) | +0.00158 AUC |

## Procedure

```
1. Research what external data winners used
   - Check Kaggle discussion forums
   - Search for academic papers on the same problem
   - Look at the data sources listed in the problem

2. Check data availability
   - Public datasets (HuggingFace, Kaggle, government open data)
   - Pre-trained models (transformers, embeddings)
   - Domain-specific (molecular, financial, etc.)

3. Validate before adding
   - Sample 100 examples
   - Check distribution overlap
   - Verify same target definition

4. Integrate
   - As features
   - As embeddings
   - As pre-trained model fine-tuning
   - As additional training data

5. Measure impact
   - Single base model with new data vs without
   - Then add to stack
```

## Common External Data Sources

| Domain | Sources |
|--------|---------|
| NLP | HuggingFace models, Wikipedia dumps, Common Crawl |
| Vision | ImageNet pretrained, COCO, domain-specific datasets |
| Tabular | Government statistics, Kaggle public datasets |
| Time series | Bloomberg, FRED, market data APIs |
| Molecules | PubChem, ChEMBL, QM9 |
| Code | GitHub, StackOverflow, code corpora |

## Anti-Patterns

- ❌ **Surface-level fusion**: just adding features without domain understanding
- ❌ **Dumping raw data**: not aggregating, not processing, just adding columns
- ❌ **Different target**: using a model trained for different objective
- ❌ **Distribution shift ignored**: training data from 2010, competition data from 2020

## Diagnostic: Will External Data Help Here?

Ask these questions:
1. Do top 3 winners mention external data? (Check Kaggle forums)
2. Is there a public dataset that addresses this specific problem?
3. Is there a pre-trained model relevant to this domain?
4. Does my current model feel "data-starved" (e.g., 1000 rows for 100 features)?

If 2+ answers are yes, prioritize external data over hyperparameter tuning.

## Related Principles

- `work_smart_not_hard` — 7x ROI evidence
- `feature-engineering-roi` — when in-domain features are enough