from pandas_profiling import ProfileReport
import pandas as pd

# Load cleaned data
enhancer = pd.read_parquet("processed_data/enhancers_clean.parquet")

# Generate report
report = ProfileReport(
    enhancer,
    title="Enhancer Dataset Quality Report",
    explorative=True,
    variables={
        "descriptions": {
            "Pubmed": "PubMed article identifier",
            "exp_ChIP-seq": "ChIP-seq experimental evidence flag",
        }
    },
    missing_diagrams={
        "heatmap": False,
        "dendrogram": False
    }
)

# Save to HTML
report.to_file("reports/enhancer_quality_report.html")