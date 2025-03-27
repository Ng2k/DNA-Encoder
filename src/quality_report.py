from ydata_profiling import ProfileReport
import pandas as pd
import os
import numpy as np
from pathlib import Path

def safe_profile_report(df: pd.DataFrame, output_path: str):
    """Genera il report con gestione degli errori incorporata"""
    config = {
        "title": "Enhancer Dataset Quality Report",
        "explorative": True,
        "vars": {
            "cat": {"words": False}  # Disabilita analisi testuale
        },
        "correlations": {
            "auto": {"calculate": False}  # Disabilita correlazioni automatiche
        },
        "missing_diagrams": {
            "heatmap": False,
            "dendrogram": False
        },
        "samples": {
            "head": 50  # Limita il numero di campioni mostrati
        }
    }
    
    try:
        report = ProfileReport(df, **config)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.to_file(output_path)
        print(f"Report generato con successo: {output_path}")
    except Exception as e:
        print(f"Errore nella generazione del report: {str(e)}")
        print("Generazione report in modalità minimal...")
        safe_config = config.copy()
        safe_config.update({
            "correlations": None,
            "missing_diagrams": None,
            "interactions": None
        })
        minimal_report = ProfileReport(df, **safe_config)
        minimal_report.to_file(output_path.replace(".html", "_minimal.html"))

if __name__ == "__main__":
    input_path = Path(__file__).parent.parent / "data" / "processed" / "enhancer" / "enhancers_clean.parquet"
    output_path = Path(__file__).parent.parent / "reports" / "enhancer_quality_report.html"
    
    try:
        enhancer = pd.read_parquet(input_path)
        safe_profile_report(enhancer, str(output_path))
    except Exception as e:
        print(f"Errore critico: {str(e)}")
        print("Verificare:")
        print(f"1. Il file {input_path} esiste")
        print("2. Le colonne contengono dati validi")
        print("3. Le dipendenze sono aggiornate")