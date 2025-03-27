# test_enhancer_pipeline.py
import pytest
import pandas as pd
import pandera as pa
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from pandas.api.types import CategoricalDtype
from src.enhancer_cleaning import (
    DataLoader, ChromosomeCleaner, PositionProcessor,
    GeneStandardizer, ExperimentEncoder, BooleanConverter,
    TextCleaner, MissingValueHandler, DataValidator,
    DataSaver, EnhancerCleaningPipeline
)

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Chromosome': ['chr1', 'chrX', 'chrMT', 'invalid'],
        'Start_position': ['100', '200', '300', 'invalid'],
        'End_position': ['150', '250', '350', 'invalid'],
        'Enhancer_related_genes': ['Gene1, Gene2', 'Gene3', None, 'Gene4'],
        'Enhancer_experiment': ['ChIP-seq;RNA-seq', 'Western blot', 'FISH', 'Unknown'],
        'SNP_related': ['是', '否', '是', 'invalid'],
        'Description': ['  Extra  spaces  ', None, 'Noise   ', ''],
        'Tissue_class': ['A', 'B', None, 'C'],
        'Enhancer_function': [None, '  func  ', '  ', np.nan]
    })

def test_data_loader_file_not_found():
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_data("nonexistent_file.tsv")

def test_data_loader_success(tmp_path, sample_data):
    test_file = tmp_path / "test.tsv"
    sample_data.to_csv(test_file, sep='\t', index=False)
    
    loader = DataLoader()
    df = loader.load_data(str(test_file))
    assert df.shape == (4, 8)
    assert df['Chromosome'].dtype == 'category'

def test_chromosome_cleaner_valid_data(sample_data):
    cleaner = ChromosomeCleaner()
    cleaned = cleaner.clean(sample_data)
    assert len(cleaned) == 3
    assert set(cleaned['Chromosome']) == {'1', 'X', 'MT'}

def test_chromosome_cleaner_empty_data():
    cleaner = ChromosomeCleaner()
    df = pd.DataFrame({'Chromosome': []})
    cleaned = cleaner.clean(df)
    assert cleaned.empty

def test_position_processor_valid_data(sample_data):
    processor = PositionProcessor()
    processed = processor.process(sample_data)
    assert processed['Start_position'].dtype == 'int64'
    assert len(processed) == 3

def test_position_processor_invalid_data():
    processor = PositionProcessor()
    df = pd.DataFrame({
        'Start_position': ['invalid', '200', None],
        'End_position': ['300', 'invalid', '400']
    })
    processed = processor.process(df)
    assert processed.empty

@patch('enhancer_pipeline.MyGeneInfo')
def test_gene_standardizer_success(mock_mygene, sample_data):
    mock_results = pd.DataFrame({
        'symbol': ['GENE1', 'GENE2', 'GENE3'],
        'query': ['Gene1', 'Gene2', 'Gene3']
    })
    mock_mygene.return_value.querymany.return_value = mock_results
    
    standardizer = GeneStandardizer()
    df = standardizer.standardize_genes(sample_data, 'Enhancer_related_genes')
    assert isinstance(df['Enhancer_related_genes'].iloc[0], list)
    assert len(df['Enhancer_related_genes'].iloc[0]) == 2

@patch('enhancer_pipeline.MyGeneInfo')
def test_gene_standardizer_api_failure(mock_mygene, sample_data):
    mock_mygene.return_value.querymany.side_effect = Exception("API Error")
    
    standardizer = GeneStandardizer()
    df = standardizer.standardize_genes(sample_data, 'Enhancer_related_genes')
    assert len(df['Enhancer_related_genes'].iloc[0]) == 0

def test_experiment_encoder_success(sample_data):
    encoder = ExperimentEncoder(['ChIP-seq'])
    encoded = encoder.encode(sample_data, 'Enhancer_experiment')
    assert 'exp_chip_seq' in encoded.columns
    assert encoded['exp_chip_seq'].sum() == 1

def test_experiment_encoder_null_handling():
    encoder = ExperimentEncoder(['ChIP-seq'])
    df = pd.DataFrame({'Enhancer_experiment': [None, np.nan, '']})
    encoded = encoder.encode(df, 'Enhancer_experiment')
    assert encoded['exp_chip_seq'].sum() == 0

def test_boolean_converter_valid_mapping(sample_data):
    converter = BooleanConverter()
    converted = converter.convert(sample_data, 'SNP_related', {'是': 1, '否': 0})
    assert set(converted['SNP_related']) == {0, 1}

def test_boolean_converter_missing_values():
    converter = BooleanConverter()
    df = pd.DataFrame({'SNP_related': [None, np.nan, '']})
    converted = converter.convert(df, 'SNP_related', {'是': 1})
    assert converted['SNP_related'].sum() == 0

def test_text_cleaner_success(sample_data):
    cleaner = TextCleaner()
    cleaned = cleaner.clean_columns(sample_data, ['Description', 'Enhancer_function'])
    assert cleaned['Description'].iloc[0] == 'Extra spaces'
    assert cleaned['Enhancer_function'].iloc[1] == 'func'

def test_text_cleaner_empty_columns():
    cleaner = TextCleaner()
    df = pd.DataFrame({'Test': ['  ', None, np.nan]})
    cleaned = cleaner.clean_columns(df, ['Test'])
    assert cleaned['Test'].iloc[0] == ''

def test_missing_value_handler_categorical():
    handler = MissingValueHandler()
    df = pd.DataFrame({'Tissue_class': pd.Categorical(['A', 'B', None])})
    imputed = handler.impute(df, 'Tissue_class', 'unknown')
    assert imputed['Tissue_class'].isna().sum() == 0
    assert isinstance(imputed['Tissue_class'].dtype, CategoricalDtype)

def test_missing_value_handler_non_categorical():
    handler = MissingValueHandler()
    df = pd.DataFrame({'Tissue_class': ['A', None, 'B']})
    imputed = handler.impute(df, 'Tissue_class', 'unknown')
    assert imputed['Tissue_class'].isna().sum() == 0

def test_data_validator_success():
    validator = DataValidator()
    df = pd.DataFrame({
        'Chromosome': ['1', 'X', 'MT'],
        'Start_position': [100, 200, 300],
        'End_position': [150, 250, 350],
        'SNP_related': [1, 0, 1],
        'Tissue_class': ['A', 'B', 'C']
    })
    validator.validate(df)  # Should not raise

def test_data_validator_failure():
    validator = DataValidator()
    df = pd.DataFrame({
        'Chromosome': ['invalid'],
        'Start_position': [-1],
        'End_position': [0],
        'SNP_related': [2],
        'Tissue_class': [None]
    })
    with pytest.raises(pa.errors.SchemaErrors):
        validator.validate(df)

def test_data_saver_success(tmp_path, sample_data):
    saver = DataSaver()
    output_path = tmp_path / "output.parquet"
    saver.save_parquet(sample_data, str(output_path))
    assert output_path.exists()

def test_data_saver_invalid_path():
    saver = DataSaver()
    with pytest.raises(IOError):
        saver.save_parquet(pd.DataFrame(), "/invalid/path/output.parquet")

@patch('enhancer_pipeline.MyGeneInfo')
def test_full_pipeline_success(mock_mygene, tmp_path, sample_data):
    input_path = tmp_path / "input.tsv"
    output_path = tmp_path / "output.parquet"
    sample_data.to_csv(input_path, sep='\t', index=False)
    
    mock_results = pd.DataFrame({'symbol': ['GENE1'], 'query': ['Gene1']})
    mock_mygene.return_value.querymany.return_value = mock_results
    
    pipeline = EnhancerCleaningPipeline()
    result = pipeline.execute_pipeline(str(input_path), str(output_path))
    assert result is True
    assert output_path.exists()

def test_pipeline_file_not_found():
    pipeline = EnhancerCleaningPipeline()
    with pytest.raises(FileNotFoundError):
        pipeline.execute_pipeline("invalid.tsv", "output.parquet")

def test_pipeline_validation_failure(tmp_path, sample_data):
    input_path = tmp_path / "input.tsv"
    output_path = tmp_path / "output.parquet"
    sample_data['Start_position'] = [-100, -200, -300, -400]
    sample_data.to_csv(input_path, sep='\t', index=False)
    
    pipeline = EnhancerCleaningPipeline()
    with pytest.raises(pa.errors.SchemaErrors):
        pipeline.execute_pipeline(str(input_path), str(output_path))

def test_pipeline_gene_standardization_failure(tmp_path, sample_data):
    input_path = tmp_path / "input.tsv"
    output_path = tmp_path / "output.parquet"
    sample_data.to_csv(input_path, sep='\t', index=False)
    
    with patch('enhancer_pipeline.MyGeneInfo') as mock_mygene:
        mock_mygene.return_value.querymany.side_effect = Exception("API Failure")
        pipeline = EnhancerCleaningPipeline()
        with pytest.raises(Exception):
            pipeline.execute_pipeline(str(input_path), str(output_path))