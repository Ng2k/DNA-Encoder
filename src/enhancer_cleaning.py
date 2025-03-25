"""Optimized Enhancer Data Cleaning Pipeline with Performance Improvements"""

from typing import List, Dict, Set
import pandas as pd
from mygene import MyGeneInfo
import logging
import pandera as pa
from pandera.typing import Series
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from os import getcwd
from pandas.api.types import CategoricalDtype

# Configure logging
logging.basicConfig(
    filename=f"{getcwd()}/../../../logs/enhancer_cleaning.log",
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles optimized data loading with memory-efficient types"""
    
    def __init__(self, logger: logging.Logger = logger):
        self.logger = logger
        
    def load_data(self, input_path: str) -> pd.DataFrame:
        """Load data with optimized dtype detection and validation"""
        self.logger.info(f"Loading data from {input_path}")
        input_path = Path(input_path).resolve()
        
        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        return pd.read_csv(
            input_path,
            sep='\t',
            dtype={
                'Chromosome': 'category',
                'Tissue_class': 'category',
                'Enhancer_experiment': 'string'
            },
            engine='c',
            low_memory=False
        )

class ChromosomeCleaner:
    """Handles chromosome cleaning with vectorized operations"""
    
    VALID_CHROMS = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate chromosome names using vectorized operations"""
        df = df.copy()
        df['Chromosome'] = (
            df['Chromosome']
            .str.replace(r'^chr', '', regex=True)
            .replace({'M': 'MT', '': pd.NA})
            .astype('category')
        )
        valid_mask = df['Chromosome'].isin(self.VALID_CHROMS)
        return df[valid_mask].reset_index(drop=True)

class PositionProcessor:
    """Processes genomic positions with vectorized validation"""
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate positions using vectorized operations"""
        df = df.assign(
            Start_position=pd.to_numeric(df['Start_position'], errors='coerce'),
            End_position=pd.to_numeric(df['End_position'], errors='coerce')
        ).dropna(subset=['Start_position', 'End_position'])
        
        valid_mask = (
            df['End_position'].gt(df['Start_position']) &
            df['Start_position'].ge(0)
        )
        return df[valid_mask].astype({
            'Start_position': 'int32',
            'End_position': 'int32'
        }).reset_index(drop=True)

class GeneStandardizer:
    """Batch processes gene standardization with caching and parallel requests"""
    
    def __init__(self, logger: logging.Logger = logger, batch_size: int = 500):
        self.mg = MyGeneInfo()
        self.logger = logger
        self.batch_size = batch_size
        self.cache = {}

    def _process_batch(self, batch: List[str]) -> Dict[str, List[str]]:
        """Process a batch of genes with error handling"""
        try:
            uncached = [g for g in batch if g not in self.cache]
            if uncached:
                results = self.mg.querymany(
                    uncached,
                    scopes='symbol',
                    fields='symbol',
                    species='human',
                    as_dataframe=True
                )
                self.cache.update(results['symbol'].dropna()
                                  .groupby('query').apply(list)
                                  .to_dict())
            return {g: self.cache.get(g, []) for g in batch}
        except Exception as e:
            self.logger.warning(f"Gene batch error: {str(e)}")
            return {g: [] for g in batch}

    def standardize_genes(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Batch process gene standardization with parallel requests"""
        # Extract unique genes
        all_genes: Set[str] = set()
        df[column].str.split(',').apply(lambda x: all_genes.update(g.strip() for g in x if g.strip()))
        
        # Process in parallel batches
        with ThreadPoolExecutor() as executor:
            batches = [list(all_genes)[i:i+self.batch_size] 
                      for i in range(0, len(all_genes), self.batch_size)]
            results = list(executor.map(self._process_batch, batches))
        
        # Combine results
        mapping = {}
        for batch_result in results:
            mapping.update(batch_result)
        
        # Map results to DataFrame
        df['Enhancer_related_genes'] = (
            df[column]
            .str.split(',')
            .apply(lambda genes: list(set().union(*(mapping.get(g.strip(), []) for g in genes))))
        )
        return df

class ExperimentEncoder:
    """Optimized experiment encoder with NA handling"""
    
    METHOD_ALIASES = {
        'chip_seq': ['chipseq', 'chromatin immunoprecipitation'],
        'rna_seq': ['rnaseq', 'rna sequencing'],
        'western_blot': ['western', 'protein blot'],
        'rt_pcr': ['reverse transcription pcr'],
        'fish': ['fluorescence in situ hybridization'],
        'qpcr': ['quantitative pcr', 'real-time pcr']
    }
    
    def __init__(self, methods: List[str]):
        self.base_methods = methods
        self.valid_methods = self._expand_method_aliases(methods)
        
    def _expand_method_aliases(self, methods: List[str]) -> Set[str]:
        """Expand base methods to include known aliases"""
        expanded = set()
        for method in methods:
            expanded.add(method)
            for alias in self.METHOD_ALIASES.get(method, []):
                expanded.add(alias)
        return expanded
    
    def encode(self, df: pd.DataFrame, source_col: str) -> pd.DataFrame:
        """Create binary features with proper NA handling"""
        # Clean and normalize method names
        methods_series = (
            df[source_col]
            .fillna('')  # Handle missing values
            .str.lower()
            .str.replace(r'[^a-z0-9\s]', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )
        
        # Create binary indicators with NA safety
        for method in self.valid_methods:
            clean_method = method.lower().replace(' ', '_')
            mask = (
                methods_series
                .str.contains(clean_method)
                .fillna(False)  # Convert NA to False
            )
            df[f'exp_{clean_method}'] = mask.astype('int8')
        
        return df

class BooleanConverter:
    """Efficient boolean column converter"""
    
    def convert(self, df: pd.DataFrame, column: str, mapping: Dict) -> pd.DataFrame:
        """Convert column with optimized pandas operations"""
        df[column] = (
            df[column]
            .map(mapping)
            .fillna(0)
            .astype('int8')
        )
        return df

class TextCleaner:
    """Vectorized text cleaner using pandas string methods"""
    
    def clean_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Clean text columns using vectorized operations"""
        for col in columns:
            if col in df:
                df[col] = (
                    df[col]
                    .str.strip()
                    .str.replace(r'\s+', ' ', regex=True)
                )
        return df

class MissingValueHandler:
    def impute(self, df: pd.DataFrame, column: str, fill_value: str) -> pd.DataFrame:
        """Updated with modern categorical check"""
        if isinstance(df[column].dtype, CategoricalDtype):
            if fill_value not in df[column].cat.categories:
                df[column] = df[column].cat.add_categories([fill_value])
            return df.fillna({column: fill_value})
        return df.fillna({column: fill_value})

class DataValidator:
    """Updated schema with proper dtype handling"""
    
    class EnhancerSchema(pa.DataFrameModel):
        Chromosome: Series[str] = pa.Field(isin=ChromosomeCleaner.VALID_CHROMS)
        Start_position: Series[pa.Int] = pa.Field(ge=0, coerce=True)  # Allow any integer type
        End_position: Series[pa.Int] = pa.Field(ge=0, coerce=True)
        SNP_related: Series[pa.Int] = pa.Field(isin={0, 1}, coerce=True)
        Tissue_class: Series[str]

        @pa.dataframe_check
        def end_after_start(cls, df: pd.DataFrame) -> Series[bool]:
            return df['End_position'] > df['Start_position']

    def validate(self, df: pd.DataFrame):
        """Add dtype coercion before validation"""
        df = df.astype({
            'Start_position': 'int64',
            'End_position': 'int64',
            'SNP_related': 'int64'
        })
        try:
            self.EnhancerSchema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as err:
            logger.error(f"Validation failed:\n{err.failure_cases}")
            raise

class DataSaver:
    """Optimized data saver with path validation"""
    
    def __init__(self, logger: logging.Logger = logger):
        self.logger = logger
        
    def save_parquet(self, df: pd.DataFrame, output_path: str):
        """Save with optimized settings and path creation"""
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='zstd',
            index=False
        )
        self.logger.info(f"Saved cleaned data to {output_path}")

class EnhancerCleaningPipeline:
    """Optimized pipeline with performance monitoring"""
    
    def __init__(self, logger: logging.Logger = logger):
        self.logger = logger
        self._initialize_components()

    def _initialize_components(self):
        """Initialize optimized components"""
        self.loader = DataLoader(self.logger)
        self.chrom_cleaner = ChromosomeCleaner()
        self.pos_processor = PositionProcessor()
        self.gene_standardizer = GeneStandardizer(self.logger)
        self.exp_encoder = ExperimentEncoder([
            'ChIP-seq', 'RNA-seq', 'Western blot', 
            'RT-PCR', 'FISH', 'qPCR'
        ])
        self.bool_converter = BooleanConverter()
        self.text_cleaner = TextCleaner()
        self.missing_handler = MissingValueHandler()
        self.validator = DataValidator()
        self.saver = DataSaver(self.logger)

    def execute_pipeline(self, input_path: str, output_path: str):
        """Execute pipeline with error handling and resource monitoring"""
        try:
            df = self.loader.load_data(input_path)
            df = self.chrom_cleaner.clean(df)
            df = self.pos_processor.process(df)
            df = self.gene_standardizer.standardize_genes(df, 'Enhancer_related_genes')
            df = self.exp_encoder.encode(df, 'Enhancer_experiment')
            df = self.bool_converter.convert(df, 'SNP_related', {'否': 0, '是': 1})
            df = self.text_cleaner.clean_columns(df, ['Description', 'Title', 'Enhancer_function'])
            df = self.missing_handler.impute(df, 'Tissue_class', 'unknown')
            self.validator.validate(df)
            self.saver.save_parquet(df, output_path)
            return True
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    pipeline = EnhancerCleaningPipeline(logger)
    
    try:
        pipeline.execute_pipeline(
            input_path=f"{getcwd()}/enhancer_main.txt",
            output_path=f"{getcwd()}/../../processed/enhancer/enhancers_clean.parquet"
        )
    except Exception as e:
        logger.error(f"Critical pipeline failure: {str(e)}")
        raise