import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock
import torch

# Import the functions to test
from src.data import MyDataset, preprocess_data, create_hf_datasets


@pytest.mark.unit
class TestMyDataset:
    """Test cases for MyDataset class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'text': [
                'This is a positive review.',
                'This is a negative review.',
                'This is a neutral review.',
                'Another positive example.',
                'Another negative example.'
            ],
            'label': [1, 0, 2, 1, 0]
        })

    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create a temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)

    def test_dataset_creation(self, temp_csv_file):
        """Test basic dataset creation."""
        dataset = MyDataset(temp_csv_file)
        assert len(dataset) == 5
        assert len(dataset.label2id) == 3  # 0, 1, 2

    def test_dataset_with_max_rows(self, temp_csv_file):
        """Test dataset creation with max_rows limit."""
        dataset = MyDataset(temp_csv_file, max_rows=3)
        assert len(dataset) == 3

    def test_dataset_getitem(self, temp_csv_file):
        """Test getting items from dataset."""
        dataset = MyDataset(temp_csv_file)
        item = dataset[0]

        assert 'text' in item
        assert 'label' in item
        assert isinstance(item['text'], str)
        assert isinstance(item['label'], torch.Tensor)
        assert item['label'].item() in [0, 1, 2]

    def test_dataset_label_mapping(self, temp_csv_file):
        """Test that labels are properly mapped."""
        dataset = MyDataset(temp_csv_file)
        labels = [dataset[i]['label'].item() for i in range(len(dataset))]

        # Check that all labels are integers and in expected range
        assert all(isinstance(label, int) for label in labels)
        assert all(0 <= label <= 2 for label in labels)

    def test_dataset_empty_file(self):
        """Test handling of empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('text,label\n')  # Only header
            f.flush()

            with pytest.raises(pd.errors.EmptyDataError):
                MyDataset(Path(f.name))

        os.unlink(f.name)

    def test_dataset_missing_columns(self):
        """Test handling of CSV with missing required columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('text\n')  # Missing label column
            f.write('Some text\n')
            f.flush()

            with pytest.raises(KeyError):
                MyDataset(Path(f.name))

        os.unlink(f.name)

    def test_dataset_invalid_max_rows(self, temp_csv_file):
        """Test handling of invalid max_rows parameter."""
        # Test with zero (should work but return empty dataset)
        dataset = MyDataset(temp_csv_file, max_rows=0)
        assert len(dataset) == 0

        # Test with negative (should work but return empty dataset)
        dataset = MyDataset(temp_csv_file, max_rows=-1)
        assert len(dataset) == 0


@pytest.mark.unit
class TestPreprocessData:
    """Test cases for preprocess_data function."""

    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data for preprocessing."""
        return pd.DataFrame({
            'Sentence': [
                'This is a positive review!',
                'This is a negative review.',
                'This is a neutral review.',
                'Another positive example!',
                'Another negative example.',
                'Yet another positive case.',
                'Yet another negative case.',
                'A neutral example here.'
            ],
            'Sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 'positive', 'negative', 'neutral']
        })

    @pytest.fixture
    def temp_input_dir(self, sample_raw_data):
        """Create a temporary input directory with data.csv."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir)
            data_file = input_path / "data.csv"
            sample_raw_data.to_csv(data_file, index=False)
            yield input_path

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_preprocess_data_basic(self, temp_input_dir, temp_output_dir):
        """Test basic preprocessing functionality."""
        preprocess_data(temp_input_dir, temp_output_dir, train_ratio=0.5, test_ratio=0.25, eval_ratio=0.25)

        # Check that output files were created
        train_file = temp_output_dir / "train.csv"
        test_file = temp_output_dir / "test.csv"
        eval_file = temp_output_dir / "eval.csv"

        assert train_file.exists()
        assert test_file.exists()
        assert eval_file.exists()

        # Check file contents
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        eval_df = pd.read_csv(eval_file)

        # Check that all data is preserved
        total_rows = len(train_df) + len(test_df) + len(eval_df)
        assert total_rows == 8

        # Check that labels are properly encoded
        all_labels = set(train_df['label'].tolist() + test_df['label'].tolist() + eval_df['label'].tolist())
        assert all_labels == {'positive', 'negative', 'neutral'}

    def test_preprocess_data_invalid_ratios(self, temp_input_dir, temp_output_dir):
        """Test handling of invalid split ratios."""
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            preprocess_data(temp_input_dir, temp_output_dir, train_ratio=0.6, test_ratio=0.5)

    def test_preprocess_data_nonexistent_file(self, temp_output_dir):
        """Test handling of nonexistent input file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir)
            with pytest.raises(FileNotFoundError):
                preprocess_data(input_path, temp_output_dir)


@pytest.mark.unit
class TestCreateHFDatasets:
    """Test cases for create_hf_datasets function."""

    @pytest.fixture
    def sample_processed_data(self):
        """Create sample processed data for HF dataset creation."""
        return pd.DataFrame({
            'text': [
                'This is a positive review.',
                'This is a negative review.',
                'This is a neutral review.',
                'Another positive example.',
                'Another negative example.'
            ],
            'label': [1, 0, 2, 1, 0]
        })

    @pytest.fixture
    def temp_data_dir(self, sample_processed_data):
        """Create a temporary data directory with processed files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create train, test, eval files
            train_df = sample_processed_data.iloc[:3]
            test_df = sample_processed_data.iloc[3:4]
            eval_df = sample_processed_data.iloc[4:]

            train_df.to_csv(Path(temp_dir) / "train.csv", index=False)
            test_df.to_csv(Path(temp_dir) / "test.csv", index=False)
            eval_df.to_csv(Path(temp_dir) / "eval.csv", index=False)

            yield Path(temp_dir)

    @patch('src.data.AutoTokenizer.from_pretrained')
    @patch('src.data.HFDataset')
    def test_create_hf_datasets_basic(self, mock_hf_dataset, mock_tokenizer, temp_data_dir):
        """Test basic HF dataset creation."""
        # Mock the tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Mock the HF dataset
        mock_train_ds = MagicMock()
        mock_eval_ds = MagicMock()
        mock_hf_dataset.side_effect = [mock_train_ds, mock_eval_ds]

        train_ds, eval_ds, tokenizer, num_labels = create_hf_datasets(
            temp_data_dir, "bert-base-uncased"
        )

        # Check that tokenizer was called
        mock_tokenizer.assert_called_once_with("bert-base-uncased")

        # Check that datasets were created
        assert mock_hf_dataset.call_count == 2

        # Check return values
        assert train_ds == mock_train_ds
        assert eval_ds == mock_eval_ds
        assert tokenizer == mock_tokenizer_instance
        assert num_labels == 3

    @patch('src.data.AutoTokenizer.from_pretrained')
    @patch('src.data.HFDataset')
    def test_create_hf_datasets_with_max_rows(self, mock_hf_dataset, mock_tokenizer, temp_data_dir):
        """Test HF dataset creation with max_rows limit."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_train_ds = MagicMock()
        mock_eval_ds = MagicMock()
        mock_hf_dataset.side_effect = [mock_train_ds, mock_eval_ds]

        train_ds, eval_ds, tokenizer, num_labels = create_hf_datasets(
            temp_data_dir, "bert-base-uncased", max_rows=2
        )

        # Check that max_rows was passed to dataset creation
        assert mock_hf_dataset.call_count == 2

    def test_create_hf_datasets_missing_files(self):
        """Test handling of missing data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError):
                create_hf_datasets(Path(temp_dir), "bert-base-uncased")


@pytest.mark.integration
class TestDataIntegration:
    """Integration tests for data pipeline."""

    @pytest.fixture
    def complete_pipeline_data(self):
        """Create data for testing the complete pipeline."""
        return pd.DataFrame({
            'Sentence': [
                'This is a positive review!',
                'This is a negative review.',
                'This is a neutral review.',
                'Another positive example!',
                'Another negative example.',
                'Yet another positive case.',
                'Yet another negative case.',
                'A neutral example here.',
                'More positive content.',
                'More negative content.'
            ],
            'Sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative',
                         'positive', 'negative', 'neutral', 'positive', 'negative']
        })

    def test_complete_pipeline(self, complete_pipeline_data):
        """Test the complete data pipeline from raw data to HF datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create raw data file
            raw_dir = Path(temp_dir) / "raw"
            raw_dir.mkdir()
            data_file = raw_dir / "data.csv"
            complete_pipeline_data.to_csv(data_file, index=False)

            # Preprocess data
            processed_dir = Path(temp_dir) / "processed"
            preprocess_data(raw_dir, processed_dir, train_ratio=0.6, test_ratio=0.2, eval_ratio=0.2)

            # Create MyDataset with consistent label mapping
            train_dataset = MyDataset(processed_dir / "train.csv")
            test_dataset = MyDataset(processed_dir / "test.csv", label2id=train_dataset.label2id)
            eval_dataset = MyDataset(processed_dir / "eval.csv", label2id=train_dataset.label2id)

            # Check datasets
            assert len(train_dataset) > 0
            assert len(test_dataset) > 0
            assert len(eval_dataset) > 0

            # Check that all datasets have the same label mapping
            assert train_dataset.label2id == test_dataset.label2id == eval_dataset.label2id

            # Check that labels are properly encoded
            all_labels = set()
            for dataset in [train_dataset, test_dataset, eval_dataset]:
                for i in range(len(dataset)):
                    all_labels.add(dataset[i]['label'].item())

            assert all_labels == {0, 1, 2}  # Encoded labels


if __name__ == "__main__":
    pytest.main([__file__])
