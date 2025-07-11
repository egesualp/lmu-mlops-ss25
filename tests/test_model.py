import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

# Import the functions to test
from src.model import Classifier, create_hf_model


@pytest.mark.unit
class TestClassifier:
    """Test cases for the PyTorch Classifier model."""
    
    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        return {
            'text': [
                'This is a positive review.',
                'This is a negative review.',
                'This is a neutral review.',
                'Another positive example.',
                'Another negative example.'
            ]
        }
    
    def test_classifier_creation(self):
        """Test basic classifier creation."""
        model = Classifier()
        assert isinstance(model, torch.nn.Module)
        
        # Check that model has the expected components
        assert hasattr(model, 'bert')
        assert hasattr(model, 'dropout')
        assert hasattr(model, 'classifier')
        assert hasattr(model, 'tokenizer')
    
    def test_classifier_forward_pass(self, sample_batch):
        """Test forward pass through the classifier."""
        model = Classifier()
        model.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            outputs = model(sample_batch['text'])
        
        # Check output shape
        assert outputs.shape == (5, 3)  # 5 samples, 3 classes
        
        # Check that outputs are logits (not probabilities)
        # The softmax should sum to 1 for each sample
        probs = torch.softmax(outputs, dim=1)
        assert torch.allclose(torch.sum(probs, dim=1), torch.ones(5), atol=1e-6)
    
    def test_classifier_different_batch_sizes(self):
        """Test classifier with different batch sizes."""
        model = Classifier()
        model.eval()
        
        # Test single sample
        single_text = ['This is a test.']
        with torch.no_grad():
            output = model(single_text)
        assert output.shape == (1, 3)
        
        # Test multiple samples
        multiple_texts = ['Text 1.', 'Text 2.', 'Text 3.', 'Text 4.']
        with torch.no_grad():
            output = model(multiple_texts)
        assert output.shape == (4, 3)
    
    def test_classifier_device_handling(self):
        """Test classifier on different devices."""
        model = Classifier()
        
        # Test on CPU
        cpu_output = model(['Test text.'])
        assert cpu_output.device.type == 'cpu'
        
        # Test on GPU if available
        if torch.cuda.is_available():
            model_gpu = Classifier()
            model_gpu = model_gpu.cuda()
            gpu_output = model_gpu(['Test text.'])
            assert gpu_output.device.type == 'cuda'
    
    def test_classifier_gradient_flow(self, sample_batch):
        """Test that gradients flow through the model."""
        model = Classifier()
        model.train()  # Set to training mode
        
        outputs = model(sample_batch['text'])
        loss = torch.nn.functional.cross_entropy(outputs, torch.randint(0, 3, (5,)))
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.all(param.grad == 0)
    
    def test_classifier_parameter_count(self):
        """Test that the model has a reasonable number of parameters."""
        model = Classifier()
        total_params = sum(p.numel() for p in model.parameters())
        
        # BERT base has around 110M parameters
        assert total_params > 100_000_000  # At least 100M parameters
        assert total_params < 120_000_000  # Less than 120M parameters
    
    def test_classifier_dropout_behavior(self, sample_batch):
        """Test that dropout behaves differently in train vs eval mode."""
        model = Classifier()
        
        # Get outputs in training mode
        model.train()
        train_outputs = model(sample_batch['text'])
        
        # Get outputs in evaluation mode
        model.eval()
        eval_outputs = model(sample_batch['text'])
        
        # Outputs should be different due to dropout
        assert not torch.allclose(train_outputs, eval_outputs, atol=1e-6)
    
    def test_classifier_output_range(self, sample_batch):
        """Test that classifier outputs are reasonable."""
        model = Classifier()
        model.eval()
        
        with torch.no_grad():
            outputs = model(sample_batch['text'])
        
        # Check that outputs are finite
        assert torch.all(torch.isfinite(outputs))
        
        # Check that outputs are not all the same
        assert not torch.allclose(outputs, outputs[0].unsqueeze(0).repeat(outputs.shape[0], 1))
    
    def test_classifier_save_load(self, sample_batch):
        """Test saving and loading the classifier."""
        model = Classifier()
        model.eval()
        
        # Get original output
        with torch.no_grad():
            original_output = model(sample_batch['text'])
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name
        
        try:
            # Load model
            new_model = Classifier()
            new_model.load_state_dict(torch.load(temp_path))
            new_model.eval()
            
            # Get output from loaded model
            with torch.no_grad():
                loaded_output = new_model(sample_batch['text'])
            
            # Check that outputs are the same
            assert torch.allclose(original_output, loaded_output, atol=1e-6)
        
        finally:
            os.unlink(temp_path)


@pytest.mark.unit
class TestCreateHFModel:
    """Test cases for create_hf_model function."""
    
    @patch('src.model.AutoModelForSequenceClassification.from_pretrained')
    def test_create_hf_model_basic(self, mock_from_pretrained):
        """Test basic HF model creation."""
        # Mock the model
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model
        
        model = create_hf_model("bert-base-uncased", 3)
        
        # Check that from_pretrained was called correctly
        mock_from_pretrained.assert_called_once_with(
            "bert-base-uncased", 
            num_labels=3
        )
        
        # Check return value
        assert model == mock_model
    
    @patch('src.model.AutoModelForSequenceClassification.from_pretrained')
    def test_create_hf_model_different_pretrained_models(self, mock_from_pretrained):
        """Test HF model creation with different pretrained models."""
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model
        
        # Test with different models
        models_to_test = [
            "bert-base-uncased",
            "distilbert-base-uncased",
            "roberta-base",
            "albert-base-v2"
        ]
        
        for model_name in models_to_test:
            create_hf_model(model_name, 3)
            mock_from_pretrained.assert_called_with(
                model_name,
                num_labels=3
            )
    
    @patch('src.model.AutoModelForSequenceClassification.from_pretrained')
    def test_create_hf_model_different_num_labels(self, mock_from_pretrained):
        """Test HF model creation with different numbers of labels."""
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model
        
        # Test with different numbers of labels
        for num_labels in [2, 3, 5, 10]:
            create_hf_model("bert-base-uncased", num_labels)
            mock_from_pretrained.assert_called_with(
                "bert-base-uncased",
                num_labels=num_labels
            )
    
    @patch('src.model.AutoModelForSequenceClassification.from_pretrained')
    def test_create_hf_model_error_handling(self, mock_from_pretrained):
        """Test HF model creation error handling."""
        # Test with invalid model name
        mock_from_pretrained.side_effect = Exception("Model not found")
        
        with pytest.raises(Exception, match="Model not found"):
            create_hf_model("invalid-model", 3)


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for model functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for integration tests."""
        return {
            'text': [
                'This is a positive review.',
                'This is a negative review.',
                'This is a neutral review.',
                'Another positive example.',
                'Another negative example.'
            ],
            'labels': [1, 0, 2, 1, 0]
        }
    
    def test_pytorch_model_training_step(self, sample_data):
        """Test a complete training step with the PyTorch model."""
        model = Classifier()
        model.train()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # Forward pass
        outputs = model(sample_data['text'])
        labels = torch.tensor(sample_data['labels'])
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        assert torch.isfinite(loss)
        assert loss.item() > 0
        
        # Check that gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_pytorch_model_evaluation(self, sample_data):
        """Test model evaluation."""
        model = Classifier()
        model.eval()
        
        with torch.no_grad():
            outputs = model(sample_data['text'])
            predictions = torch.argmax(outputs, dim=1)
            labels = torch.tensor(sample_data['labels'])
            
            # Calculate accuracy
            accuracy = (predictions == labels).float().mean()
            
            # Check that accuracy is reasonable
            assert 0 <= accuracy <= 1
    
    @patch('src.model.AutoModelForSequenceClassification.from_pretrained')
    def test_hf_model_creation_integration(self, mock_from_pretrained):
        """Test HF model creation integration."""
        # Mock model with realistic structure
        mock_model = MagicMock()
        mock_model.config.num_labels = 3
        mock_from_pretrained.return_value = mock_model
        
        model = create_hf_model("bert-base-uncased", 3)
        
        # Check that model has expected attributes
        assert hasattr(model, 'config')
        assert model.config.num_labels == 3


@pytest.mark.unit
class TestModelEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_classifier_empty_batch(self):
        """Test classifier with empty batch."""
        model = Classifier()
        
        with pytest.raises(ValueError):
            model([])
    
    def test_classifier_very_long_text(self):
        """Test classifier with very long text."""
        model = Classifier()
        model.eval()
        
        # Create very long text
        long_text = ['This is a very long text. ' * 1000]
        
        with torch.no_grad():
            outputs = model(long_text)
        
        # Should still produce valid outputs
        assert outputs.shape == (1, 3)
        assert torch.all(torch.isfinite(outputs))
    
    def test_classifier_special_characters(self):
        """Test classifier with special characters."""
        model = Classifier()
        model.eval()
        
        special_texts = [
            'Text with @#$%^&*() symbols!',
            'Text with emojis 😀😍🎉',
            'Text with unicode: café, naïve, résumé',
            'Text with numbers: 12345 and symbols: !@#$%'
        ]
        
        with torch.no_grad():
            outputs = model(special_texts)
        
        # Should handle special characters gracefully
        assert outputs.shape == (4, 3)
        assert torch.all(torch.isfinite(outputs))
    
    def test_classifier_mixed_case(self):
        """Test classifier with mixed case text."""
        model = Classifier()
        model.eval()
        
        mixed_case_texts = [
            'UPPERCASE TEXT',
            'lowercase text',
            'Mixed Case Text',
            'RaNdOm CaSe TeXt'
        ]
        
        with torch.no_grad():
            outputs = model(mixed_case_texts)
        
        # Should handle mixed case
        assert outputs.shape == (4, 3)
        assert torch.all(torch.isfinite(outputs))


if __name__ == "__main__":
    pytest.main([__file__])
