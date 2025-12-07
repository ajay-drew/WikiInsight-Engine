"""
SHAP explainer for models used in the WikiInsight Engine (e.g., clustering prototypes or auxiliary models).
"""

import shap
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP explainer for model predictions."""
    
    def __init__(self, model, X_background: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (XGBoost, LightGBM, or LogisticRegression)
            X_background: Background dataset for SHAP (sample of training data)
            feature_names: Optional list of feature names
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names
        
        # Create appropriate SHAP explainer based on model type
        model_type = type(model).__name__.lower()
        
        if 'xgboost' in model_type or 'xgb' in model_type:
            self.explainer = shap.TreeExplainer(model)
        elif 'lightgbm' in model_type or 'lgb' in model_type:
            self.explainer = shap.TreeExplainer(model)
        elif 'logistic' in model_type or 'linear' in model_type:
            self.explainer = shap.LinearExplainer(model, X_background)
        else:
            # Fallback to KernelExplainer
            logger.warning("Using KernelExplainer (slower). Consider using TreeExplainer for tree models.")
            self.explainer = shap.KernelExplainer(model.predict_proba, X_background)
        
        logger.info(f"Initialized SHAP explainer for {model_type}")
    
    def explain_instance(self, X_instance: np.ndarray) -> Dict:
        """
        Explain a single instance prediction.
        
        Args:
            X_instance: Single instance feature vector
            
        Returns:
            Dictionary with SHAP values and explanation
        """
        shap_values = self.explainer.shap_values(X_instance)
        
        # Handle multi-output models
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get values for positive class (gap)
        
        return {
            "shap_values": shap_values,
            "base_value": self.explainer.expected_value,
            "feature_names": self.feature_names
        }
    
    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Explain a batch of predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            SHAP values for all instances
        """
        shap_values = self.explainer.shap_values(X)
        
        # Handle multi-output models
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get values for positive class
        
        return shap_values
    
    def get_feature_importance(self, X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get global feature importance from SHAP.
        
        Args:
            X: Optional dataset to compute importance on (uses background if None)
            
        Returns:
            Dictionary of feature names to importance scores
        """
        if X is None:
            X = self.X_background
        
        shap_values = self.explain_batch(X)
        
        # Average absolute SHAP values across instances
        importance = np.abs(shap_values).mean(axis=0)
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        return dict(enumerate(importance))
    
    def get_waterfall_data(self, X_instance: np.ndarray) -> Dict:
        """
        Get data for SHAP waterfall plot.
        
        Args:
            X_instance: Single instance feature vector
            
        Returns:
            Dictionary with waterfall plot data
        """
        explanation = self.explain_instance(X_instance)
        
        # Sort features by absolute SHAP value
        shap_vals = explanation["shap_values"].flatten()
        feature_importance = np.abs(shap_vals)
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        return {
            "shap_values": shap_vals[sorted_indices],
            "feature_names": [self.feature_names[i] if self.feature_names else f"Feature_{i}" 
                            for i in sorted_indices],
            "feature_values": X_instance.flatten()[sorted_indices],
            "base_value": explanation["base_value"]
        }

