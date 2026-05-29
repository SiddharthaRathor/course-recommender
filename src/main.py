"""
Comprehensive Hybrid Course Recommendation System Pipeline

This module orchestrates the complete recommendation workflow:
1. Data Loading & Preprocessing
2. Content-Based Filtering
3. Collaborative Filtering
4. Hybrid Recommendation Generation
5. Cold Start Handling
6. Evaluation & Metrics
7. Visualization & Reporting
8. Explainability Analysis

Author: ML Team
Version: 2.0
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preprocessing import create_sample_dataset
from synthetic_data import create_interaction_dataset
from content_model import ContentBasedRecommender
from collaborative_model import CollaborativeFilteringRecommender
from src.hybrid_recommender import HybridRecommender
from src.cold_start import ColdStartHandler
from src.evaluation_metrics import EvaluationMetrics, RecommendationEvaluator
from src.visualizations import RecommendationVisualizer
from src.explainability import HybridRecommendationExplainer, ExplainabilityDashboard


class HybridRecommendationPipeline:
    """
    Main pipeline orchestrating the complete recommendation system.
    """
    
    def __init__(self, output_dir: str = 'outputs', config: Dict = None):
        """
        Initialize the recommendation pipeline.
        
        Args:
            output_dir: Directory to save outputs
            config: Optional configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            'num_users': 750,
            'interaction_sparsity': 0.15,
            'test_size': 0.2,
            'top_n_recommendations': 5,
            'alpha': 0.7,  # CF weight
            'beta': 0.3,   # Content weight
            'min_interactions_for_cold_start': 2,
            'k_values_for_evaluation': [1, 5, 10],
            'visualization_dpi': 300
        }
        
        if config:
            self.config.update(config)
        
        # Initialize components
        self.courses_df = None
        self.users_df = None
        self.interactions_df = None
        self.content_model = None
        self.collab_model = None
        self.hybrid_model = None
        self.cold_start_handler = None
        self.evaluator = None
        self.visualizer = None
        self.explainer = None
        
        # Results storage
        self.results = {}
        
    def run_full_pipeline(self, verbose: bool = True) -> Dict:
        """
        Execute the complete recommendation system pipeline.
        
        Args:
            verbose: Print progress information
            
        Returns:
            Dictionary with complete results
        """
        print("\n" + "="*80)
        print("HYBRID COURSE RECOMMENDATION SYSTEM - FULL PIPELINE")
        print("="*80 + "\n")
        
        # Step 1: Load and prepare data
        if verbose:
            print("[1/8] Loading and preparing data...")
        self._load_data(verbose)
        
        # Step 2: Build content-based model
        if verbose:
            print("\n[2/8] Building content-based recommendation model...")
        self._build_content_model(verbose)
        
        # Step 3: Build collaborative filtering model
        if verbose:
            print("\n[3/8] Building collaborative filtering model...")
        self._build_collaborative_model(verbose)
        
        # Step 4: Build hybrid model
        if verbose:
            print("\n[4/8] Building hybrid recommendation model...")
        self._build_hybrid_model(verbose)
        
        # Step 5: Handle cold start scenarios
        if verbose:
            print("\n[5/8] Initializing cold start handler...")
        self._init_cold_start_handler(verbose)
        
        # Step 6: Generate recommendations
        if verbose:
            print("\n[6/8] Generating recommendations...")
        recommendations = self._generate_recommendations(verbose)
        
        # Step 7: Evaluate performance
        if verbose:
            print("\n[7/8] Evaluating recommendation performance...")
        metrics = self._evaluate_recommendations(recommendations, verbose)
        
        # Step 8: Generate visualizations and reports
        if verbose:
            print("\n[8/8] Generating visualizations and reports...")
        self._generate_reports_and_visualizations(metrics, recommendations, verbose)
        
        # Summary
        self._print_summary(metrics)
        
        return self.results
    
    def _load_data(self, verbose: bool = True):
        """Load course and interaction data."""
        if verbose:
            print("   - Loading course dataset...")
        self.courses_df = create_sample_dataset()
        
        if verbose:
            print(f"     ✓ Loaded {len(self.courses_df)} courses")
            print("   - Generating synthetic users and interactions...")
        
        self.users_df, _, self.interactions_df = create_interaction_dataset(
            None,
            num_users=self.config['num_users'],
            interaction_sparsity=self.config['interaction_sparsity']
        )
        
        if verbose:
            print(f"     ✓ Generated {len(self.users_df)} users")
            print(f"     ✓ Generated {len(self.interactions_df)} interactions")
            print(f"     ✓ Sparsity: {len(self.interactions_df) / (len(self.users_df) * len(self.courses_df)):.2%}")
    
    def _build_content_model(self, verbose: bool = True):
        """Build and fit content-based recommender."""
        try:
            self.content_model = ContentBasedRecommender(self.courses_df)
            if verbose:
                print("     ✓ Content-based model built successfully")
        except Exception as e:
            if verbose:
                print(f"     ✗ Error building content model: {str(e)}")
            raise
    
    def _build_collaborative_model(self, verbose: bool = True):
        """Build and fit collaborative filtering models."""
        try:
            self.collab_model = CollaborativeFilteringRecommender(self.interactions_df)
            if verbose:
                print("     ✓ User-based CF model trained")
                print("     ✓ Item-based CF model trained")
        except Exception as e:
            if verbose:
                print(f"     ✗ Error building collaborative model: {str(e)}")
            raise
    
    def _build_hybrid_model(self, verbose: bool = True):
        """Build hybrid recommendation model."""
        try:
            self.hybrid_model = HybridRecommender(
                self.content_model,
                self.collab_model,
                alpha=self.config['alpha'],
                beta=self.config['beta']
            )
            if verbose:
                print(f"     ✓ Hybrid model built with weights (α={self.config['alpha']}, β={self.config['beta']})")
        except Exception as e:
            if verbose:
                print(f"     ✗ Error building hybrid model: {str(e)}")
            raise
    
    def _init_cold_start_handler(self, verbose: bool = True):
        """Initialize cold start handler."""
        try:
            self.cold_start_handler = ColdStartHandler(
                self.courses_df,
                self.interactions_df,
                min_interactions=self.config['min_interactions_for_cold_start']
            )
            if verbose:
                print("     ✓ Cold start handler initialized")
        except Exception as e:
            if verbose:
                print(f"     ✗ Error initializing cold start handler: {str(e)}")
            raise
    
    def _generate_recommendations(self, verbose: bool = True) -> Dict[str, List[str]]:
        """Generate recommendations for all users."""
        recommendations = {}
        failed_users = 0
        cold_start_users = 0
        
        sample_size = min(100, len(self.users_df))
        sample_indices = np.random.choice(len(self.users_df), sample_size, replace=False)
        
        for idx, user_idx in enumerate(sample_indices):
            user_id = self.users_df.iloc[user_idx]['user_id']
            
            try:
                # Check if cold start
                if self.cold_start_handler.is_cold_start_user(user_id):
                    cold_start_users += 1
                    recs = self.cold_start_handler.recommend_for_new_user(
                        user_id, self.config['top_n_recommendations'], strategy='hybrid'
                    )
                    recommendations[user_id] = recs['course_id'].tolist()
                else:
                    # Use hybrid model
                    recs = self.hybrid_model.get_top_recommendations(
                        user_id, self.interactions_df, self.courses_df,
                        top_n=self.config['top_n_recommendations']
                    )
                    recommendations[user_id] = recs['course_id'].tolist()
            
            except Exception as e:
                if verbose and idx < 3:
                    print(f"     ! Error for user {user_id}: {str(e)}")
                failed_users += 1
                recommendations[user_id] = []
        
        if verbose:
            print(f"     ✓ Generated recommendations for {len(recommendations)} users")
            print(f"     ✓ Cold start handled: {cold_start_users} users")
            if failed_users > 0:
                print(f"     ! Failed for {failed_users} users")
        
        return recommendations
    
    def _evaluate_recommendations(self, recommendations: Dict[str, List[str]], 
                                 verbose: bool = True) -> Dict:
        """Evaluate recommendation quality."""
        try:
            # Create evaluation metrics instance
            evaluator = EvaluationMetrics(self.interactions_df)
            
            # Evaluate at different K values
            metrics_by_k = {}
            for k in self.config['k_values_for_evaluation']:
                metrics = evaluator.evaluate_recommendations(recommendations, k=k)
                metrics_by_k[k] = metrics
            
            # Flatten metrics for storage
            flat_metrics = {}
            for k, metrics in metrics_by_k.items():
                for metric_name, value in metrics.items():
                    flat_metrics[f'{metric_name}@{k}'] = value
            
            if verbose:
                print("     ✓ Recommendation metrics calculated:")
                for k in sorted(metrics_by_k.keys()):
                    print(f"\n       Metrics @K={k}:")
                    for metric_name, value in metrics_by_k[k].items():
                        print(f"         • {metric_name}: {value:.4f}")
            
            self.results['metrics'] = flat_metrics
            return flat_metrics
        
        except Exception as e:
            if verbose:
                print(f"     ✗ Error in evaluation: {str(e)}")
            return {}
    
    def _generate_reports_and_visualizations(self, metrics: Dict, 
                                            recommendations: Dict[str, List[str]],
                                            verbose: bool = True):
        """Generate visualizations and reports."""
        try:
            # Initialize visualizer
            self.visualizer = RecommendationVisualizer(
                output_dir=str(self.output_dir),
                dpi=self.config['visualization_dpi']
            )
            
            # Generate visualizations
            if verbose:
                print("     - Generating metrics visualizations...")
            
            if metrics:
                viz_files = []
                
                # Plot metrics at K
                if any(k in str(m) for m in metrics.keys() for k in ['1', '5', '10']):
                    viz_files.append(
                        self.visualizer.plot_metrics_at_k(metrics, system_name="Hybrid System")
                    )
                
                # Generate summary report
                viz_files.append(
                    self.visualizer.generate_summary_report(metrics, system_name="Hybrid System")
                )
                
                if verbose:
                    print(f"     ✓ Generated {len(viz_files)} visualization files")
                    for vf in viz_files:
                        print(f"       • {vf}")
            
            # Generate explainability analysis
            if verbose:
                print("     - Generating explainability analysis...")
            
            self.explainer = HybridRecommendationExplainer(
                self.hybrid_model, self.interactions_df, self.courses_df
            )
            
            # Sample users for explanation
            sample_users = self.users_df.sample(min(5, len(self.users_df)))['user_id'].tolist()
            
            explanations = []
            for user_id in sample_users:
                if user_id in recommendations and recommendations[user_id]:
                    course_id = recommendations[user_id][0]
                    explanation = self.explainer.explain_recommendation(user_id, course_id)
                    explanations.append(explanation)
            
            # Save explanations
            explanations_file = self.output_dir / 'explanations.json'
            with open(explanations_file, 'w') as f:
                json.dump(explanations, f, indent=2, default=str)
            
            if verbose:
                print(f"     ✓ Generated {len(explanations)} explanations")
                print(f"       • Saved to {explanations_file}")
            
            self.results['visualizations_dir'] = str(self.output_dir)
            self.results['explanations_file'] = str(explanations_file)
        
        except Exception as e:
            if verbose:
                print(f"     ✗ Error generating visualizations: {str(e)}")
    
    def _print_summary(self, metrics: Dict):
        """Print execution summary."""
        print("\n" + "="*80)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*80)
        
        print(f"\nData:")
        print(f"  • Courses: {len(self.courses_df)}")
        print(f"  • Users: {len(self.users_df)}")
        print(f"  • Interactions: {len(self.interactions_df)}")
        print(f"  • Sparsity: {len(self.interactions_df) / (len(self.users_df) * len(self.courses_df)):.2%}")
        
        print(f"\nHybrid Model Configuration:")
        print(f"  • Collaborative Filtering Weight (α): {self.config['alpha']}")
        print(f"  • Content-Based Weight (β): {self.config['beta']}")
        
        if metrics:
            print(f"\nRecommendation Quality Metrics:")
            for metric_name, value in sorted(metrics.items()):
                print(f"  • {metric_name}: {value:.4f}")
        
        print(f"\nOutput Files:")
        print(f"  • Directory: {self.output_dir}")
        for file_path in self.output_dir.glob("*"):
            print(f"    - {file_path.name}")
        
        print("\n" + "="*80)
        print(f"Pipeline execution completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
    
    def get_recommendations_for_user(self, user_id: str, top_n: int = 5,
                                    include_explanation: bool = True) -> Dict:
        """
        Get recommendations for a specific user with optional explanations.
        
        Args:
            user_id: User identifier
            top_n: Number of recommendations
            include_explanation: Include detailed explanations
            
        Returns:
            Dictionary with recommendations and explanations
        """
        # Check if cold start
        if self.cold_start_handler.is_cold_start_user(user_id):
            recs = self.cold_start_handler.recommend_for_new_user(user_id, top_n, strategy='hybrid')
            recommendations = recs.to_dict('records')
        else:
            recs = self.hybrid_model.get_top_recommendations(
                user_id, self.interactions_df, self.courses_df, top_n=top_n
            )
            recommendations = recs.to_dict('records')
        
        result = {
            'user_id': user_id,
            'recommendations': recommendations,
            'include_explanation': include_explanation
        }
        
        if include_explanation and self.explainer:
            explanations = []
            for rec in recommendations[:3]:  # Explain top 3
                exp = self.explainer.explain_recommendation(user_id, rec['course_id'])
                explanations.append(exp)
            
            result['explanations'] = explanations
        
        return result


def main():
    """Main execution function."""
    
    # Configuration
    config = {
        'num_users': 500,
        'interaction_sparsity': 0.15,
        'top_n_recommendations': 5,
        'alpha': 0.7,
        'beta': 0.3,
        'k_values_for_evaluation': [1, 5, 10]
    }
    
    # Initialize and run pipeline
    pipeline = HybridRecommendationPipeline(output_dir='outputs', config=config)
    results = pipeline.run_full_pipeline(verbose=True)
    
    # Demonstrate user-specific recommendations
    print("\n" + "="*80)
    print("SAMPLE USER RECOMMENDATIONS WITH EXPLANATIONS")
    print("="*80 + "\n")
    
    sample_users = pipeline.users_df.sample(min(3, len(pipeline.users_df)))
    
    for _, user in sample_users.iterrows():
        user_id = user['user_id']
        user_recs = pipeline.get_recommendations_for_user(user_id, top_n=3, include_explanation=True)
        
        print(f"\nUser: {user_id}")
        print(f"Preference: {user['preferred_skill']} | Difficulty: {user['preferred_difficulty']}\n")
        
        for i, rec in enumerate(user_recs['recommendations'][:3], 1):
            print(f"{i}. {rec['course_name']} (Score: {rec.get('hybrid_score', rec.get('score', 0)):.4f})")
        
        if user_recs.get('explanations'):
            print(f"\nTop Recommendation Explanation:")
            print(user_recs['explanations'][0]['recommendation_reason'])
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
