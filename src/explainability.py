import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class HybridRecommendationExplainer:
    """
    Provides explainability for hybrid recommendation systems.
    
    Explains why a particular course was recommended by breaking down:
    - Content-based contribution
    - Collaborative filtering contribution
    - Feature importance
    - Similar users/items that influenced the recommendation
    
    Attributes:
        hybrid_model: HybridRecommender instance
        interactions_df: User-course interactions
        courses_df: Course metadata
    """
    
    def __init__(self, hybrid_model, interactions_df: pd.DataFrame, courses_df: pd.DataFrame):
        """
        Initialize explainer.
        
        Args:
            hybrid_model: HybridRecommender instance
            interactions_df: User-course interactions
            courses_df: Course metadata
        """
        self.hybrid_model = hybrid_model
        self.interactions_df = interactions_df
        self.courses_df = courses_df
        self.content_model = hybrid_model.content_model
        self.collab_model = hybrid_model.collab_model
    
    def explain_recommendation(self, user_id: str, course_id: str, top_n: int = 5) -> Dict:
        """
        Generate detailed explanation for why a course was recommended.
        
        Explanation includes:
        - Content-based reasoning (similar to courses user liked)
        - Collaborative reasoning (similar users who liked this course)
        - Overall hybrid score breakdown
        - Contributing factors
        
        Args:
            user_id: User identifier
            course_id: Course identifier
            top_n: Top N influencing factors to include
            
        Returns:
            Dictionary with detailed explanation
        """
        explanation = {
            'user_id': user_id,
            'course_id': course_id,
            'course_name': self.courses_df[self.courses_df['course_id'] == course_id]['course_name'].values[0],
            'recommendation_reason': '',
            'content_based_reasoning': self._explain_content_based(user_id, course_id, top_n),
            'collaborative_reasoning': self._explain_collaborative(user_id, course_id, top_n),
            'hybrid_score_breakdown': self._get_score_breakdown(user_id, course_id),
            'confidence_level': self._compute_confidence(user_id, course_id),
            'alternative_courses': self._get_alternatives(user_id, course_id, top_n)
        }
        
        # Generate natural language explanation
        explanation['recommendation_reason'] = self._generate_natural_explanation(explanation)
        
        return explanation
    
    def _explain_content_based(self, user_id: str, course_id: str, top_n: int = 5) -> Dict:
        """
        Explain content-based component of the recommendation.
        
        Args:
            user_id: User identifier
            course_id: Course identifier
            top_n: Top N similar courses to include
            
        Returns:
            Dictionary with content-based reasoning
        """
        # Get user's rated courses
        user_courses = self.interactions_df[self.interactions_df['user_id'] == user_id]
        
        if user_courses.empty:
            return {'reason': 'No user history available', 'similar_courses': []}
        
        # Find courses similar to this one that user has rated positively
        highly_rated_courses = user_courses[user_courses['rating'] >= 3.5]['course_id'].values
        
        similar_to_liked = []
        for liked_course in highly_rated_courses[:top_n]:
            # Compute similarity (simplified - would use actual similarity matrix in production)
            similarity_score = self._compute_course_similarity(liked_course, course_id)
            if similarity_score > 0:
                similar_to_liked.append({
                    'course_id': liked_course,
                    'course_name': self.courses_df[self.courses_df['course_id'] == liked_course]['course_name'].values[0],
                    'user_rating': float(user_courses[user_courses['course_id'] == liked_course]['rating'].values[0]),
                    'similarity_score': similarity_score
                })
        
        similar_to_liked = sorted(similar_to_liked, key=lambda x: x['similarity_score'], reverse=True)[:top_n]
        
        return {
            'reason': f'Similar to {len(similar_to_liked)} course(s) you rated highly',
            'similar_courses': similar_to_liked,
            'contribution': 'Content similarity'
        }
    
    def _explain_collaborative(self, user_id: str, course_id: str, top_n: int = 5) -> Dict:
        """
        Explain collaborative filtering component of the recommendation.
        
        Args:
            user_id: User identifier
            course_id: Course identifier
            top_n: Top N similar users to include
            
        Returns:
            Dictionary with collaborative reasoning
        """
        # Find similar users who rated this course highly
        course_ratings = self.interactions_df[self.interactions_df['course_id'] == course_id]
        
        if course_ratings.empty:
            return {'reason': 'No user ratings available for this course', 'similar_users': []}
        
        # Get highly-rated ratings
        high_ratings = course_ratings[course_ratings['rating'] >= 3.5].head(top_n)
        
        similar_users = []
        for _, rating in high_ratings.iterrows():
            similar_users.append({
                'user_id': rating['user_id'],
                'rating_given': float(rating['rating']),
                'similarity_reason': 'Similar learning preferences'
            })
        
        return {
            'reason': f'{len(similar_users)} similar user(s) rated this course highly',
            'similar_users': similar_users,
            'contribution': 'User-based collaboration'
        }
    
    def _compute_course_similarity(self, course_id_1: str, course_id_2: str) -> float:
        """
        Compute similarity between two courses based on metadata.
        
        Args:
            course_id_1: First course ID
            course_id_2: Second course ID
            
        Returns:
            Similarity score in [0, 1]
        """
        try:
            course_1 = self.courses_df[self.courses_df['course_id'] == course_id_1].iloc[0]
            course_2 = self.courses_df[self.courses_df['course_id'] == course_id_2].iloc[0]
            
            # Check if courses share difficulty level
            difficulty_match = 1.0 if course_1['difficulty_level'] == course_2['difficulty_level'] else 0.5
            
            # Check skill overlap
            skills_1 = set(str(course_1.get('skills', '')).split(','))
            skills_2 = set(str(course_2.get('skills', '')).split(','))
            skills_overlap = len(skills_1 & skills_2) / max(len(skills_1 | skills_2), 1)
            
            # Combined similarity
            similarity = 0.4 * difficulty_match + 0.6 * skills_overlap
            
            return similarity
        except:
            return 0.0
    
    def _get_score_breakdown(self, user_id: str, course_id: str) -> Dict:
        """
        Get breakdown of hybrid score components.
        
        Args:
            user_id: User identifier
            course_id: Course identifier
            
        Returns:
            Dictionary with score components
        """
        content_scores = self.hybrid_model.get_content_scores(
            user_id, self.interactions_df, self.courses_df, top_n=100
        )
        
        user_cf_scores, item_cf_scores = self.hybrid_model.get_collaborative_scores(
            user_id, self.courses_df, top_n=100
        )
        
        content_score = content_scores.get(course_id, 0.0)
        user_cf_score = user_cf_scores.get(course_id, 0.0)
        item_cf_score = item_cf_scores.get(course_id, 0.0)
        cf_score = np.mean([user_cf_score, item_cf_score]) if (user_cf_score + item_cf_score) > 0 else 0.0
        
        # Normalize to [0, 1]
        content_norm = min(content_score / 5.0, 1.0) if content_score > 0 else 0.0
        cf_norm = cf_score / 5.0 if cf_score > 0 else 0.0
        
        hybrid_score = self.hybrid_model.alpha * cf_norm + self.hybrid_model.beta * content_norm
        
        return {
            'content_based_score': content_norm,
            'collaborative_filtering_score': cf_norm,
            'hybrid_score': hybrid_score,
            'alpha_weight': self.hybrid_model.alpha,
            'beta_weight': self.hybrid_model.beta,
            'calculation': f"{self.hybrid_model.alpha:.2f} * {cf_norm:.4f} + {self.hybrid_model.beta:.2f} * {content_norm:.4f} = {hybrid_score:.4f}"
        }
    
    def _compute_confidence(self, user_id: str, course_id: str) -> float:
        """
        Compute confidence level in the recommendation.
        
        Factors:
        - User interaction history (more history = higher confidence)
        - Course rating availability (more ratings = higher confidence)
        - Score magnitude (higher score = higher confidence)
        
        Args:
            user_id: User identifier
            course_id: Course identifier
            
        Returns:
            Confidence score in [0, 1]
        """
        user_interactions = len(self.interactions_df[self.interactions_df['user_id'] == user_id])
        course_ratings = len(self.interactions_df[self.interactions_df['course_id'] == course_id])
        
        score_breakdown = self._get_score_breakdown(user_id, course_id)
        hybrid_score = score_breakdown['hybrid_score']
        
        # Normalize by interaction counts
        user_factor = min(user_interactions / 10.0, 1.0)
        course_factor = min(course_ratings / 50.0, 1.0)
        
        confidence = 0.4 * user_factor + 0.3 * course_factor + 0.3 * hybrid_score
        
        return min(confidence, 1.0)
    
    def _get_alternatives(self, user_id: str, course_id: str, top_n: int = 3) -> List[Dict]:
        """
        Get alternative course recommendations if user rejects this one.
        
        Args:
            user_id: User identifier
            course_id: Course identifier (to exclude)
            top_n: Number of alternatives
            
        Returns:
            List of alternative courses with scores
        """
        recommendations = self.hybrid_model.get_top_recommendations(
            user_id, self.interactions_df, self.courses_df, top_n=top_n + 1
        )
        
        # Exclude the current course
        alternatives = recommendations[recommendations['course_id'] != course_id].head(top_n)
        
        return [
            {
                'course_id': row['course_id'],
                'course_name': row['course_name'],
                'difficulty_level': row['difficulty_level'],
                'score': float(row['hybrid_score'])
            }
            for _, row in alternatives.iterrows()
        ]
    
    def _generate_natural_explanation(self, explanation: Dict) -> str:
        """
        Generate natural language explanation for the recommendation.
        
        Args:
            explanation: Explanation dictionary
            
        Returns:
            Natural language explanation string
        """
        parts = []
        
        # Content-based part
        content = explanation['content_based_reasoning']
        if content['similar_courses']:
            num_similar = len(content['similar_courses'])
            parts.append(f"This course is similar to {num_similar} course(s) you've rated highly")
        
        # Collaborative part
        collab = explanation['collaborative_reasoning']
        if collab['similar_users']:
            num_users = len(collab['similar_users'])
            parts.append(f"{num_users} users with similar learning preferences rated it highly")
        
        # Score breakdown
        breakdown = explanation['hybrid_score_breakdown']
        parts.append(f"Overall recommendation score: {breakdown['hybrid_score']:.2%}")
        
        return ". ".join(parts) + "."
    
    def generate_explanation_report(self, user_id: str, course_id: str) -> str:
        """
        Generate formatted explanation report.
        
        Args:
            user_id: User identifier
            course_id: Course identifier
            
        Returns:
            Formatted explanation report string
        """
        explanation = self.explain_recommendation(user_id, course_id)
        
        report = f"""
{'='*70}
RECOMMENDATION EXPLANATION REPORT
{'='*70}

User ID: {explanation['user_id']}
Course: {explanation['course_name']} ({explanation['course_id']})

Recommendation: {explanation['recommendation_reason']}

Confidence Level: {explanation['confidence_level']:.0%}

{'='*70}
SCORE BREAKDOWN
{'='*70}

Content-Based Score:      {explanation['hybrid_score_breakdown']['content_based_score']:.4f}
Collaborative Score:      {explanation['hybrid_score_breakdown']['collaborative_filtering_score']:.4f}
Hybrid Score:             {explanation['hybrid_score_breakdown']['hybrid_score']:.4f}

Calculation:
{explanation['hybrid_score_breakdown']['calculation']}

{'='*70}
CONTENT-BASED REASONING
{'='*70}

{explanation['content_based_reasoning']['reason']}

Similar Courses You Liked:
"""
        for course in explanation['content_based_reasoning']['similar_courses']:
            report += f"\n  • {course['course_name']} (Rating: {course['user_rating']:.1f}/5, Similarity: {course['similarity_score']:.2%})"
        
        report += f"""

{'='*70}
COLLABORATIVE FILTERING REASONING
{'='*70}

{explanation['collaborative_reasoning']['reason']}
"""
        
        if explanation['collaborative_reasoning']['similar_users']:
            report += "\nSimilar Users Who Liked This Course:\n"
            for user in explanation['collaborative_reasoning']['similar_users']:
                report += f"\n  • User {user['user_id']}: {user['rating_given']:.1f}/5"
        
        if explanation['alternative_courses']:
            report += f"""

{'='*70}
ALTERNATIVE RECOMMENDATIONS
{'='*70}
"""
            for i, alt in enumerate(explanation['alternative_courses'], 1):
                report += f"\n{i}. {alt['course_name']} ({alt['difficulty_level']}) - Score: {alt['score']:.4f}"
        
        report += f"\n{'='*70}\n"
        
        return report


class ExplainabilityDashboard:
    """
    Generates summary statistics and insights about recommendation explainability.
    """
    
    def __init__(self, explainer: HybridRecommendationExplainer):
        """
        Initialize dashboard.
        
        Args:
            explainer: HybridRecommendationExplainer instance
        """
        self.explainer = explainer
    
    def get_explanation_statistics(self, user_ids: List[str], top_n: int = 5) -> Dict:
        """
        Generate statistics about explanations across multiple users.
        
        Args:
            user_ids: List of user IDs to analyze
            top_n: Top N recommendations per user to explain
            
        Returns:
            Dictionary with explanation statistics
        """
        confidence_scores = []
        content_contributions = []
        collab_contributions = []
        
        for user_id in user_ids:
            recommendations = self.explainer.hybrid_model.get_top_recommendations(
                user_id, self.explainer.interactions_df, self.explainer.courses_df, top_n=top_n
            )
            
            for _, rec in recommendations.iterrows():
                explanation = self.explainer.explain_recommendation(user_id, rec['course_id'])
                confidence_scores.append(explanation['confidence_level'])
                
                breakdown = explanation['hybrid_score_breakdown']
                content_contributions.append(breakdown['content_based_score'])
                collab_contributions.append(breakdown['collaborative_filtering_score'])
        
        return {
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'avg_content_contribution': np.mean(content_contributions) if content_contributions else 0.0,
            'avg_collaborative_contribution': np.mean(collab_contributions) if collab_contributions else 0.0,
            'total_explanations': len(confidence_scores),
            'confidence_distribution': {
                'min': float(np.min(confidence_scores)) if confidence_scores else 0.0,
                'max': float(np.max(confidence_scores)) if confidence_scores else 0.0,
                'median': float(np.median(confidence_scores)) if confidence_scores else 0.0,
                'std': float(np.std(confidence_scores)) if confidence_scores else 0.0
            }
        }
