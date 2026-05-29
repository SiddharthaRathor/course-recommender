"""
MovieLens 100K Dataset Preprocessing

This module converts MovieLens 100K data (movies.dat, ratings.dat) into:
- courses.csv: Course metadata (id, name, skills/genres, difficulty level, description)
- ratings.csv: User-course interactions (user_id, course_id, rating, timestamp)

MovieLens Dataset Structure:
- ml-100k/u.data: Tab-separated user_id, item_id, rating, timestamp
- ml-100k/u.item: Pipe-separated movie info including genres

Author: Course Recommender System
Version: 1.0
"""

import pandas as pd
import numpy as np
import os
import zipfile
from pathlib import Path
from typing import Tuple, Optional


class MovieLensPreprocessor:
    """
    Convert MovieLens 100K dataset into course recommendation format.
    """
    
    # MovieLens dataset file names
    RATINGS_FILE = 'u.data'
    MOVIES_FILE = 'u.item'
    GENRES_FILE = 'u.genre'
    
    # Genre mapping to course skills
    GENRE_TO_SKILL = {
        'Action': 'Action Films',
        'Adventure': 'Adventure',
        'Animation': 'Animation',
        'Children\'s': 'Children Content',
        'Comedy': 'Comedy',
        'Crime': 'Crime Drama',
        'Documentary': 'Documentary',
        'Drama': 'Drama',
        'Fantasy': 'Fantasy',
        'Film-Noir': 'Film Noir',
        'Horror': 'Horror',
        'Musical': 'Musical',
        'Mystery': 'Mystery',
        'Romance': 'Romance',
        'Sci-Fi': 'Science Fiction',
        'Thriller': 'Thriller',
        'War': 'War Films',
        'Western': 'Western'
    }
    
    # Difficulty mapping based on ratings
    DIFFICULTY_MAPPING = {
        'very_easy': 'Beginner',      # avg_rating >= 4.5
        'easy': 'Beginner',            # avg_rating >= 4.0
        'medium': 'Intermediate',      # avg_rating >= 3.0
        'hard': 'Advanced',            # avg_rating >= 2.0
        'very_hard': 'Advanced'        # avg_rating < 2.0
    }
    
    def __init__(self, dataset_path: str = 'ml-100k'):
        """
        Initialize preprocessor with dataset path.
        
        Args:
            dataset_path: Path to ml-100k directory or zip file
        """
        self.dataset_path = Path(dataset_path)
        self.data_dir = None
        self.ratings_df = None
        self.movies_df = None
        self.courses_df = None
        self.interactions_df = None
        
    def extract_dataset(self) -> bool:
        """
        Extract MovieLens dataset from zip if needed.
        
        Returns:
            True if extraction successful or directory already exists
        """
        # Check if directory already exists
        if self.dataset_path.is_dir():
            self.data_dir = self.dataset_path
            print(f"✓ Dataset directory found: {self.dataset_path}")
            return True
        
        # Check for zip file
        zip_path = Path(str(self.dataset_path) + '.zip')
        if zip_path.exists():
            print(f"Extracting dataset from {zip_path}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall()
                self.data_dir = self.dataset_path
                print(f"✓ Dataset extracted successfully")
                return True
            except Exception as e:
                print(f"✗ Error extracting dataset: {str(e)}")
                return False
        
        print(f"✗ Dataset not found at {self.dataset_path} or {zip_path}")
        return False
    
    def load_ratings(self) -> pd.DataFrame:
        """
        Load ratings from u.data file.
        
        Format: user_id, movie_id, rating, timestamp
        
        Returns:
            DataFrame with ratings data
        """
        ratings_path = self.data_dir / self.RATINGS_FILE
        
        if not ratings_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
        
        print(f"Loading ratings from {ratings_path}...")
        
        # Read tab-separated ratings file
        ratings = pd.read_csv(
            ratings_path,
            sep='\t',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        
        # Convert user_id and movie_id to string format for consistency
        ratings['user_id'] = 'U' + ratings['user_id'].astype(str).str.zfill(5)
        ratings['movie_id'] = 'C' + ratings['movie_id'].astype(str).str.zfill(4)
        
        # Rename movie_id to course_id for consistency
        ratings = ratings.rename(columns={'movie_id': 'course_id'})
        
        print(f"✓ Loaded {len(ratings)} ratings from {len(ratings['user_id'].unique())} users")
        
        self.ratings_df = ratings
        return ratings
    
    def load_movies(self) -> pd.DataFrame:
        """
        Load movie metadata from u.item file.
        
        Returns:
            DataFrame with movie information
        """
        movies_path = self.data_dir / self.MOVIES_FILE
        
        if not movies_path.exists():
            raise FileNotFoundError(f"Movies file not found: {movies_path}")
        
        print(f"Loading movie metadata from {movies_path}...")
        
        # Read pipe-separated movies file
        # Columns: movie_id, title, release_date, video_release_date, imdb_url,
        #          unknown, action, adventure, animation, children's, comedy, crime,
        #          documentary, drama, fantasy, film-noir, horror, musical, mystery,
        #          romance, sci-fi, thriller, war, western
        
        movies = pd.read_csv(
            movies_path,
            sep='|',
            header=None,
            encoding='latin-1',
            usecols=range(24)  # We only need up to the genre columns
        )
        
        # Set column names
        column_names = [
            'movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'
        ]
        genre_names = list(self.GENRE_TO_SKILL.keys())
        column_names.extend(genre_names)
        
        movies.columns = column_names[:len(movies.columns)]
        
        # Convert movie_id to string format
        movies['movie_id'] = 'C' + movies['movie_id'].astype(str).str.zfill(4)
        
        # Extract genres
        genre_columns = genre_names
        movies['genres'] = movies[genre_columns].apply(
            lambda row: ', '.join([self.GENRE_TO_SKILL[genre] 
                                  for genre in genre_columns 
                                  if row[genre] == 1]), axis=1
        )
        
        # Extract year from release date
        movies['release_year'] = pd.to_datetime(
            movies['release_date'], format='%d-%b-%Y', errors='coerce'
        ).dt.year
        
        # Clean title (remove year)
        movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
        
        # Select relevant columns
        movies = movies[['movie_id', 'title', 'genres', 'release_year']]
        
        print(f"✓ Loaded metadata for {len(movies)} movies")
        
        self.movies_df = movies
        return movies
    
    def create_courses_dataset(self) -> pd.DataFrame:
        """
        Create course dataset from movies with enhanced metadata.
        
        Returns:
            DataFrame with course information
        """
        if self.movies_df is None or self.ratings_df is None:
            raise ValueError("Movies and ratings data must be loaded first")
        
        print("Creating courses dataset...")
        
        # Get rating statistics per movie
        rating_stats = self.ratings_df.groupby('course_id').agg({
            'rating': ['mean', 'std', 'count', 'min', 'max']
        }).reset_index()
        
        rating_stats.columns = ['course_id', 'avg_rating', 'rating_std', 
                               'num_ratings', 'min_rating', 'max_rating']
        
        # Merge movies with rating statistics
        courses = self.movies_df.merge(rating_stats, on='course_id', how='left')
        
        # Fill missing rating stats for movies with no ratings
        courses['avg_rating'] = courses['avg_rating'].fillna(3.0)
        courses['num_ratings'] = courses['num_ratings'].fillna(0)
        
        # Create difficulty level based on average rating
        def map_difficulty(avg_rating):
            if avg_rating >= 4.5:
                return 'Beginner'
            elif avg_rating >= 4.0:
                return 'Beginner'
            elif avg_rating >= 3.0:
                return 'Intermediate'
            elif avg_rating >= 2.0:
                return 'Advanced'
            else:
                return 'Advanced'
        
        courses['difficulty_level'] = courses['avg_rating'].apply(map_difficulty)
        
        # Create course descriptions
        def create_description(row):
            return (f"This course covers {row['title']}. "
                   f"It includes topics on {row['genres']}. "
                   f"Released in {row['release_year']}. "
                   f"Average rating: {row['avg_rating']:.2f}/5 from {int(row['num_ratings'])} users.")
        
        courses['course_description'] = courses.apply(create_description, axis=1)
        
        # Rename columns for consistency
        courses = courses.rename(columns={
            'movie_id': 'course_id',
            'title': 'course_name',
            'genres': 'skills'
        })
        
        # Select and order columns
        courses = courses[[
            'course_id', 'course_name', 'course_description', 
            'skills', 'difficulty_level', 'avg_rating', 'num_ratings'
        ]]
        
        print(f"✓ Created course dataset with {len(courses)} courses")
        
        self.courses_df = courses
        return courses
    
    def create_interactions_dataset(self) -> pd.DataFrame:
        """
        Convert ratings to interactions dataset.
        
        Returns:
            DataFrame with user-course interactions
        """
        if self.ratings_df is None:
            raise ValueError("Ratings data must be loaded first")
        
        print("Creating interactions dataset...")
        
        # Create interactions from ratings
        interactions = self.ratings_df[['user_id', 'course_id', 'rating', 'timestamp']].copy()
        
        print(f"✓ Created interactions dataset with {len(interactions)} ratings")
        
        self.interactions_df = interactions
        return interactions
    
    def save_datasets(self, output_dir: str = '.', 
                     save_courses: bool = True, 
                     save_ratings: bool = True) -> Tuple[str, str]:
        """
        Save courses and ratings datasets to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
            save_courses: Whether to save courses.csv
            save_ratings: Whether to save ratings.csv
            
        Returns:
            Tuple of (courses_path, ratings_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        courses_path = None
        ratings_path = None
        
        if save_courses and self.courses_df is not None:
            courses_path = output_path / 'courses.csv'
            self.courses_df.to_csv(courses_path, index=False)
            print(f"✓ Saved courses to {courses_path}")
        
        if save_ratings and self.interactions_df is not None:
            ratings_path = output_path / 'ratings.csv'
            self.interactions_df.to_csv(ratings_path, index=False)
            print(f"✓ Saved ratings to {ratings_path}")
        
        return str(courses_path), str(ratings_path)
    
    def process(self, output_dir: str = '.') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            output_dir: Directory to save output files
            
        Returns:
            Tuple of (courses_df, interactions_df)
        """
        print("\n" + "="*60)
        print("MOVIELENS 100K PREPROCESSING")
        print("="*60 + "\n")
        
        # Extract dataset if needed
        if not self.extract_dataset():
            raise RuntimeError("Failed to extract/find dataset")
        
        # Load raw data
        self.load_ratings()
        self.load_movies()
        
        # Create processed datasets
        self.create_courses_dataset()
        self.create_interactions_dataset()
        
        # Save to CSV
        self.save_datasets(output_dir)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60 + "\n")
        
        return self.courses_df, self.interactions_df


def load_courses_csv(csv_path: str = 'courses.csv') -> pd.DataFrame:
    """
    Load courses dataset from CSV file.
    
    Args:
        csv_path: Path to courses.csv
        
    Returns:
        DataFrame with course information
    """
    courses = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(courses)} courses from {csv_path}")
    return courses


def load_ratings_csv(csv_path: str = 'ratings.csv') -> pd.DataFrame:
    """
    Load interactions dataset from CSV file.
    
    Args:
        csv_path: Path to ratings.csv
        
    Returns:
        DataFrame with user-course interactions
    """
    interactions = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(interactions)} interactions from {csv_path}")
    return interactions


def load_movielens_data(dataset_path: str = 'ml-100k', 
                       output_dir: str = '.',
                       use_csv: bool = False,
                       csv_courses: str = 'courses.csv',
                       csv_ratings: str = 'ratings.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens data, either from raw dataset or preprocessed CSV.
    
    Args:
        dataset_path: Path to ml-100k directory
        output_dir: Directory to save preprocessed CSV files
        use_csv: If True, load from CSV files instead of processing
        csv_courses: Path to courses.csv if use_csv=True
        csv_ratings: Path to ratings.csv if use_csv=True
        
    Returns:
        Tuple of (courses_df, interactions_df)
    """
    if use_csv:
        # Load from existing CSV files
        print("Loading from CSV files...")
        courses_df = load_courses_csv(csv_courses)
        interactions_df = load_ratings_csv(csv_ratings)
        return courses_df, interactions_df
    else:
        # Process raw MovieLens dataset
        preprocessor = MovieLensPreprocessor(dataset_path)
        return preprocessor.process(output_dir)


if __name__ == '__main__':
    """
    Example usage: Convert MovieLens 100K to course recommendation format
    
    Usage:
        python movielens_preprocessing.py
    
    This will:
        1. Extract ml-100k.zip if it exists
        2. Load movies and ratings data
        3. Create courses.csv and ratings.csv
        4. Save to current directory
    """
    # Process MovieLens dataset
    courses_df, interactions_df = load_movielens_data()
    
    # Display statistics
    print("\nDataset Statistics:")
    print(f"  Courses: {len(courses_df)}")
    print(f"  Interactions: {len(interactions_df)}")
    print(f"  Users: {interactions_df['user_id'].nunique()}")
    print(f"  Sparsity: {len(interactions_df) / (len(courses_df) * interactions_df['user_id'].nunique()):.2%}")
    
    print("\nSample Courses:")
    print(courses_df[['course_id', 'course_name', 'difficulty_level', 'skills']].head())
    
    print("\nSample Interactions:")
    print(interactions_df.head())
