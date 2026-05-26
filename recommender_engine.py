import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess_data(filepath):
    """
    Ingests the CSV file and cleans the skill strings.
    """
    print(f"Ingesting dataset from: {filepath}...")
    df = pd.read_csv(filepath)
    
    def clean_skills(skills_str):
        if pd.isna(skills_str):
            return ""
        try:
            # Safely evaluate string representation of a Python list
            skills_list = ast.literal_eval(skills_str)
            cleaned = [s.strip().lower() for s in skills_list]
            return " ".join(cleaned)
        except (ValueError, SyntaxError):
            # Fallback if the data format varies from standard Python list literals
            return str(skills_str).replace("[", "").replace("]", "").replace("'", "").replace(",", " ").lower().strip()

    df['cleaned_skills'] = df['raw_skills'].apply(clean_skills)
    return df

def recommend_jobs(user_skills, df, top_n=5):
    """
    Computes TF-IDF similarity scores and recommends the top N matching jobs.
    """
    # Normalize and format user skills into a single space-separated string
    user_profile = " ".join([s.strip().lower() for s in user_skills])
    
    # Combine job skill corpuses and the user profile to build the TF-IDF vocabulary
    all_profiles = df['cleaned_skills'].tolist() + [user_profile]
    
    # Vectorize text profiles
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_profiles)
    
    # Extract vectors
    job_vectors = tfidf_matrix[:-1] # All rows except the last one
    user_vector = tfidf_matrix[-1]  # The final row represents the user profile
    
    # Compute Cosine Similarity between user profile and all jobs
    similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()
    
    # Add scores back to a copy of the dataframe
    recommended_df = df.copy()
    recommended_df['similarity_score'] = similarity_scores
    
    # Sort by highest score and pick top N
    top_recommendations = recommended_df.sort_values(by='similarity_score', ascending=False).head(top_n)
    
    return top_recommendations[['id', 'raw_skills', 'url', 'similarity_score']]

if __name__ == "__main__":
    # 1. Ingest dataset
    csv_file = 'raw_skills.csv'
    df_jobs = load_and_preprocess_data(csv_file)
    
    # 2. Define default user details
    default_user_skills = ['python', 'machine learning', 'data science', 'tensorflow', 'nlp']
    print(f"\nDefault User Skills Profile:\n  {default_user_skills}\n")
    
    # 3. Compute Top 5 job recommendations
    top_matches = recommend_jobs(user_skills=default_user_skills, df=df_jobs, top_n=5)
    
    # 4. Display Results
    print("=" * 60)
    print("                     TOP JOB RECOMMENDATIONS                   ")
    print("=" * 60)
    for index, row in top_matches.iterrows():
        print(f"Rank/Job ID     : {row['id']}")
        print(f"Match Score     : {row['similarity_score']:.4%}")
        print(f"Required Skills : {row['raw_skills']}")
        print(f"Job URL         : {row['url']}")
        print("-" * 60)
        