import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to scrape content from the website
def get_website_content(url):
    # Sending a request to the website and getting the response
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extracting all the paragraph texts
    paragraphs = soup.find_all('p')
    content = [para.get_text() for para in paragraphs if para.get_text()]  # Cleaning up empty paragraphs
    return content

# Function to convert the extracted content into numerical data using TF-IDF
def generate_tfidf_matrix(content):
    vectorizer = TfidfVectorizer(stop_words='english')  # Remove common words that don't add much meaning
    tfidf_matrix = vectorizer.fit_transform(content)  # Create the TF-IDF matrix from content
    return tfidf_matrix, vectorizer

# Function to transform the user's query into the same TF-IDF format
def convert_query_to_tfidf(query, vectorizer):
    query_vector = vectorizer.transform([query])  # Transform the query text into a vector
    return query_vector

# Function to calculate the similarity between the query and website content
def calculate_similarity(query_vector, content_matrix):
    similarity_scores = cosine_similarity(query_vector, content_matrix)  # Compare the query with the content
    return similarity_scores[0]  # Return the similarity scores for each content chunk

# Function to find the most relevant content chunk based on similarity
def find_most_relevant_content(similarity_scores, content):
    # Finding the index of the most similar chunk
    best_match_index = similarity_scores.argmax()
    return content[best_match_index]  # Return the most relevant chunk

# Main flow
def main():
    url = "https://khaleedhkhan.github.io/portfolio/" 
    query = "What is the main topic of the article?"

    # Step 1: Get website content
    content = get_website_content(url)

    # Step 2: Convert website content into numerical format using TF-IDF
    tfidf_matrix, vectorizer = generate_tfidf_matrix(content)

    # Step 3: Convert the query into the same format for comparison
    query_vector = convert_query_to_tfidf(query, vectorizer)

    # Step 4: Calculate the similarity between the query and content
    similarity_scores = calculate_similarity(query_vector, tfidf_matrix)

    # Step 5: Find the most relevant chunk of content
    best_answer = find_most_relevant_content(similarity_scores, content)

    # Print the response
    print("Response:", best_answer)

# Run the program
if __name__ == "__main__":
    main()
