# Import the necessary libraries for data processing, graph creation, and visualization
import pandas as pd  # For data manipulation and reading CSV files
import networkx as nx  # For creating and manipulating the graph structure
import matplotlib.pyplot as plt  # For plotting the graph
from sklearn.metrics.pairwise import cosine_similarity  # For calculating similarity between users
import seaborn as sns  # For enhanced plotting with seaborn

# Reading the datasets containing movie and ratings data
df1 = pd.read_csv(r'movies.csv')  # Read the movie dataset (movieId and title)
df2 = pd.read_csv(r'ratings.csv')  # Read ratings dataset (userId, movieId, rating)

# Identify the top 100 movies based on the number of ratings
top_movies = df2.groupby('movieId').size().sort_values(ascending=False).head(100).index  # Get top 100 most rated movies

# Filter both datasets to include only the top 100 movies
df1 = df1[df1['movieId'].isin(top_movies)]  # Filter movie dataset by top movies
df2 = df2[df2['movieId'].isin(top_movies)]  # Filter ratings dataset by top movies

# Merge both datasets to combine movie details with ratings
df = df2.merge(df1, left_on='movieId', right_on='movieId', how='left')  # Merge on 'movieId'

# Remove unnecessary columns like timestamp and genres
df = df.drop(columns=['timestamp', 'genres'])  # Drop unnecessary columns

# Create a user-movie matrix (pivot table) 
user_movie_matrix = pd.pivot_table(df, values='rating', index='movieId', columns='userId').fillna(0)  # Fill NaN values with 0 (no rating)

# Calculate cosine similarity between users based on their ratings
cosine_sim_matrix = cosine_similarity(user_movie_matrix.T)  # Transpose matrix to get user similarity
user_user_matrix = pd.DataFrame(cosine_sim_matrix, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)  # Create DataFrame

# Dynamically set active user
active_user = 1  # Change this value to set a different active user

# Extract the top 5 most similar users for the specified active user
df_similar_users = pd.DataFrame(user_user_matrix.loc[active_user].sort_values(ascending=False).head(6)).reset_index()  # Get similarity scores for the active user
df_similar_users.columns = ['userId', 'similarity']  # Rename columns for clarity
df_similar_users = df_similar_users[df_similar_users['userId'] != active_user]  # Exclude the active user from the similar users list

# Merge the similar users with the ratings and movie information
final_df = df_similar_users.merge(df, on='userId', how='left')  # Merge ratings of similar users
final_df['score'] = final_df['similarity'] * final_df['rating']  # Calculate weighted score based on similarity and rating

# Ensure movies are restricted to the top 100
final_df = final_df[final_df['movieId'].isin(top_movies)]  # Filter final DataFrame by top 100 movies

# Filtering out movies already watched by the active user
watched_df = df[df['userId'] == active_user]  # Get the movies watched by the active user
final_df = final_df[~final_df['movieId'].isin(watched_df['movieId'])]  # Remove movies already watched by the active user

# Get the top 10 recommended movies for the active user based on the weighted score
recommended_df = final_df.sort_values(by='score', ascending=False).head(10)[['movieId', 'title', 'score']]  # Top 10 recommended movies

# Print the recommended movies
print(f"Recommended Movies for User {active_user}:")
print(recommended_df)

# Construct the graph to visualize relationships between users and movies
B = nx.Graph()  # Initialize an undirected graph

# Get similar users and movies rated by them, along with the recommended movies
similar_users = df_similar_users['userId'].tolist()  # List of similar users
movies_rated_by_similar_users = final_df['movieId'].unique()  # List of movies rated by similar users
recommended_movies = recommended_df['movieId'].unique()  # List of recommended movies

# Add nodes to the graph for each group
B.add_nodes_from(similar_users, bipartite=0)  # Add similar user nodes (Group 0)
B.add_nodes_from(movies_rated_by_similar_users, bipartite=1)  # Add rated movie nodes (Group 1)
B.add_nodes_from(recommended_movies, bipartite=1)  # Add recommended movie nodes (Group 1)
B.add_node(active_user, bipartite=0)  # Add active user node

# Add edges for movies rated by similar users
for _, row in final_df.iterrows():
    B.add_edge(row['userId'], row['movieId'], weight=row['rating'])  # Add an edge from similar user to rated movie

# Add edges for recommended movies (between active user and recommended movies)
for _, row in recommended_df.iterrows():
    B.add_edge(active_user, row['movieId'], weight=row['score'])  # Add an edge from active user to recommended movie

# Prepare the data for visualization using Seaborn
edges = list(B.edges(data=True))  # Get list of edges with weight data
nodes = list(B.nodes)  # Get list of all nodes in the graph

# Extracting node positions using a spring layout algorithm for better visual spread
pos = nx.spring_layout(B, k=0.5, iterations=50, seed=42)  # Adjust k for spacing

# Extract node positions into a DataFrame for Seaborn visualization
node_positions = pd.DataFrame([(node, pos[node][0], pos[node][1]) for node in nodes], columns=['Node', 'X', 'Y'])

# Assigning types to nodes for color mapping (Active user, Similar users, etc.)
node_positions['Type'] = node_positions['Node'].apply(
    lambda x: 'Active User' if x == active_user else
              ('Similar Users' if x in similar_users else
               ('Recommended Movies' if x in recommended_movies else 'Rated Movies'))
)

# Map node types to colors for visualization
color_map = {
    'Active User': 'red',  # Active user node will be red
    'Similar Users': 'blue',  # Similar users will be blue
    'Recommended Movies': 'orange',  # Recommended movies will be orange
    'Rated Movies': 'green'  # Rated movies will be green
}
node_positions['Color'] = node_positions['Type'].map(color_map)  # Apply color map based on node type

# Extract edge positions (start and end points) and weight for labeling
edge_positions = pd.DataFrame(
    [(pos[edge[0]][0], pos[edge[0]][1], pos[edge[1]][0], pos[edge[1]][1], edge[2]['weight'])
     for edge in edges],
    columns=['X1', 'Y1', 'X2', 'Y2', 'Weight']  # Coordinates and weight for each edge
)

# Plotting the graph using Seaborn for better visualization
plt.figure(figsize=(18, 12))  # Set the figure size

# Draw edges: Plot each edge with a line from node1 to node2
for _, edge in edge_positions.iterrows():
    plt.plot([edge['X1'], edge['X2']], [edge['Y1'], edge['Y2']], color='gray', alpha=0.6, linewidth=1)
    mid_x = (edge['X1'] + edge['X2']) / 2  # Find the midpoint of the edge for label positioning
    mid_y = (edge['Y1'] + edge['Y2']) / 2
    plt.text(mid_x, mid_y, f"{edge['Weight']:.2f}", fontsize=8, color='black', alpha=0.7,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))  # Label the edge with the similarity score

# Draw nodes using Seaborn scatterplot
sns.scatterplot(
    x='X', y='Y', hue='Type', palette=color_map, s=300,
    data=node_positions, legend='full', alpha=0.9  # Plot nodes with different colors for each type
)

# Add labels to nodes (display Node ID at each node position)
for _, node in node_positions.iterrows():
    plt.text(node['X'], node['Y'] + 0.02, str(node['Node']), fontsize=10, ha='center', color='black', fontweight='bold')

# Add legend and title to the plot
plt.title("Graph: Similar Users and Recommended Movies", fontsize=18)  # Set the plot title
plt.legend(loc='upper left', title="Node Type", fontsize=12)  # Add legend
plt.axis('off')  # Turn off axis
plt.show()  # Display the plot
