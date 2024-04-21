from dotenv import load_dotenv
import ast

load_dotenv()

def generate_cypher_query(client, question):
    # System message setting the context for the AIas
    prompt = """
    You are an expert Neo4j Cypher translator who understands the question in English and converts it to Cypher strictly based on the Neo4j Schema provided and following the instructions below:
    1. Generate Cypher query compatible ONLY with Neo4j Version 5
    2. Do not use EXISTS, SIZE,toInt keywords in the cypher. Use alias when using the WITH keyword
    3. Please do not use same variable names for different nodes and relationships in the query.
    4. Use only Nodes and relationships mentioned in the schema
    5. Always enclose the Cypher output inside 3 backticks
    6. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Company name use `toLower(c.name) contains 'neo4j'`
    8. Always use aliases to refer the node in the query
    9. 'Answer' is NOT a Cypher keyword. Answer should never be used in a query.
    10. Cypher is NOT SQL. So, do not mix and match the syntaxes.
    11. Every Cypher query always starts with a MATCH keyword.
    12. Feel free to come up with more queries, outside the sample templates provided if you find they are appropriate.
    13. By default filter by all movies to 1990 by using a where clause unless specified
    14. Return all queries in the form of a list, separated by a comma
    15. Incase no useful information is provided, say recommend something yourself, as the last value in the list
    16. BE VERY CAREFUL WITH THE QUERY FOR VECTOR SIMILARITY QUERY, 'CALL db.index.vector.queryNodes' has to always be there
    17. Try to come up with as many queries as possible
    18.
   

    Schema Details:
    - Node types:
      1. Movie: Attributes include id, primaryTitle, titleType, runtimeMinutes, startYear, numVotes, endYear, About, tconst, originalTitle, genres, averageRating, embedding, isAdult.
      2. Person: Attributes include id, birthYear, deathYear, primaryProfession, nconst, primaryName.

    - Relationships:
      1. :DIRECTED_BY (between Person and Movie)
      2. :WRITTEN_BY (between Person and Movie)
      3. :ACTED_IN (between Person and Movie) with attributes: characters, ordering, job.
    
    Samples:
      1. For querying movies by a specific actor:
        Actor Name: <actor_name>
        MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE toLower(p.primaryName) contains <actor_name> RETURN m.primaryTitle AS MovieTitle, m.startYear AS ReleaseYear LIMIT 3

      2. For querying movies by a specific genre:
        Genre: <genre>
        Query: MATCH (m:Movie) WHERE '<genre>' in m.genres RETURN m.primaryTitle AS MovieTitle, m.startYear AS ReleaseYear

      3. For searching movies by vector similarity to a given movie:
        Reference Movie Title: <reference_movie_title>
        Number of Results: <number_of_results>
        Query: MATCH (m:Movie {{primaryTitle: '<reference_movie_title>'}}) CALL db.index.vector.queryNodes('moviePlots', <number_of_results>, m.embedding) YIELD node, score RETURN node.primaryTitle AS title, node.about AS plot, score

      4. For querying movies directed by a specific person:
        Director Name: <director_name>
        Query: MATCH (p:Person {{primaryName: "<director_name>"}})-[:DIRECTED_BY]->(m:Movie) RETURN m.primaryTitle AS MovieTitle, m.startYear AS Release Year

      5. For quering moves directed by a specific person and acted in by a specific actor:
        Director Name: <director_name>
        Actor Name: <actor_name>
        Query: MATCH (director:Person {primaryName: "<director_name>"})-[:DIRECTED_BY]->(movie:Movie),(actor:Person {primaryName: "<actor_name>"})-[:ACTED_IN]->(movie) RETURN movie.tconst AS MovieID, movie.originalTitle AS MovieTitle, director.primaryName AS Director, actor.primaryName AS Actor
      
      6. For querying who has worked with the most with a particular person in a particular relation
        Person Name: <person_name>
        Relation: <relation_type_1>
        Relation to other person: <realtion_type_2>
        Number : <number_mentioned>, by default go for 1
        
        MATCH (p:Person {primaryName: "<person_name>"})-[:<relation_type_1>]->(movie:Movie)<-[:realtion_type_2]-(actor:Person)
        WHERE NOT p = actor
        RETURN actor.primaryName AS Actor, COUNT(movie) AS NumberOfMovies
        ORDER BY NumberOfMovies DESC
        LIMIT <number_mentioned>

    
    Output Format, python list:
    ["query1',"query2",...]
    """ 
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question}
    ]

    # Generate the completion using GPT-4
    completion = client.chat.completions.create(
        model="gpt-4",  # Ensure you have access to GPT-4
        messages=messages
    )

    # Extracting the latest message from the completion
    # Assuming the last message in the list is the AI's response

    return completion.choices[0].message.content

def find_similar_movies(client, question):
    prompt = """
    1)You are a movie expert.
    2)Based on what the user provides as a prompt, if nothing solid referenceing an actor, genre, director or writer is mentioned, and user only mentions mood, recommend 2 movies, otherwise return empty list.
    OUTPUT FORMAT: [Movie1, Movie2, Movie3]"""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question}
    ]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",  # Ensure you have access to GPT-4
        messages=messages
    )
    return completion.choices[0].message.content

def execute_queries(driver, queries):
    # Connection details
    
    # Initialize the Neo4j driver
    
    # Function to execute a single query
    def execute_query(session, query):
        result = session.run(query)
        return [record for record in result]
    
    # Start a session and execute each query, storing results in a string
    results_string = ""
    with driver.session() as session:
        for query in queries:
            results = execute_query(session, query)
            # Convert each result record to string and join with newlines
            results_string += '\n'.join(str(record) for record in results) + "\n\n"
    
    # Close the driver
    driver.close()
    
    return results_string

def query_kg(neo4j_driver,string):
    try:
        # Remove leading and trailing whitespace and newlines
        list_of_strings =  ast.literal_eval(string)

        # Concatenate all the strings into a single string
        combined_results = ""
        for query in list_of_strings:
            with neo4j_driver.session() as session:
                result = session.run(query)
                for record in result:
                    combined_results += str(record) + "\n"

        return combined_results
    except Exception as e:
        return f"Error: {e}"

def get_context(neo4j_driver, open_ai, user_question):
    cypher_queries = generate_cypher_query(open_ai, user_question)
    context = query_kg(neo4j_driver, cypher_queries)
    extra = """similar movies are {}""".format(find_similar_movies(open_ai,user_question))
    context_2 = """\n<{}>""".format(extra)

    return "THIS IS THE EXTRA CONTEXT ******* " +context + context_2  + "CONTEXT ENDS *******"


print('import succesful')
