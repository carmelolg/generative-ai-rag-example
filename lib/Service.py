from lib import OllamaUtils as OllamaUtils
from lib import MathUtils as MathUtils


def build_dataset(file_path):
    """Reads a text file and splits it into overlapping chunks.

    Args:
        file_path (str): Path to the text file.
    """
    with open(file_path, 'r') as file:
        dataset = file.readlines()
        print(f'Loaded {len(dataset)} entries')
        return dataset


def build_knowledge(dataset):
    """Builds a knowledge graph from a dataset.
    Args: dataset (list): A list of lines.
    """
    knowledge = []
    for i, chunk in enumerate(dataset):
        embedding = OllamaUtils.embed_text(text=chunk)
        knowledge.append((chunk, embedding))
        print(f'Added chunk {i + 1}/{len(dataset)} to the database')
    return knowledge


def get_most_relevant_chunks(query, knowledge, top_n=3):
        """Finds the most relevant chunks from a knowledge base based on a query.

        Args:
            query (str): The input query string to find relevant chunks for.
            knowledge (list): A list of tuples where each tuple contains a chunk (str)
                and its corresponding embedding (list or array).
            top_n (int, optional): The number of most relevant chunks to return. Defaults to 3.

        Returns:
            list: A list of the top N most relevant chunks, each represented as a tuple
                containing the chunk (str) and its similarity score (float).
        """
        query_embedding = OllamaUtils.embed_text(text=query)
        # Temporary list to store (chunk, similarity) pairs
        similarities = []
        for chunk, embedding in knowledge:
            similarity = MathUtils.cosine_similarity(query_embedding, embedding)
            similarities.append((chunk, similarity))
        # Sort by similarity in descending order, because higher similarity means more relevant chunks
        similarities.sort(key=lambda x: x[1], reverse=True)
        # Finally, return the top N most relevant chunks
        return similarities[:top_n]
