from dotenv import load_dotenv

from lib import Service as IOUtils
from lib import Service as Service
from lib import OllamaUtils as OllamaUtils
from lib import PromptUtils as PromptUtils

# Load .env (optional) and read model names with fallbacks
load_dotenv()

# Load the dataset
dataset = IOUtils.build_dataset('static/cat-facts.txt')

# Each element in the "knowledge" will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
knowledge = IOUtils.build_knowledge(dataset=dataset)

# Chatbot
user_query = input('Ask me a question: ')
retrieved_knowledge = Service.get_most_relevant_chunks(query=user_query, knowledge=knowledge, top_n=5)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk}')

OllamaUtils.chat(
    user_prompt=user_query,
    system_prompt=PromptUtils.get_system_prompt(knowledge=retrieved_knowledge)
)
