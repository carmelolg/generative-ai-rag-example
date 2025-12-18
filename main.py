from lib import Service as Service, OllamaUtils as OllamaUtils, PromptUtils as PromptUtils


# Load the dataset
dataset = Service.build_dataset(file_path='static/mc-cartney-story.txt')

# Each element in the "knowledge" will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
knowledge = Service.build_knowledge(dataset=dataset)

def get_most_relevant_chunks(query, knowledge, top_n):
    return Service.get_most_relevant_chunks(query=query, knowledge=knowledge, top_n=top_n)


def chat(user_prompt):
    retrieved_knowledge = get_most_relevant_chunks(user_prompt, knowledge, 5)
    #print('Retrieved knowledge:')
    #for chunk, similarity in retrieved_knowledge:
    #    print(f' - (similarity: {similarity:.2f}) {chunk}')

    OllamaUtils.chat(
        user_prompt=user_prompt,
        system_prompt=PromptUtils.get_system_prompt(knowledge=retrieved_knowledge)
    )

# Start chat
while True:
    user_query = input('\nUser > ')
    if user_query.lower() in ['exit', 'quit']:
        break
    else:
        chat(user_query)



