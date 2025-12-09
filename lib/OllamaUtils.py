"""
A thin wrapper around the installed `ollama` client to provide a small,
well-documented helper for producing embeddings from text.

This module intentionally imports the real `ollama` package (the one
installed from PyPI) and exposes a simple function `embed_text` that
accepts a string and returns its embedding as a list of floats.
"""
from pyexpat.errors import messages
from typing import List
import ollama as _ollama_client
from lib import PromptUtils as PromptUtils

import os

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text:latest')
LANGUAGE_MODEL = os.getenv('LANGUAGE_MODEL', 'gemma3:270m')


def embed_text(text: str) -> List[float]:
    return _ollama_client.embed(model=EMBEDDING_MODEL, input=text)['embeddings'][0]


def chat(
        user_prompt: str,
        system_prompt: str = None,
        assistant_prompt: str = None
):
    _messages = []
    if system_prompt is not None:
        _messages.append({'role': 'system', 'content': system_prompt})
    if assistant_prompt is not None:
        _messages.append({'role': 'assistant', 'content': assistant_prompt})

    _messages.append({'role': 'user', 'content': user_prompt})

    stream = _ollama_client.chat(
        model=LANGUAGE_MODEL,
        messages=_messages,
        stream=True,
    )

    # print the response from the chatbot in real-time
    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
