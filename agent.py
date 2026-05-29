import os
import asyncio
import streamlit as st
from typing import TypedDict, Annotated, Optional, Any
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage,AIMessage
from tools import (get_n_random_words, get_n_random_words_by_difficulty_level,
translate_words,generate_examples_and_text)

key = st.secrets["GROQ_API_KEY"]
local_tools = [get_n_random_words, get_n_random_words_by_difficulty_level, translate_words,
               generate_examples_and_text]

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    source_language: Optional[str]
    number_of_words: Optional[str]
    word_difficulty: Optional[str]
    target_language: Optional[str]

async def setup_tools():
    return [*local_tools]

def assistant(state: AgentState):
    sys_msg = SystemMessage(content="""
    You are an intelligent language-learning assistant.
    Your purpose is to help users learn vocabulary in different languages.

    Available tools:
    1. get_n_random_words(language: str, n: int):
        - Generates random vocabulary words
    2. def get_n_random_words_by_difficulty_level(language: str,
       difficulty_level: str, n: int ) -> list:
    - Generates random vocabulary words by difficulty level
    - Supports difficulty levels:
         beginner
         intermediate
         advanced
    3. def translate_words(random_words: list,
       source_language: str, target_language: str) -> dict:
       - Translates generated words
    4. def generate_examples_and_text(words: list,
       source_language: str,target_language: str) -> dict:
       - Generates example sentences
       - Generates a short practice text
    
    Rules:
    
    Whenever a user asks for vocabulary:
    Step 1: Generate words using get_n_random_words 
    (or using get_n_random_words_by_difficulty_level if difficulty level is given).
    Step 2: Translate the generated words.
    Step 3: Generate example sentences.
    Step 4: Generate a short practice text.Do not stop after generating words.
            Do not stop after translation.
    
    Always provide:
    - vocabulary
    - translations
    - example sentences
    - short practice text
    
    Examples:
    
    User: Give me 10 beginner German words translated to English.
    Workflow:
    get_words
    → translate_words
    → generate_examples_and_text
    
    User: Give me 20 advanced Spanish words.
    Workflow:
    get_n_random_words_by_difficulty_level
    → translate_words
    → generate_examples_and_text
    
    User: Give me 15 random Japanese words.
    Workflow:
    get_n_random_words
    → translate_words
    → generate_examples_and_text  """)

    tools = assistant.tools if hasattr(assistant, "tools") else []
    llm = ChatGroq(model="qwen/qwen3-32b", groq_api_key=key)
    llm_with_tools = llm.bind_tools(tools)
    try:
        msgs = [llm_with_tools.invoke([sys_msg] + state["messages"])]
    except Exception as e:
        msgs = [AIMessage(content="Model temporarily overloaded. Please try again.")]
    return {"messages": msgs,
            "source_language": state.get("source_language"),
            "number_of_words": state.get("number_of_words"),
            "word_difficulty": state.get("word_difficulty"),
            "target_language": state.get("target_language")}

async def build_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    """" Build the state graph with properly initialized tools """
    tools = await setup_tools()
    assistant.tools = local_tools
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()

async def main():
    """ Main async function to run the application."""
    react_graph = await build_graph()
    user_prompt = ("Get 10 beginner words in German, translated to English."
                   "The translation of the words should be in English script")
    messages = [HumanMessage(content=user_prompt)]
    result = await react_graph.ainvoke({
        "messages": messages, "source_language": None, "number_of_words": None,
        "word_difficulty": None, "target_language": None})
    print("\n")
    print("=" * 80)
    print("FINAL RESPONSE")
    print("=" * 80)
    print("\n")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
