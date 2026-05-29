import streamlit as st
import random
import asyncio
from langchain_core.messages import HumanMessage
from agent import build_graph

st.markdown(""" <style> div[data-baseweb="select"] > div {cursor: pointer !important;} 
button {cursor: pointer !important;} </style> """,unsafe_allow_html=True)
r = random.randint(1001,9999)
st.set_page_config(page_title="WordNest AI Language Assistant",page_icon="🌍",layout="wide")
st.title("WordNest - An AI Language Assistant 🌍")
st.text("An AI-powered language learning assistant that helps you learn a new language by "
        "providing words and sentences in it.")

st.sidebar.header("Learning Settings")
source_language = st.sidebar.selectbox("Language You Want To Learn",
[ "German","Spanish","French","Japanese","Croatian","Greek","Portuguese","Russian",
        "Italian","Korean","Swedish","English"])
target_language = st.sidebar.selectbox("Base Language (You Already Know)",
["English","French","German","Spanish","Japanese","Croatian","Greek","Portuguese","Russian",
        "Italian","Korean","Swedish" ])
difficulty = st.sidebar.selectbox("Difficulty Level",
        ["beginner","intermediate","advanced"])
st.sidebar.text("Made by - Kavyansh Gaur")

st.markdown("## Lets start learning...")
num_words = st.slider("Number Of Words",min_value=5,max_value=50,value=10)
download_option = st.checkbox("Download Content as Text File ?")

async def run_agent(pmpt):
    graph = await build_graph()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=pmpt ) ],
        "source_language":source_language,"target_language": target_language,
        "number_of_words": num_words, "word_difficulty": difficulty})
    return result

if st.button("Generate Learning Content"):
    final_prompt = f"""
    Generate {num_words} {difficulty} words in {source_language}. 
    Base language: {target_language} .
    Provide:
    1. Vocabulary
    2. Meanings
    3. Example Sentences
    4. Short Practice Paragraph
    """
    with st.spinner("Generating learning content..."):
        result = asyncio.run(run_agent(final_prompt))

    response = result["messages"][-1].content
    st.markdown("---")
    st.markdown("## 📘 Learning Content")
    st.write(response)
    if download_option:
        st.download_button( label="Download Learning Content",data=response,
        file_name=f"{source_language}_{difficulty}_{r}.txt", mime="text/plain")
