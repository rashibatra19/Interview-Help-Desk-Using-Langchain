from pathlib import Path

import streamlit as st
from langchain.chains import ConversationChain, RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from streamlit_chat import message

HF_TOKEN = st.secrets["HUGGINGFACE_ACCESS_TOKEN"]

st.write('# AI Interview Help')
st.write('Please upload your resume in pdf format')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.success("Successfully uploaded the resume")
    save_folder = 'resume'
    save_path = Path(save_folder, uploaded_file.name)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, mode='wb') as w:
        w.write(uploaded_file.getvalue())

    # Directly read from the PDF
    loader = PyPDFLoader(save_path)
    pdf_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(pdf_docs)

    # Embedding using HuggingFace
    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    hf = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
    )

    # Vectorstore creation
    vectorstore = FAISS.from_documents(final_documents[:1000], huggingface_embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt_template = """
    Based on the following resume content, generate a detailed list of potential interview questions that an interviewer might ask the candidate. Ensure the questions comprehensively cover various aspects such as the candidate's skills, projects, experience, and qualifications. Provide the questions in a numbered list format. Include a mix of 10 medium questions and 10 advanced questions, clearly categorized.

        Resume Content:
        {context}

        Here is the few list of questions that can be asked in the interview based on your resume..
        
        Questions:
        Basic Questions:
        1.
        2.
        3.
        4.
        5.
        6.
        7.
        8.
        9.
        10.

        Advanced Questions:
        1.
        2.
        3.
        4.
        5.
        6.
        7.
        8.
        9.
        10.
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])

    retrievalQA = RetrievalQA.from_chain_type(
        llm=hf,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    query = "Generate a detailed list of interview questions based on the resume provided. Ensure that the questions comprehensively cover various aspects such as the candidate's skills, projects, experience, and qualifications. Provide the questions in a numbered list format. Include a mix of 10 medium questions and 10 advanced questions, clearly categorized as indicated in the prompt template."

    result = retrievalQA.invoke({"query": query})
    interview_questions = result['result']
    st.write("Based on your resume, here are some potential interview questions that you might be asked:")
    st.write(interview_questions)

    # Initialize conversation memory
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    # Store interview questions in memory
    st.session_state.buffer_memory.save_context(
        {"input": "Generated interview questions"},
        {"output": interview_questions}
    )

    system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, use the information from the uploaded resume.""")

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    chat_prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    conversation_llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=HF_TOKEN, max_length=128, temperature=0.5)

    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=chat_prompt_template, llm=conversation_llm, verbose=True)

    response_container = st.container()
    textcontainer = st.container()

    with textcontainer:
        query = st.text_input("Query: ", key="input")

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i], key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

    if query and query != st.session_state.get("last_query", ""):
        # Save the last query to avoid re-processing the same input
        st.session_state["last_query"] = query

        # Retrieve relevant context from vector database
        search_results = retriever.get_relevant_documents(query)

        # Combine the interview questions and retrieved context for the conversation
        combined_context = f"{interview_questions}\n\nAdditional Context from Resume:\n"
        for doc in search_results:
            combined_context += f"{doc.page_content}\n"

        with st.spinner("typing..."):
            response = conversation.predict(input=f"Context:\n {combined_context} \n\n Query:\n{query}")
        
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
        st.rerun()
