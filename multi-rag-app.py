import streamlit as st
import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.chains import ConversationalRetrievalChain # Will be removed effectively
from langchain.memory import ConversationBufferMemory # Will be effectively unused by new chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from base64 import b64decode
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # Ensure using langchain_openai
from langchain_community.vectorstores import Chroma
import uuid

load_dotenv(dotenv_path='./multimodal-rag/.env')
# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") # Add this if you use a custom base

os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

# Sidebar: File Uploader
st.sidebar.header("Upload your PDFs for multimodal RAG")
uploaded_files = st.sidebar.file_uploader(
    "Select one or more PDF files", type=['pdf'], accept_multiple_files=True
)
process = st.sidebar.button("Process Documents")

# Session State
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Initialize as empty list for manual management

# Imports for message types if not already present (though usually pulled in by memory/chain)
from langchain_core.messages import HumanMessage, AIMessage

# Conversation History Summarization Function
def summarize_conversation_history(chat_history, llm):
    """
    Summarizes the conversation history using the provided LLM.
    """
    if not chat_history or len(chat_history) < 3: # e.g., less than 1 full user-AI exchange + next user query
        return "No significant prior conversation to summarize."

    formatted_history = ""
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            formatted_history += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            formatted_history += f"AI: {msg.content}\n"
    
    prompt_template_str = """
You are an expert at summarizing dialogues. Given the following conversation history, provide a concise summary that captures the main topics, questions asked, and key information exchanged. This summary will be used to provide context for a new question.

Conversation History:
{formatted_history}

Concise Summary:
"""
    
    summary_prompt = ChatPromptTemplate.from_template(prompt_template_str)
    
    # Assuming llm is a ChatOpenAI instance or similar, already configured with API keys
    # and StrOutputParser is implicitly handled by how llm.invoke or chain.invoke returns if it's a simple model call
    # For more complex chains, ensure an output parser is used.
    # If llm is a raw model, we might need `| StrOutputParser()`
    
    # Create a simple chain for this
    summarization_chain = summary_prompt | llm | StrOutputParser()
    
    summary = summarization_chain.invoke({"formatted_history": formatted_history.strip()})
    
    return summary

# Document Processing
def process_documents(files):
    # Read & Partition PDF into elements
    raw_elements = []
    for uploaded in files:
        # Save uploaded file to disk for unstructured
        with open(uploaded.name, "wb") as f:
            f.write(uploaded.getbuffer())
        elems = partition_pdf(
            filename=uploaded.name,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True
        )
        raw_elements.extend(elems)

    # Filter out footers and irrelevant text
    filtered = [e for e in raw_elements if e.category not in ["Footer", "UncategorizedText"]]

    # Chunk by titles for text
    text_chunks = chunk_by_title(
        elements=[e for e in filtered if e.category == 'Text'],
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    # Collect images & tables
    # Separate elements
    text_elements = [e for e in filtered if e.category == 'Text'] # These are already chunked by title
    table_elements_raw = [e for e in raw_elements if e.category == 'Table']
    image_elements_raw = [e for e in raw_elements if e.category == 'Image']

    # Initialize LLMs
    text_summarization_llm = ChatOpenAI(
        model="gpt-4", openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE, temperature=0
    )
    image_summarization_llm = ChatOpenAI(
        model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE, temperature=0
    )

    # --- Text Summarization ---
    text_prompt_template = """
    You are an assistant tasked with summarizing text. Give a concise summary of the text.
    Respond only with the summary, no additional comment.
    Text chunk: {element_text}
    """
    text_prompt = ChatPromptTemplate.from_template(text_prompt_template)
    text_summarize_chain = {"element_text": lambda x: x.text} | text_prompt | text_summarization_llm | StrOutputParser()
    # text_chunks were already processed by chunk_by_title, which returns Text objects
    # We need to convert them to Document objects if they are not, or access their text content
    # Assuming text_chunks contains objects with a .text attribute
    texts_to_summarize = [chunk for chunk in text_chunks] # chunk_by_title gives list of Text objects
    text_summaries = text_summarize_chain.batch([t.text for t in texts_to_summarize], {"max_concurrency": 3})


    # --- Table Summarization ---
    table_prompt_template = """
    You are an assistant tasked with summarizing tables. Give a concise summary of the table.
    Respond only with the summary, no additional comment.
    Table: {table_html}
    """
    table_prompt = ChatPromptTemplate.from_template(table_prompt_template)
    table_summarize_chain = {"table_html": lambda x: x} | table_prompt | text_summarization_llm | StrOutputParser()
    tables_html = [tbl.metadata.text_as_html for tbl in table_elements_raw]
    table_summaries = table_summarize_chain.batch(tables_html, {"max_concurrency": 3})

    # --- Image Summarization ---
    # Using the structure from multimdal-rag.py
    image_prompt_text = """Describe the image in detail. For context,
                         the image might be from a research paper or technical document. 
                         Be specific about graphs, charts, diagrams, and any text visible in the image."""

    def create_image_messages(image_base64_string):
        return [
            ("user", [
                {"type": "text", "text": image_prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_string}"}},
            ])
        ]

    image_summarize_chain = ChatPromptTemplate.from_messages(create_image_messages("")[0]) | image_summarization_llm | StrOutputParser()
    images_base64 = [img.metadata.image_base64 for img in image_elements_raw]
    # Need to handle if images_base64 is empty to avoid error with batch processing
    image_summaries = []
    if images_base64:
        # Create message list for each image
        image_messages_list = [create_image_messages(img_b64) for img_b64 in images_base64]
        # The chain expects a single message input, so we pass the constructed message list
        # This part needs adjustment as batch expects a list of inputs, not a list of message lists directly for this chain structure.
        # A more straightforward way for batching with complex inputs to ChatPromptTemplate.from_messages
        # is to invoke it individually or adapt the chain.
        # For simplicity, let's iterate for now, or refine the batch call.
        # Iterative approach for image summarization:
        image_summaries = [
            image_summarize_chain.invoke({"image_base64_string": img_b64}) 
            for img_b64 in images_base64
        ]
        # A more robust batch approach would be:
        # image_summaries = image_summarization_llm.batch(
        #    [ChatPromptTemplate.from_messages(create_image_messages(img_b64)).format_messages() for img_b64 in images_base64],
        #    {"max_concurrency": 3}
        # )
        # image_summaries = [summary.content for summary in image_summaries] # if using llm.batch

    # Store original documents and their summaries
    summary_docs = []
    original_doc_map = {}

    # Original text elements (content of Text objects from chunk_by_title)
    original_texts = [chunk.text for chunk in texts_to_summarize]

    for i, summary in enumerate(text_summaries):
        doc_id = str(uuid.uuid4())
        summary_docs.append(Document(page_content=summary, metadata={"doc_id": doc_id, "type": "text_summary"}))
        original_doc_map[doc_id] = Document(page_content=original_texts[i], metadata={"type": "original_text"})


    for i, summary in enumerate(table_summaries):
        doc_id = str(uuid.uuid4())
        summary_docs.append(Document(page_content=summary, metadata={"doc_id": doc_id, "type": "table_summary"}))
        # Store the original HTML table content
        original_doc_map[doc_id] = Document(page_content=tables_html[i], metadata={"type": "original_table_html"})

    for i, summary in enumerate(image_summaries):
        doc_id = str(uuid.uuid4())
        summary_docs.append(Document(page_content=summary, metadata={"doc_id": doc_id, "type": "image_summary"}))
        # Store the original base64 image string
        original_doc_map[doc_id] = Document(page_content=images_base64[i], metadata={"type": "original_image_base64", "is_base64": True})
    
    return summary_docs, original_doc_map


if process and uploaded_files:
    with st.spinner("Processing documents..."):
        summary_docs, original_doc_map = process_documents(uploaded_files)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE)

        # Build index & retriever
        store = InMemoryStore()
        vectorstore = Chroma(collection_name="multimodal_summaries", embedding_function=embeddings)
        
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key="doc_id", # This should match the metadata key in summary_docs
        )
        
        # Add summary documents to the vectorstore
        if summary_docs: # Ensure there are summaries to add
            retriever.vectorstore.add_documents(summary_docs)
        
        # Add original documents to the docstore
        if original_doc_map:
            retriever.docstore.mset(list(original_doc_map.items()))

        # Setup LLMs for new RAG chain
        llm_summarizer = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE, temperature=0)
        llm_qa = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE, temperature=0)

        # Helper function: Parse retrieved documents
        def parse_retrieved_docs(docs):
            parsed_texts = []
            parsed_tables = []
            parsed_images = []
            for doc in docs: # docs are Document objects from MultiVectorRetriever's docstore
                if doc.metadata.get("type") == "original_text":
                    parsed_texts.append(doc.page_content)
                elif doc.metadata.get("type") == "original_table_html":
                    parsed_tables.append(doc.page_content) # page_content is HTML
                elif doc.metadata.get("type") == "original_image_base64" and doc.metadata.get("is_base64"):
                    parsed_images.append(doc.page_content) # page_content is base64
            return {"texts": parsed_texts, "tables": parsed_tables, "images": parsed_images}

        # Helper function: Build QA prompt
        def build_qa_prompt(parsed_docs, question):
            context_str = ""
            if parsed_docs["texts"]:
                context_str += "Text Context:\n" + "\n\n".join(parsed_docs["texts"])
            if parsed_docs["tables"]:
                context_str += "\n\nTable Context (HTML):\n" + "\n\n".join(parsed_docs["tables"])

            # Base prompt messages
            prompt_messages_content = [
                {"type": "text", "text": f"Based on the following context, please answer the question.\n\nContext:\n{context_str}\n\nQuestion: {question}"}
            ]
            
            # Add images to the prompt messages
            for img_b64 in parsed_docs["images"]:
                prompt_messages_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                )
            
            # System message + Human message with all content blocks
            final_prompt_messages = [
                 ("system", "You are an assistant for question-answering tasks. Answer the question based only on the retrieved context, which can include text, tables, and images. If you don't know the answer, just say that you don't know. Be concise."),
                 ("human", prompt_messages_content)
            ]
            return ChatPromptTemplate.from_messages(final_prompt_messages)

        # Define LCEL RAG chain
        def get_summary_from_input(input_dict):
            # Ensure chat_history is a list, even if None initially from session_state
            current_chat_history = input_dict.get('chat_history') or []
            return summarize_conversation_history(current_chat_history, llm_summarizer)

        def create_augmented_query_from_input(input_dict):
            summary = input_dict['summary']
            question = input_dict['question']
            if summary and summary != "No significant prior conversation to summarize.":
                return f"Conversation summary: {summary}\n\nUser query: {question}"
            return question

        def retrieve_docs_from_query(augmented_query):
            # MultiVectorRetriever returns a list of Document objects (original content)
            return retriever.get_relevant_documents(augmented_query)


        rag_chain = (
            {
                "summary": RunnableLambda(get_summary_from_input),
                "question": RunnablePassthrough(), # Passes the original 'question' from input_dict
                "chat_history": RunnablePassthrough() # Passes 'chat_history'
            }
            | RunnablePassthrough.assign(
                augmented_query=lambda x: create_augmented_query_from_input({"summary": x["summary"], "question": x["question"]["question"]})
            ) # x["question"] is the original input_dict here, so x["question"]["question"] is the user query
            | {
                "retrieved_docs": RunnableLambda(lambda x: retrieve_docs_from_query(x['augmented_query'])),
                "question": RunnableLambda(lambda x: x['question']['question']) # Pass original question string
            }
            | RunnablePassthrough.assign(
                parsed_context=RunnableLambda(lambda x: parse_retrieved_docs(x['retrieved_docs']))
            )
            | {
                "prompt": RunnableLambda(lambda x: build_qa_prompt(x['parsed_context'], x['question'])),
                "retrieved_docs_for_source": RunnableLambda(lambda x: x['retrieved_docs']) 
            }
            | {
                "answer": RunnableLambda(lambda x: x['prompt']) | llm_qa | StrOutputParser(),
                "retrieved_docs_for_source": RunnableLambda(lambda x: x['retrieved_docs_for_source'])
            }
        )
        st.session_state.conversation = rag_chain
        st.success("Documents processed and new RAG chain initialized.")

# ---------------------------------
# Chat Interface
# ---------------------------------
st.header("Multimodal RAG Chat")
user_query = st.text_input("Ask a question about your documents:")

if user_query and st.session_state.conversation:
    # Ensure chat_history is initialized if it's None (e.g., first run after clearing)
    if st.session_state.chat_history is None:
        st.session_state.chat_history = []

    # Append user's query to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # Invoke the RAG chain
    result = st.session_state.conversation.invoke({
        "question": user_query, 
        "chat_history": st.session_state.chat_history # Pass current history
    })
    
    # Append AI's answer to chat history
    st.session_state.chat_history.append(AIMessage(content=result['answer']))
    
    # Display chat history
    # The st.session_state.chat_history already contains the latest user query and AI response
    # So, we iterate through it. When we encounter the *last* AIMessage, 
    # we also display its sources from the `result` variable captured in this turn.

    for i, msg in enumerate(st.session_state.chat_history):
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f"**Bot:** {msg.content}")
            # Check if this is the last message in the history (the one just added)
            if i == len(st.session_state.chat_history) - 1:
                # Display sources for this AI message
                sources = result.get('retrieved_docs_for_source')
                if sources:
                    with st.expander("View Sources Used for This Answer"):
                        for idx, source_doc in enumerate(sources):
                            # Assuming source_doc is a Langchain Document object
                            doc_content = source_doc.page_content
                            doc_metadata = source_doc.metadata
                            source_type = doc_metadata.get("type")

                            if source_type == "original_text":
                                st.markdown("**Retrieved Text Snippet:**")
                                st.markdown(doc_content)
                            elif source_type == "original_table_html":
                                st.markdown("**Retrieved Table:**")
                                st.markdown(doc_content, unsafe_allow_html=True)
                            elif source_type == "original_image_base64" and doc_metadata.get("is_base64"):
                                st.markdown("**Retrieved Image:**")
                                st.image(doc_content)
                            else:
                                st.markdown(f"**Unknown Source Type:** {source_type}")
                                st.markdown(doc_content)
                            
                            # Display other metadata if available, e.g., page number
                            # This was not explicitly part of original doc storage, but good for future
                            if "page_number" in doc_metadata: # Example, if page_number was stored
                                st.caption(f"Source Page: {doc_metadata['page_number']}")
                            
                            if idx < len(sources) - 1:
                                st.divider() 
        st.markdown("---") # Separator between messages/turns
