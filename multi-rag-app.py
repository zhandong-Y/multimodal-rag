import streamlit as st
import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# from base64 import b64decode
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# os.environ["OPENAI_API_KEY"] = "sk-Q9oWfNAtDAm3mQLnvjRD0F3V0kBfgo3aVHVZK00RBbP9Swk8"
# os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "sk-oVyIUOAAvros6Yi6NcGWmkI00C4VpG6a8krBEBlkJexMhvpp"
os.environ["OPENAI_API_BASE"] = "https://api.aiproxy.io"
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
    st.session_state.chat_history = None


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
    images = [e.metadata.image_base64 for e in raw_elements if e.category == 'Image']
    tables = [e.metadata.text_as_html for e in raw_elements if e.category == 'Table']

    # Build Documents list
    docs = []
    for txt in text_chunks:
        docs.append(Document(page_content=txt, metadata={"type":"text"}))
    for tbl in tables:
        docs.append(Document(page_content=tbl, metadata={"type":"table"}))
    for img in images:
        docs.append(Document(page_content=img, metadata={"type":"image", "is_base64": True}))
    return docs

if process and uploaded_files:
    with st.spinner("Processing documents..."):
        docs = process_documents(uploaded_files)

        embeddings = OpenAIEmbeddings()

        # Build index & retriever
        store = InMemoryStore()
        retriever = MultiVectorRetriever(
            vectorstore=Chroma(collection_name="multimodal", embedding_function=embeddings),  # initialized below
            docstore=store,
            id_key="doc_id",
        )
        
        # Add docs
        ids = []
        for doc in docs:
            _id = str(len(ids))
            ids.append(_id)
            retriever.vectorstore.add_documents([Document(page_content=doc.page_content, metadata={"doc_id": _id, **doc.metadata})])
            store.mset([(_id, doc)])

        # Setup Conversation Chain
        llm = ChatOpenAI(temperature=0)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        st.session_state.conversation = conv_chain
        st.success("Documents processed and RAG chain initialized.")

# ---------------------------------
# Chat Interface
# ---------------------------------
st.header("Multimodal RAG Chat")
user_query = st.text_input("Ask a question about your documents:")
if user_query and st.session_state.conversation:
    result = st.session_state.conversation({"question": user_query})
    chat_history = result['chat_history']
    # Display chat
    for idx, msg in enumerate(chat_history):
        if idx % 2 == 0:
            st.markdown(f"**You:** {msg.content}")
        else:
            # Bot reply may contain image base64 or table HTML
            content = msg.content
            # Detect base64 images
            if msg.content.startswith('data:image'):
                st.image(msg.content)
            else:
                st.markdown(f"**Bot:** {content}")
