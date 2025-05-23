  # -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
print(f".env file found at: {dotenv_path}")
load_dotenv(dotenv_path='./multimodal-rag/.env')
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE") 

import nltk
nltk.data.path = ["/home/amax/nltk_data"]

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title  

output_path = '/home/amax/data3/yzd/multimodal-rag/'
file_path = output_path + 'attention.pdf'

elements = partition_pdf(
    filename=file_path,
    strategy="hi_res",
    infer_table_structure=True,
    extract_image_block_types=["Image", "Table"],
    extract_image_block_to_payload=True
)

filtered_elements = [
    e for e in elements if e.category not in ["Footer", "UncategorizedText",'Table', 'Image']
]

text_chunks = chunk_by_title(
    elements=filtered_elements,
    max_characters=10000,
    combine_text_under_n_chars=2000,
    new_after_n_chars=6000,
)

# text_chunks = partition_pdf(
#     filename=file_path,
#     infer_table_structure=True,            # extract tables
#     strategy="hi_res",                     # mandatory to infer tables

#     extract_image_block_types=["Image", "Table"],
 
#     # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

#     extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

#     chunking_strategy="by_title",          # or 'basic'
#     max_characters=10000,                  # defaults to 500
#     combine_text_under_n_chars=2000,       # defaults to 0
#     new_after_n_chars=6000,

#     # extract_images_in_pdf=True,          # deprecated
# )

chunks = text_chunks + [e for e in elements if e.category in ['Table', 'Image']]

# Separate extracted elements into tables, text, and images

tables = []
texts = []
images = []

for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)

    if "CompositeElement" in str(type((chunk))):
        texts.append(chunk)

    if "Image" in str(type((chunk))):
        images.append(chunk.metadata.image_base64)

# # Get the images from the CompositeElement objects
# def get_images_base64(chunks):
#     images_b64 = []
#     for chunk in chunks:
#         if "CompositeElement" in str(type(chunk)):
#             chunk_els = chunk.metadata.orig_elements
#             for el in chunk_els:
#                 if "Image" in str(type(el)):
#                     images_b64.append(el.metadata.image_base64)
#     return images_b64

# images = get_images_base64(chunks)

"""#### Check what the images look like"""

import base64
from IPython.display import Image, display

def display_base64_image(base64_code):
    # Decode the base64 string to binary
    image_data = base64.b64decode(base64_code)
    # Display the image
    display(Image(data=image_data))

def save_base64_image(base64_code, filename="output.png"):
    with open(filename, "wb") as f:
        f.write(base64.b64decode(base64_code))
    print(f"Saved image to {filename}")

# display_base64_image(images[0])

## Summarize the data

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Prompt
prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}

"""
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
llm =  ChatOpenAI(model="gpt-4",
                openai_api_key=api_key,
                openai_api_base=api_base ) 
summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

# Summarize text
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

# Summarize tables
tables_html = [table.metadata.text_as_html for table in tables]
table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})


prompt_template = """Describe the image in detail. For context,
                  the image is part of a research paper explaining the transformers
                  architecture. Be specific about graphs, such as bar plots."""
messages = [
    (
        "user",
        [
            {"type": "text", "text": prompt_template},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image}"},
            },
        ],
    )
]

prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
image_summaries = chain.batch(images)

## Load data and summaries to vectorstore

import uuid
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

"""### Load the summaries and link the to the original data"""

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))

# Add image summaries
img_ids = [str(uuid.uuid4()) for _ in images]
summary_img = [
    Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
]
retriever.vectorstore.add_documents(summary_img)
retriever.docstore.mset(list(zip(img_ids, images)))

"""### Check retrieval"""

# # Retrieve
# docs = retriever.invoke(
#     "who are the authors of the paper?"
# )

# for doc in docs:
#     print(str(doc) + "\n\n" + "-" * 80)

"""## RAG pipeline"""

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from base64 import b64decode


def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    You are an assistant for question-answering tasks. 
    Answer the question based only on the retrieved context, which can include text, tables, and the below image.
    If you don't know the answer, just say that you don't know. 
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )


chain = (
    {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

response = chain.invoke(
    "What is the attention mechanism?"
)

print(response)

chain_with_sources = {
    "context": retriever | RunnableLambda(parse_docs),
    "question": RunnablePassthrough(),
} | RunnablePassthrough().assign(
    response=(
        RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )
)

response = chain_with_sources.invoke(
    "What is Scaled Dot-Product Attention?"
)

print("Response:", response['response'])

print("\n\nContext:")

for text in response['context']['texts']:
    print(text.text)
    print("Page number: ", text.metadata.page_number)
    print("\n" + "-"*50 + "\n")
for image in response['context']['images']:
    display_base64_image(image)

