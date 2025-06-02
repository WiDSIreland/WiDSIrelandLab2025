---
title: Women in Data Science Workshop Lab 2
description: Retrieval Augmented Generation (RAG) with Langchain
logo: images/ibm-blue-background.png
---

# Retrieval Augmented Generation (RAG) with Langchain

[Retrieval Augumented Generation (RAG)](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) is an architectural pattern that can be used to augment the performance of language models by recalling factual information from a knowledge base, and adding that information to the model query.

The goal of this lab is to show how you can use RAG with an [IBM Granite](https://www.ibm.com/granite) model to augment the model query answer using a publicly available document. The most common approach in RAG is to create dense vector representations of the knowledge base in order to retrieve text chunks that are semantically similar to a given user query.

RAG use cases include:
- Customer service: Answering questions about a product or service using facts from the product documentation.
- Domain knowledge: Exploring a specialized domain (e.g., finance) using facts from papers or articles in the knowledge base.
- News chat: Chatting about current events by calling up relevant recent news articles.

In its simplest form, RAG requires 3 steps:

- Initial setup:
  - Index knowledge-base passages for efficient retrieval. In this recipe, we take embeddings of the passages and store them in a vector database.
- Upon each user query:
  - Retrieve relevant passages from the database. In this recipe, we use an embedding of the query to retrieve semantically similar passages.
  - Generate a response by feeding retrieved passage into a large language model, along with the user query.

## Prerequisites

This lab is a [Jupyter notebook](https://jupyter.org/). Please follow the instructions in [pre-work](../pre-work/readme.md) to run the lab.


## Loading the Lab


[![Retrieval Augmented Generation (RAG) with Langchain notebook](https://badgen.net/badge/icon/github?icon=github&label=View%20on "View on GitHub")](https://github.com/WiDSIreland/WiDSIrelandLab2025/blob/main/notebooks/RAG_with_Langchain.ipynb)
[![Document Summarization with Granite notebook](https://colab.research.google.com/assets/colab-badge.svg "Open In Colab")](https://colab.research.google.com/github/WiDSIreland/WiDSIrelandLab2025/blob/main/notebooks/RAG_with_Langchain.ipynb)


To run the notebook from your command line in Jupyter using the active virtual environment from the [pre-work](../pre-work/readme.md), run:

```shell
jupyter-lab
```

When Jupyter Lab opens the path to the `notebooks/RAG_with_Langchain.ipynb` notebook file is relative to the `sample-wids` folder from the git clone in the [pre-work](../pre-work/readme.md). The folder navigation pane on the left-hand side can be used to navigate to the file. Once the notebook has been found it can be double clicked and it will open to the pane on the right. 


## Running and Lab (with explanations)

This notebook demonstrates an application of long document summarisation techniques to a work of literature using Granite.

The notebook contains both `code` cells and `markdown` text cells. The text cells each give a brief overview of the code in the following code cell(s). These cells are not executable. You can execute the code cells by placing your cursor in the cell and then either hitting the **Run this cell** button at the top of the page or by pressing the `Shift` + `Enter` keys together. The main `code` cells are described in detail below.


## Choosing the Embeddings Model

```python
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
embeddings_model = HuggingFaceEmbeddings(
    model_name=embeddings_model_path,
)
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)
```

Here we are using the Hugging Face Transformers library to load a pre-trained model for generating embeddings (vector representations of text). Here's a breakdown of what each line does:

1. `from langchain_huggingface import HuggingFaceEmbeddings`: This line imports the `HuggingFaceEmbeddings` class from the `langchain_huggingface` module. This class is used to load pre-trained models for generating embeddings.  

2. `from transformers import AutoTokenizer`: This line imports the `AutoTokenizer` class from the `transformers` library. This class is used to tokenize text into smaller pieces (words, subwords, etc.) that can be processed by the model.  

3. `embeddings_model_path = "ibm-granite/granite-embedding-30m-english"` : This line sets a variable `embeddings_model_path` to the path of the pre-trained model. In this case, it's a model called "granite-embedding-30m-english" developed by IBM's Granite project.

4. `embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_path)`: This line creates an instance of the `HuggingFaceEmbeddings` class, loading the pre-trained model specified by `embeddings_model_path`.

5. `embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)`: This line creates an instance of the `AutoTokenizer` class, loading the tokenizer that was trained alongside the specified model. This tokenizer will be used to convert text into a format that the model can process.

In summary, we are setting up a system for generating embeddings from text using a pre-trained model and its associated tokenizer. The embeddings can then be used for various natural language processing tasks, such as text classification, clustering, or similarity comparison.

To use a model from a provider other than Huggingface, replace this code cell with one from [this Embeddings Model recipe](https://github.com/ibm-granite-community/granite-kitchen/blob/main/recipes/Components/Langchain_Embeddings_Models.ipynb).


## Choose your Vector Database

Specify the database to use for storing and retrieving embedding vectors.

To connect to a vector database other than Milvus substitute this code cell with one from [this Vector Store recipe](https://github.com/ibm-granite-community/granite-kitchen/blob/main/recipes/Components/Langchain_Vector_Stores.ipynb).

```python
from langchain_milvus import Milvus
import tempfile

db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
print(f"The vector database will be saved to {db_file}")

vector_db = Milvus(
    embedding_function=embeddings_model,
    connection_args={"uri": db_file},
    auto_id=True,
    index_params={"index_type": "AUTOINDEX"},
)
```

This Python script is setting up a vector database using Milvus, a vector database built for AI applications, and Hugging Face's Transformers library for embeddings. It uses the previously created Embeddings Model. Here's a breakdown of what the code does:

1. It imports `tempfile` and `Milvus` from `langchain_milvus`.
2. It creates a temporary file for the Milvus database using `tempfile.NamedTemporaryFile()`. This file will store the vector database.
3. It initializes an instance of `Milvus`with the embedding function set to the previously created `embeddings_model`. The connection arguments specify the URI of the database file, which is the temporary file created in the previous step. The `auto_id` parameter is set to True, which means Milvus will automatically generate IDs for the vectors. The `index_params` parameter sets the index type to "AUTOINDEX", which allows Milvus to automatically choose the most suitable index for the data.

In summary, this script sets up a vector database using Milvus and a pre-trained embedding model from Hugging Face. The database is stored in a temporary file, and it's ready to index and search vector representations of text data.


## Selecting your model

Select a Granite model to use. Here we use a Langchain client to connect to  the model. If there is a locally accessible Ollama server, we use an  Ollama client to access the model. Otherwise, we use a Replicate client to access the model.

When using Replicate, if the `REPLICATE_API_TOKEN` environment variable is not set, or a `REPLICATE_API_TOKEN` Colab secret is not set, then the notebook will ask for your [Replicate API token](https://replicate.com/account/api-tokens) in a dialog box.

```python
model_path = "ibm-granite/granite-3.2-8b-instruct"
try:  # Look for a locally accessible Ollama server for the model
    response = requests.get(os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model = OllamaLLM(
        model="granite3.2:2b",
    )
    model = model.bind(raw=True)  # Client side controls prompt
except Exception:  # Use Replicate for the model
    model = Replicate(
        model=model_path,
        replicate_api_token=get_env_var("REPLICATE_API_TOKEN"),
    )
tokenizer = AutoTokenizer.from_pretrained(model_path)

```

1. `model_path = "ibm-granite/granite-3.2-8b-instruct"`: This line assigns the string `"ibm-granite/granite-3.2-8b-instruct"` to the `model_path` variable. This is the name of the pre-trained model on the Hugging Face Model Hub that will be used for the language model.  
2. `try:`: This line starts a try block, which is used to handle exceptions that may occur during the execution of the code within the block.  
3. `response = requests.get(os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))`: This line sends a GET request to the Ollama server using the `requests.get()` function. The server address is obtained from the `OLLAMA_HOST` environment variable. If the environment variable is not set, the default address `http://127.0.0.1:11434` is used.  
4. `model = OllamaLLM(model="granite3.2:2b")`: This line creates an instance of the `OllamaLLM` class from the `ollama` library, specifying the model name as `"granite3.2:2b"`.  
5. `model = model.bind(raw=True)`: This line binds the `OllamaLLM` instance to the client-side, allowing client-side controls over the prompt.  
6. `except:`: This line starts an except block, which is used to handle exceptions that occur within the try block.  
7. `model = Replicate(model=model_path, replicate_api_token=get_env_var("REPLICATE_API_TOKEN"))`: This line creates an instance of the `Replicate` class from the `replicate` library, specifying the model path and the Replicate API token obtained from the `REPLICATE_API_TOKEN` environment variable.  
8. `tokenizer = AutoTokenizer.from_pretrained(model_path)`: This line loads a pre-trained tokenizer for the specified model using the `AutoTokenizer.from_pretrained()` method from the `transformers` library.  

In summary, the code snippet attempts to connect to a locally accessible Ollama server for the specified model. If the connection is successful, it creates an `OllamaLLM` instance and binds it to the client-side. If the connection fails, it uses the Replicate service to load the model. In both cases, a tokenizer is loaded for the specified model using the `AutoTokenizer.from_pretrained()` method.

## Building the Vector Database

In this example, we take the State of the Union speech text, split it into chunks, derive embedding vectors using the embedding model, and load it into the vector database for querying.

### Download the document

Here we use President Biden's State of the Union address from March 1, 2022.

```python
import os
import wget

filename = "state_of_the_union.txt"
url = "https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt"

if not os.path.isfile(filename):
    wget.download(url, out=filename)
```

1. `filename = "state_of_the_union.txt"`: This line assigns the string `"state_of_the_union.txt"` to the `filename` variable. This is the name of the file that will be downloaded and saved locally.  
2. `url = "https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt"`: This line assigns the URL of the file to be downloaded to the `url` variable.  
3. `if not os.path.isfile(filename)`: This line checks if the file specified by `filename` does not already exist in the current working directory. The `os.path.isfile()` function returns `True` if the file exists and `False` otherwise.  
4. `wget.download(url, out=filename)`: If the file does not exist, this line uses the `wget.download()` function to download the file from the specified URL and save it with the name `filename`. The `out` parameter is used to specify the output file name.  

In summary, the code snippet checks if a file with the specified name already exists in the current working directory. If the file does not exist, it downloads the file from the provided URL using the `wget` library and saves it with the specified filename.

# Split the document into chunks

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=embeddings_tokenizer,
    chunk_size=embeddings_tokenizer.max_len_single_sentence,
    chunk_overlap=0,
)
texts = text_splitter.split_documents(documents)
for doc_id, text in enumerate(texts):
    text.metadata["doc_id"] = doc_id
print(f"{len(texts)} text document chunks created")

```

This Python script is using the Langchain library to load a text file and split it into smaller chunks. Here's a breakdown of what each part does:

1. `from langchain.document_loaders import TextLoader`: This line imports the TextLoader class from the langchain.document_loaders module. TextLoader is used to load documents from a file.  
2. `from langchain.text_splitter import CharacterTextSplitter` : This line imports the CharacterTextSplitter class from the `langchain.text_splitter` module. `CharacterTextSplitter` is used to split text into smaller chunks.  
3. `loader = TextLoader(filename)` : This line creates an instance of `TextLoader`, which is used to load the text from the specified file `(filename)`.    
4. `documents = loader.load()` : This line loads the text from the file and stores it in the `documents` variable as a list of strings.    
5. `text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(...)` : This line creates an instance of `CharacterTextSplitter`. It takes a Hugging Face tokenizer `(embeddings_tokenizer)`, sets the chunk size to the maximum length of a single sentence that the tokenizer can handle, and sets the chunk overlap to 0 (meaning no overlap between chunks).  
6. `texts = text_splitter.split_documents(documents)`: This line splits the documents into smaller chunks using the `CharacterTextSplitter` instance. The result is stored in the texts variable as a list of lists, where each inner list contains the chunks of a single document.  
7. `for doc_id, text in enumerate(texts): text.metadata["doc_id"] = doc_id`: This loop assigns a unique identifier (doc_id) to each chunk of text. The doc_id is the index of the chunk in the texts list.  
8. `print(f"{len(texts)} text document chunks created")`: This line prints the total number of text chunks created.  

In summary, this script loads a text file, splits it into smaller chunks based on the maximum sentence length that a Hugging Face tokenizer can handle, assigns a unique identifier to each chunk, and then prints the total number of chunks created.  


## Populate the vector database

NOTE: Population of the vector database may take over a minute depending on your embedding model and service.  

```python
ids = vector_db.add_documents(texts)
print(f"{len(ids)} documents added to the vector database")
```

Next we load the `texts` object created earlier, split it into sentence-sized chunks, and adds these chunks to our vector database, associating each chunk with a unique ID.  

1. `ids = vector_db.add_documents(texts)`: This line adds the text chunks to a vector database (`vector_db`). The `add_documents` method returns a list of IDs for the added documents.  
2. `print(f"{len(ids)} documents added to the vector database")`: This line prints the number of documents added to the vector database.  

## Querying the Vector Database
### Conduct a similarity search

Search the database for similar documents by proximity of the embedded vector in vector space.

```python
query = "What did the president say about Ketanji Brown Jackson?"
docs = vector_db.similarity_search(query)
print(f"{len(docs)} documents returned")
for doc in docs:
    print(doc)
    print("=" * 80)  # Separator for clarity
```


1. `query = "What did the president say about Ketanji Brown Jackson?"`: This line assigns the string `"What did the president say about Ketanji Brown Jackson?"` to the `query` variable. This is the search query that will be used to find relevant documents in the vector database.  
2. `docs = vector_db.similarity_search(query)`: This line calls the `similarity_search()` method of the `vector_db` object, passing the `query` as an argument. The method returns a list of documents that are most similar to the query based on the vector representations of the documents in the vector database.  
3. `print(f"{len(docs)} documents returned")`: This line prints the number of documents returned by the `similarity_search()` method. The `len()` function is used to determine the length of the `docs` list.  
4. `for doc in docs:`: This line starts a loop that iterates over each document in the `docs` list.  
5. `print(doc)`: This line prints the content of the current document in the loop.  
6. `print("=" * 80)`: This line prints a separator line consisting of 80 equal signs (`=`) to improve the readability of the output by visually separating the content of each document.  

In summary, the code snippet defines a search query, uses the `similarity_search()` method of a vector database to find relevant documents, and prints the number of documents returned along with their content. The separator line improves the readability of the output by visually separating the content of each document.  


## Answering Questions

### Automate the RAG pipeline

Build a RAG chain with the model and the document retriever.

First we create the prompts for Granite to perform the RAG query. We use the Granite chat template and supply the placeholder values that the LangChain RAG pipeline will replace.

`{context}` will hold the retrieved chunks, as shown in the previous search, and feeds this to the model as document context for answering our question.

Next, we construct the RAG pipeline by using the Granite prompt templates previously created.

```python
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Create a Granite prompt for question-answering with the retrieved context
prompt = tokenizer.apply_chat_template(
    conversation=[{
        "role": "user",
        "content": "{input}",
    }],
    documents=[{
        "title": "placeholder",
        "text": "{context}",
    }],
    add_generation_prompt=True,
    tokenize=False,
)
prompt_template = PromptTemplate.from_template(template=prompt)

# Create a Granite document prompt template to wrap each retrieved document
document_prompt_template = PromptTemplate.from_template(template="""\
Document {doc_id}
{page_content}""")
document_separator="\n\n"

# Assemble the retrieval-augmented generation chain
combine_docs_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt_template,
    document_prompt=document_prompt_template,
    document_separator=document_separator,
)
rag_chain = create_retrieval_chain(
    retriever=vector_db.as_retriever(),
    combine_docs_chain=combine_docs_chain,
)
```


1. `from langchain.prompts import PromptTemplate`: This line imports the `PromptTemplate` class from the `langchain.prompts` module. This class is used to create custom prompt templates for language models.  
2. `from langchain.chains.retrieval import create_retrieval_chain`: This line imports the `create_retrieval_chain()` function from the `langchain.chains.retrieval` module. This function is used to create a retrieval-augmented generation (RAG) chain, which combines a retrieval component (e.g., a vector database) with a language model for generating context-aware responses.  
3. `from langchain.chains.combine_documents import create_stuff_documents_chain`: This line imports the `create_stuff_documents_chain()` function from the `langchain.chains.combine_documents` module. This function is used to create a chain that combines multiple retrieved documents into a single input for the language model.  
4. `prompt = tokenizer.apply_chat_template(...)`: This line creates a custom prompt template for a question-answering task using the `apply_chat_template()` method of the `tokenizer` object. The prompt template includes a user role with the input question and a document role with the retrieved context. The `add_generation_prompt` parameter is set to `True` to include a generation prompt for the language model.  
5. `prompt_template = PromptTemplate.from_template(template=prompt)`: This line creates a `PromptTemplate` object from the custom prompt template.  
6. `document_prompt_template = PromptTemplate.from_template(template="""\
Document {doc_id}
{page_content}""")`: This line creates a custom prompt template for wrapping each retrieved document. The template includes a document identifier (`{doc_id}`) and the document content (`{page_content}`).  
7. `document_separator="\n\n"`: This line assigns the string `"\n\n"` to the `document_separator` variable. This separator will be used to separate the content of each retrieved document in the combined input for the language model.  
8. `combine_docs_chain = create_stuff_documents_chain(...)`: This line creates a chain that combines multiple retrieved documents into a single input for the language model using the `create_stuff_documents_chain()` function. The function takes the language model (`model`), the prompt template (`prompt_template`), the document prompt template (`document_prompt_template`), and the document separator (`document_separator`) as arguments.  
9. `rag_chain = create_retrieval_chain(...)`: This line creates a retrieval-augmented generation (RAG) chain using the `create_retrieval_chain()` function. The function takes the retrieval component (i.e., the vector database wrapped with `as_retriever()`) and the combine documents chain (`combine_docs_chain`) as arguments.  

In summary, the code snippet imports necessary classes and functions from the `langchain` library to create a retrieval-augmented generation (RAG) chain. It defines a custom prompt template for a question-answering task, creates a document prompt template for wrapping retrieved documents, and assembles the RAG chain by combining the retrieval component and the combine documents chain.  


## Generate a retrieval-augmented response to a question

Use the RAG chain to process a question. The document chunks relevant to that question are retrieved and used as context.  

```python
output = rag_chain.invoke({"input": query})

print(output['answer'])
```


1. `output = rag_chain.invoke({"input": query})`: This line invokes the RAG chain with the input query. The `invoke()` method takes a dictionary as an argument, where the key is `"input"` and the value is the `query` string. The method returns a dictionary containing the output of the RAG chain, which includes the generated answer.  
2. `print(output['answer'])`: This line prints the generated answer from the RAG chain output. The `output` dictionary is accessed using the key `'answer'`, which corresponds to the generated answer in the RAG chain's response.  

In summary, the code snippet invokes the RAG chain with the input query and prints the generated answer from the RAG chain's output.  


## Credits

This notebook is a modified version of the IBM Granite Community [Retrieval Augmented Generation (RAG) with Langchain](https://github.com/ibm-granite-community/granite-snack-cookbook/blob/main/recipes/RAG/RAG_with_Langchain.ipynb) notebook. Refer to the [IBM Granite Community](https://github.com/ibm-granite-community) for the official notebooks.