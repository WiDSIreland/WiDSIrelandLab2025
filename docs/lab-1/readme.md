---
title: Women in Data Science Workshop Lab 1
description: Document Summarization with Granite
logo: images/ibm-blue-background.png
---

# Document Summarization with Granite

[Text summarization](https://www.ibm.com/topics/text-summarization) condenses one or more texts into shorter summaries for enhanced information extraction.

The goal of this lab is to show how the [IBM Granite](https://www.ibm.com/granite/docs/models/granite/) models can be used in order to apply long document summarization techniques to a work of literature.


## Prerequisites

This lab is a [Jupyter notebook](https://jupyter.org/). Please follow the instructions in [pre-work](../pre-work/readme.md) to run the lab.


## Loading the Lab

Using colab to run the remotely 
[![Document Summarization with Granite notebook](https://badgen.net/badge/icon/github?icon=github&label=View%20on "View on GitHub")](https://github.com/WiDSIreland/WiDSIrelandLab2025/blob/main/notebooks/Summarize.ipynb)
[![Document Summarization with Granite notebook](https://colab.research.google.com/assets/colab-badge.svg "Open In Colab")]((https://github.com/WiDSIreland/WiDSIrelandLab2025/blob/main/notebooks/Summarize.ipynb)

To run the notebook from command line in Jupyter using the active virtual environment from the [pre-work](../pre-work/readme.md), run:

```shell
jupyter-lab
```

When Jupyter Lab opens the path to the `notebooks/Summarize.ipynb` notebook file is relative to the `WiDSIreland` folder from the git clone in the [pre-work](../pre-work/readme.md). The folder navigation pane on the left-hand side can be used to navigate to the file. Once the notebook has been found it can be double clicked and it will open to the pane on the right. 


## Running and Lab (with explanations)

This notebook demonstrates an application of long document summarisation techniques to a work of literature using Granite.

The notebook contains both `code` cells and `markdown` text cells. The text cells gives a brief overview of the code and these cells are not executable. The code cells can be executed by placing the cursor in the cell and either hitting the **Run this cell** button at the top of the page or by pressing the `Shift` + `Enter` keys together. The main `code` cells are described in detail below.


## Selecting your model

Select a Granite model to use. Here we use a Langchain client to connect to the model. If there is a locally accessible Ollama server, we use an Ollama client to access the model. Otherwise, we use a Replicate client to access the model.

When using Replicate, if the `REPLICATE_API_TOKEN` environment variable is not set, or a `REPLICATE_API_TOKEN` Colab secret is not set, then the notebook will ask for your [Replicate API token](https://replicate.com/account/api-tokens) in a dialog box.

```python
try:  # Look for a locally accessible Ollama server for the model
    response = requests.get(os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model = OllamaLLM(
        model="granite3.2:2b",
        num_ctx=65536,  # 64K context window
    )
    model = model.bind(raw=True)  # Client side controls prompt
except Exception:  # Use Replicate for the model
    model = Replicate(
        model="ibm-granite/granite-3.2-8b-instruct",
        replicate_api_token=get_env_var("REPLICATE_API_TOKEN"),
        model_kwargs={
            "max_tokens": 2000,  # Set the maximum number of tokens to generate as output.
            "min_tokens": 200,  # Set the minimum number of tokens to generate as output.
            "temperature": 0.75,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    )
```

In this first piece of code we **try** to determine if there is a local Ollama server running on `http://127.0.0.1:11434`. If the Ollama server is found then an `OllamaLLM` model instance is created for use later. If the Ollama server is not found the code then reverts to using the Granite 3.2-8b model served from Replicate.


### Chunk Document

```python
def chunk_document(
    source: str,
    *,
    dropwhile: Callable[[BaseChunk], bool] = lambda c: False,
    takewhile: Callable[[BaseChunk], bool] = lambda c: True,
) -> Iterator[BaseChunk]:
    """Read the document and perform a hierarchical chunking"""
    converter = DocumentConverter()
    chunks = HierarchicalChunker().chunk(converter.convert(source=source).document)
    return itertools.takewhile(takewhile, itertools.dropwhile(dropwhile, chunks))

```

This Python function, `chunk_document`, is designed to perform hierarchical chunking on a given text document. Here's a breakdown of its components:

1. **Function Signature**: The function takes one required argument and two optional arguments. The required arguments are:

   - `source`: A string representing the text document to be chunked.
   - `*`: Is a marker that all later arguments must be passed by keyword.
   - The optional arguments are:
     - `dropwhile` : A callable (function) that takes a `BaseChunk` object and returns a boolean. This function is used to determine when to stop dropping elements from the beginning of the chunks. The default is a lambda function that always returns `False`, meaning it will never drop any elements.
     - `takewhile`: A callable (function) that takes a `BaseChunk` object and returns a boolean. This function is used to determine when to stop taking elements from the beginning of the chunks. The default is a lambda function that always returns `True`, meaning it will take all elements.

2. **Document Conversion**: The function first converts the input `source` string into a document object using a `DocumentConverter` instance (`converter` created on the first line of the function).

3. **Chunking**: It then uses a `HierarchicalChunker` instance to perform hierarchical chunking on the document. The result is a list of `BaseChunk` objects.

4. **Itertools Dropwhile and Takewhile**: Finally, the function uses `itertools.dropwhile` and `itertools.takewhile`to iterate over the chunks. The `dropwhile` function will drop elements from the beginning of the chunks as long as the `dropwhile` callable returns `True`. The `takewhile` 

   function will take elements from the remaining chunks as long as the `takewhile` callable returns `True`.

5. **Return Value**: The function returns an iterator of `BaseChunk` objects, which represent the hierarchically chunked document.

In summary, this function allows you to define custom conditions for dropping and taking chunks of a document, providing flexibility in how the document is segmented. The default behaviour is to not drop any chunks and to take all chunks.



### Merge Chunks

```python
def merge_chunks(
    chunks: Iterator[BaseChunk],
    *,
    headings: Callable[[BaseChunk], list[str]] = lambda c: c.meta.headings,
) -> Iterator[dict[str, str]]:
    """Merge chunks having the same headings"""
    prior_headings: list[str] | None = None
    document: dict[str, str] = {}
    for chunk in chunks:
        text = chunk.text.replace("\r\n", "\n")
        current_headings = headings(chunk)
        if prior_headings != current_headings:
            if document:
                yield document
            prior_headings = current_headings
            document = {"title": " - ".join(current_headings), "text": text}
        else:
            document["text"] += f"\n\n{text}"
    if document:
        yield document
```

1. **Function Signature**: The function is designed to merge chunks of text that share the same headings. It takes one required argument and one optional arguments. The required arguments are:
   - `chunks`: An an iterator of `BaseChunk` objects
   - `*`: Is a marker that all later arguments must be passed by keyword.
   - `headings`:An optional callable function that extracts headings from a chunk. The default function simply returns the `headings` attribute of the chunk's metadata.
2. **For each Chunk**: The function iterates over each chunk in the input iterator. For each chunk, it extracts the text and headings. If the current chunk's headings differ from the previous chunk's headings, it yields the accumulated document (a dictionary with 'title' and 'text' keys) and starts a new document with the current chunk's headings and text. If the headings are the same, it appends the current chunk's text to the existing document.
3. **Finally**: Finally, after the loop, if there's any remaining document (i.e., the last chunk in the iterator didn't end a section), it yields that document.

The function returns an iterator of dictionaries, where each dictionary represents a merged chunk of text with its corresponding headings. The 'title' key in the dictionary is a concatenation of the headings with " - " as a separator, and the 'text' key contains the merged text of the chunks that share the same headings.



### Chunk Dropwhile

```python
def chunk_dropwhile(chunk: BaseChunk) -> bool:
    """Ignore front matter prior to the book start"""
    return "WALDEN" not in chunk.meta.headings
```

This Python function is designed to process chunks of data, specifically in the context of a document or book. The function takes one argument, `chunk`, which is expected to be an instance of a class or type named `BaseChunk`. This class or type is presumably defined elsewhere in the codebase and is likely used to represent a segment or part of a larger document.

The `chunk` object has a property called `meta`, which is assumed to be an object containing metadata about the chunk. This metadata includes a list of headings, stored in `meta.headings`.

The function checks if the string "WALDEN" is not in the list of headings. If "WALDEN" is not found in the headings, the function returns `True`, indicating that the chunk should be retained or processed further. If "WALDEN" is found in the headings, the function returns `False`, indicating that the chunk should be ignored or dropped.

In essence, this function is used to filter out chunks that represent front matter (like a table of contents, preface, or introduction) before the main content of the book, which is assumed to start with the heading "WALDEN". This is a common pattern in text processing, where you want to skip over certain sections of a document.



### Chunk Takewhile

```python
def chunk_takewhile(chunk: BaseChunk) -> bool:
    """Ignore remaining chunks once we see this heading"""
    return "ON THE DUTY OF CIVIL DISOBEDIENCE" not in chunk.meta.headings
```

This Python function, named `chunk_takewhile`, is designed to be used in a context where data is being processed in chunks. The function takes one argument, `chunk`, which is expected to be an instance of a class or subclass named `BaseChunk`.

The purpose of this function is to determine whether to continue processing subsequent chunks or to stop processing based on the content of the current chunk. It does this by checking if a specific string, `"ON THE DUTY OF CIVIL DISOBEDIENCE"`, is present in the `headings` attribute of the `meta` attribute of the `chunk`  object.

If the string is not found in the `headings`, the function returns `True`, indicating that the processing should continue with the next chunk. If the string is found, the function returns `False`, indicating that the processing should stop after the current chunk.

In essence, this function acts as a filter or a condition for chunk processing, allowing you to control when to stop processing based on the content of the chunks. This can be particularly useful when dealing with large datasets or files that can be divided into smaller, more manageable chunks.



### Chunk Headings

```python
def chunk_headings(chunk: BaseChunk) -> list[str]:
    """Use the h1 and h2 (chapter) headings"""
    return chunk.meta.headings[:2]
```

This Python function, named `chunk_headings`, is designed to extract the first two headings (h1 and h2) from a given chunk of content. The function takes one parameter, `chunk`, which is expected to be an instance of a class named `BaseChunk`.

The `BaseChunk` class, which was imported from `docling_core.transforms.chunker.base` has a `meta` attribute, which itself is expected to be an object containing metadata about the chunk. This metadata object has a `headings` attribute, which is a list of strings representing the headings found in the chunk.

The function returns a list containing the first two headings from this list. If there are fewer than two headings in the chunk, it will return all available headings.

Here's a breakdown of the function:

- This function named `chunk_headings` takes one parameter,`chunk`, which is annotated to be an instance of `BaseChunk`
- The function is expected to return a list of strings (`list[str]`).
- `return chunk.meta.headings[:2]`. This line returns a slice of the `headings` list from the `meta` attribute of the `chunk` object. The slice `[:2]` indicates that only the first two elements of the list should be returned.

In summary, this function is used to extract the first two headings (h1 and h2) from a chunk of content, which is used for creating a table of contents or for any other purpose that requires identifying the main sections of a document.



### Creating the Document Objects to Summarise

```python
documents: list[dict[str, str]] = list(
    merge_chunks(
        chunk_document(
            "https://www.gutenberg.org/cache/epub/205/pg205-images.html",
            dropwhile=chunk_dropwhile,
            takewhile=chunk_takewhile,
        ),
        headings=chunk_headings,
    )
)

```

This Python code is fetching and processing a webpage from the Project Gutenberg website. Here's a breakdown:

1. `chunk_document`
   This function is described in details earlier. It takes a URL and two other functions,`chunk_dropwhile` and `chunk_takewhile`, as arguments. It's responsible for fetching the webpage content and dividing it into smaller chunks based on the conditions defined by `chunk_dropwhile` and `chunk_takewhile`.

   might keep adding text to a chunk until a certain condition is no longer met. ` chunk_headings` is another function that takes a chunk of text and returns a new chunk containing only the headings (HTML `<h1>` and `<h2>` tags) from the original chunk.

2. `merge_chunks`
   Takes the chunks generated by `chunk_document` and combines them into a single list of dictionaries. Each dictionary represents a document with keys and values corresponding to the document's content and metadata, respectively.

In summary, this code is fetching a webpage, dividing its content into chunks based on the defined conditions, extracting headings from each chunk, merging all chunks back into a single list, and storing the result as a list of dictionaries. Each dictionary in the list represents a document with its content and metadata.



### Summarise Chunks

Here we define a method to generate a response using a list of documents and a user prompt about those documents. We create the prompt according to the [Granite Prompting Guide](https://www.ibm.com/granite/docs/models/granite/#chat-template) and provide the documents using the `documents` parameter.

```python
def generate(user_prompt: str, documents: list[dict[str, str]]):
    """Use the chat template to format the prompt"""
    prompt = tokenizer.apply_chat_template(
        conversation=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        documents=documents,  # This uses the documents support in the Granite chat template
        add_generation_prompt=True,
        tokenize=False,
    )

    print(f"Input size: {len(tokenizer.tokenize(prompt))} tokens")
    output = model.invoke(prompt)
    print(f"Output size: {len(tokenizer.tokenize(output))} tokens")

    return output
```

This Python function, named `generate` , is designed to interact with a language model, such as a chatbot or in our case our Granite Model, using a provided user prompt and a list of documents. Here's a breakdown of what the function does:

1. The function takes two arguments:
   - `user_prompt`: a string representing the user's input or question.
   - `documents`: a list of dictionaries, where each dictionary contains key-value pairs representing the documents to be used as context for the language model.
2. The function uses a tokenizer to apply a chat template to the user prompt and the provided documents. The chat template is a predefined format that structures the input for the language model. The `apply_chat_template` method formats the input as a conversation with a single "user" role and the user's prompt as the content. The `documents` argument is passed to the template to provide additional context for the model.
3. The `add_generation_prompt` parameter is set to `True` , which means that a special generation prompt will be added to the input. This prompt guides the model to generate a response rather than just selecting an answer from a predefined set.
4. The `tokenize` parameter is set to `False`, which means that the input and output will not be tokenized (i.e., broken down into individual words or subwords) before being passed to the model.
5. The function then prints the size of the input in terms of the number of tokens (words or subwords) after tokenization.
6. The `model.invoke(prompt)` line calls the language model with the formatted prompt and retrieves the generated output.
7. The function prints the size of the output in terms of the number of tokens.
8. Finally, the function returns the generated output from the language model.

In summary, this function formats a user prompt and contextual documents using a chat template, invokes a language model with the formatted input, and returns the generated output. The token sizes are printed for debugging purposes.



```python
user_prompt = """\
Using only the the book chapter document, compose a summary of the book chapter.
Your response should only include the summary. Do not provide any further explanation."""

summaries: list[dict[str, str]] = []

for document in documents:
    print(
        f"============================= {document['title']} ============================="
    )
    output = generate(user_prompt, [document])
    summaries.append({"title": document["title"], "text": output})

print("Summary count: " + str(len(summaries)))
```

We then define the prompt we wist to use and invoke this `generate` function for each chapter creating a separate summary for each and populating a list of dictionaries with the chapter title and the summary from the model.



### Final Summary

```python
user_prompt = """\
Using only the book chapter summary documents, compose a single, unified summary of the book.
Your response should only include the unified summary. Do not provide any further explanation."""

output = generate(user_prompt, summaries)
print(output)
```

To conclude we then call the `generate` function again but this time, instead of passing the full text of each chapter, we pass the list of dictionaries containing the chapter summaries and ask the model to provide an overall summary of the book by summarising the summaries of each chapter. So we have now summarized a document larger than the AI model's context window length by breaking the document down into smaller pieces to summarize and then summarizing those summaries.



## Credits

This notebook is a modified version of the IBM Granite Community [Document Summarization](https://github.com/ibm-granite-community/granite-snack-cookbook/blob/main/recipes/Summarize/Summarize.ipynb) notebook. Refer to the [IBM Granite Community](https://github.com/ibm-granite-community) for the official notebooks.
