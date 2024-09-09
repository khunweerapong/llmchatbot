import os
import streamlit as st
from llmware.resources import CustomTable
from llmware.models import ModelCatalog
from llmware.prompts import Prompt
from llmware.parsers import Parser
from llmware.configs import LLMWareConfig
from llmware.agents import LLMfx
from llmware.setup import Setup


# Keeps a running state of any CSV tables that have been loaded in the session to avoid duplicated inserts
if "loaded_tables" not in st.session_state:
    st.session_state["loaded_tables"] = []


def build_table(db=None, table_name=None, load_fp=None, load_file=None):
    """ Simple example script to take a CSV or JSON/JSONL and create a DB Table."""
    if not table_name:
        return 0

    # Avoid rebuilding the table if it has already been loaded
    if table_name in st.session_state["loaded_tables"]:
        return 0

    # Construct the full file path and check if it exists
    full_file_path = os.path.join(load_fp, load_file)
    if not os.path.isfile(full_file_path):
        st.error(f"File not found: {full_file_path}")
        return -1

    custom_table = CustomTable(db=db, table_name=table_name)
    analysis = custom_table.validate_csv(full_file_path, load_file)
    print("update: analysis from validate_csv: ", analysis)

    if load_file.endswith(".csv"):
        output = custom_table.load_csv(full_file_path, load_file)
    elif load_file.endswith(".jsonl") or load_file.endswith(".json"):
        output = custom_table.load_json(full_file_path, load_file)
    else:
        print("File type not supported for DB load")
        return -1

    print("update: output from loading file: ", output)

    # Display a sample of rows
    sample_range = min(10, len(custom_table.rows))
    for x in range(sample_range):
        print(f"update: sample rows {x}: ", custom_table.rows[x])

    # Test and remediate schema data type
    updated_schema = custom_table.test_and_remediate_schema(samples=20, auto_remediate=True)
    print("update: updated schema: ", updated_schema)

    # Insert the rows into the DB
    custom_table.insert_rows()

    # Add the table name to session state to avoid future reloads
    st.session_state["loaded_tables"].append(table_name)

    return len(custom_table.rows)


@st.cache_resource
def load_reranker_model():
    """Loads the reranker model used in the RAG process to rank the semantic similarity."""
    reranker_model = ModelCatalog().load_model("jina-reranker-turbo")
    return reranker_model


@st.cache_resource
def load_prompt_model():
    """Loads the core RAG model used for fact-based question-answering."""
    prompter = Prompt().load_model("bling-phi-3-gguf", temperature=0.0, sample=False)
    return prompter


@st.cache_resource
def load_agent_model():
    """Loads the Text2SQL model used for querying the CSV table."""
    agent = LLMfx()
    agent.load_tool("sql", sample=False, get_logits=True, temperature=0.0)
    return agent


@st.cache_resource
def parse_file(fp, doc):
    """Parses a newly uploaded file and saves the parser output as text chunks with metadata."""
    parser_output = Parser().parse_one(fp, doc, save_history=False)
    st.cache_resource.clear()
    return parser_output


def get_rag_response(prompt, parser_output, reranker_model, prompter):
    """Executes a RAG response with ranking and fact-checking."""
    if len(parser_output) > 3:
        output = reranker_model.inference(prompt, parser_output, top_n=10, relevance_threshold=0.25)
    else:
        output = [{"rerank_score": 0.0, **entry} for entry in parser_output]

    # Use top 3 relevant results for response generation
    use_top = 3
    output = output[:use_top]

    sources = prompter.add_source_query_results(output)
    responses = prompter.prompt_with_source(prompt, prompt_name="default_with_context")

    # Execute post-inference fact and source checking
    source_check = prompter.evidence_check_sources(responses)
    numbers_check = prompter.evidence_check_numbers(responses)
    nf_check = prompter.classify_not_found_response(responses, parse_response=True, evidence_match=False, ask_the_model=False)

    bot_response = ""
    for i, resp in enumerate(responses):
        bot_response = resp['llm_response']
        add_sources = True

        if "not_found_classification" in nf_check[i] and nf_check[i]["not_found_classification"]:
            add_sources = False
            bot_response += "\n\nThe answer to the question was not found in the source passage."

        if add_sources:
            bot_response += _append_fact_and_source_check(numbers_check[i], source_check[i])

    prompter.clear_source_materials()
    return bot_response


def _append_fact_and_source_check(numbers_check, source_check):
    """Helper function to append fact-check and source-check results to the response."""
    result = ""
    if numbers_check.get("fact_check"):
        fact = numbers_check["fact_check"][0]
        result += f"Text: {fact['text']}\nSource: {fact['source']}\nPage Num: {fact['page_num']}\n"

    if not result and source_check.get("source_review"):
        source = source_check["source_review"][0]
        result += f"Text: {source['text']}\nMatch Score: {source['match_score']}\nSource: {source['source']}\nPage Num: {source['page_num']}\n"

    return result


def get_sql_response(prompt, agent, db=None, table_name=None):
    """Executes a Text-to-SQL inference and returns the result."""
    show_sql = False
    if prompt.endswith(" #SHOW"):
        show_sql = True
        prompt = prompt[:-len(" #SHOW")]

    model_response = agent.query_custom_table(prompt, db=db, table=table_name)
    try:
        sql_query = model_response["sql_query"]
        db_response = model_response["db_response"]
        if show_sql:
            return f"Answer: {db_response}\n\nSQL Query: {sql_query}"
        return db_response
    except Exception:
        return f"Sorry, could not generate an answer. SQL: {model_response.get('sql_query', '')}"


def biz_bot_ui_app(db="postgres", table_name=None, fp=None, doc=None):
    """Main Biz Bot UI Application."""
    st.title("Biz Bot")

    parser_output = None
    if fp and doc and os.path.isfile(os.path.join(fp, doc)):
        parser_output = Parser().parse_one(fp, doc, save_history=False)

    prompter = load_prompt_model()
    reranker_model = load_reranker_model()
    agent = load_agent_model()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.sidebar:
        model_type = st.selectbox("Pick your mode", ("RAG", "SQL"))
        uploaded_doc = st.file_uploader("Upload Document")
        uploaded_table = st.file_uploader("Upload CSV")

        if uploaded_doc:
            doc = uploaded_doc.name
            save_uploaded_file(fp, uploaded_doc)
            parser_output = parse_file(fp, doc)

        if uploaded_table:
            tab = uploaded_table.name
            save_uploaded_file(fp, uploaded_table)
            table_name = tab.split(".")[0]
            row_count = build_table(db=db, table_name=table_name, load_fp=fp, load_file=tab)
            st.write(f"Completed - Table: {table_name} - Rows: {row_count}.")

    prompt = st.chat_input("Ask me something")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if model_type == "RAG":
                bot_response = get_rag_response(prompt, parser_output, reranker_model, prompter)
            else:
                bot_response = get_sql_response(prompt, agent, db=db, table_name=table_name)
            st.markdown(bot_response)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": bot_response})


def save_uploaded_file(fp, uploaded_file):
    """Helper function to save uploaded file to the server."""
    file_path = os.path.join(fp, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved file: {uploaded_file.name}")
