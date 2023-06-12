import tempfile
import yaml
import streamlit as st
from langchain.llms import OpenAI
from langchain.llms.openai import OpenAI
from langchain.agents.agent_toolkits.openapi import planner
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.requests import RequestsWrapper
from langchain.tools import OpenAPISpec, APIOperation
from langchain.chains import OpenAPIEndpointChain
from langchain.requests import Requests

api_key = st.text_input("Enter your API key", type = "password")

if api_key:
    llm = OpenAI(temperature=0.0, openai_api_key=api_key)
    spec = OpenAPISpec.from_url("https://www.klarna.com/us/shopping/public/openai/v0/api-docs/")
    operation = APIOperation.from_openapi_spec(spec, '/public/openai/v0/products', "get")
    chain = OpenAPIEndpointChain.from_api_operation(
        operation, 
        llm, 
        requests=Requests(), 
        verbose=True,
        return_intermediate_steps=True
    )

    query = st.text_input("Ask me something about your API!")

    if query:
        output = chain(query)
        st.success(output["output"])
else:
    st.warning("Enter your OPENAI API-KEY. Get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).\n")

# uploaded_file = st.file_uploader("Upload your Open API documentation file here (.yaml)")

# if uploaded_file:
#     bytes_data = uploaded_file.getvalue()

#     with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
#         temp_file.write(bytes_data)
#         temp_file.seek(0)
#         raw_api_spec = yaml.load(temp_file, Loader=yaml.Loader)
