import streamlit as st
import PyPDF2
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
import asyncio
import json
import re

st.set_page_config("Resume analyzer")
st.title("ATS Resume Checker")
st.header("Increase the chance to secure your dream job.", divider="grey")

jd = st.text_area("Paste job description", height=200)
upload_file = st.sidebar.file_uploader("Upload your resume", type="pdf", help="Please upload a PDF file")
submit = st.button("Submit")

def input_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return "".join(page.extract_text() for page in reader.pages)

def extract_json_from_response(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            return json.loads(json_match.group()) if json_match else None
        except:
            return None

async def analyze_resume():
    ollama_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    model = OpenAIChatCompletionsModel(model="deepseek-r1:1.5b", openai_client=ollama_client)
    set_tracing_disabled(disabled=True)

    agent = Agent(
        name="ATS Agent",
        instructions="""
        You are a skilled ATS system. You are an experienced Resume analyzer who's has 40 years experience in every tech field. Use your experience and analyze the resume against the job description carefully and provide:
        - Percentage match (e.g., "75%")
        - Missing keywords
        - Profile summary
        Return ONLY valid JSON format: {"##JD Match": "X%", "##Missing Keywords": [], "##Profile Summary": "..."}
        Keep your response short and complete you response within 100 words. Just be honest about your response because it is the question of company's policy and future i will tip you 20000 dollars for best satisfying responses. 
        """,
        model=model,
    )

    if not upload_file:
        st.error("Please upload resume")
        return

    text = input_pdf_text(upload_file)
    input_text = f"Evaluate resume:\n{text}\n\nJob Description:\n{jd}"
    
    result = Runner.run_streamed(starting_agent=agent, input=input_text)
    response_container = st.empty()  
    full_response = ""

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            delta = event.data.delta
            full_response += delta
            response_container.markdown(f"```json\n{full_response}\n```")

    # Final processing
    response_json = extract_json_from_response(full_response)
    display_results(response_json if response_json else full_response)

def display_results(data):
    st.subheader("Analysis Results")
    
    if isinstance(data, dict): 
        st.metric("JD Match", data.get("##JD Match", "N/A"))
        
        st.write("**Missing Keywords:**")
        if keywords := data.get("##Missing Keywords", []):
            st.table(keywords)
        else:
            st.write("No missing keywords found")
        
        st.write("**Profile Summary:**")
        st.write(data.get("##Profile Summary", "N/A"))
    else:  # Raw response
        st.error("Could not parse response. Raw output:")
        st.code(data)

if __name__ == "__main__":
    if submit:
        asyncio.run(analyze_resume())