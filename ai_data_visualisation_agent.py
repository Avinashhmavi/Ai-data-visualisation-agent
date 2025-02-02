import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
import re
import sys
from io import StringIO
import contextlib

# Initialize Groq client - replace with your API key
client = Groq(api_key="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")

MODELS = {
    "Llama3-70B": "llama3-70b-8192",
    "Mixtral-8x7B": "mixtral-8x7b-32768",
    "Gemma-7B": "gemma-7b-it"
}

def safe_execute_code(code: str, df: pd.DataFrame):
    """Execute code safely with enhanced validation"""
    # Remove any file reading operations
    code = re.sub(r'pd\.read_csv\(.*?\)', 'df', code)
    
    # Security checks
    forbidden = ['pd.read_csv', 'pd.read_excel', 'open(', 'os.', 'sys.']
    for keyword in forbidden:
        if keyword in code:
            raise ValueError(f"Forbidden operation detected: {keyword}")

    # Create execution environment
    env = {
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'df': df.copy(),
        '__builtins__': {**__builtins__, 'open': None}
    }

    # Capture output
    output = StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            exec(code, env)
        except Exception as e:
            raise RuntimeError(f"Execution error: {str(e)}")
    
    return output.getvalue(), env

def generate_analysis_code(df: pd.DataFrame, query: str, model: str):
    """Generate analysis code with strict instructions"""
    system_prompt = f"""You are analyzing a DataFrame called 'df' with columns: {list(df.columns)}
    
    Important Rules:
    1. Use the existing 'df' variable - DO NOT load data from files
    2. Never use pd.read_csv() or any file loading functions
    3. Create visualizations using plt.show()
    4. Include proper error handling
    
    Example Code Structure:
    ```python
    # Data cleaning
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except:
        pass
    
    # Visualization
    plt.figure(figsize=(10,6))
    sns.countplot(x='Category', data=df)
    plt.show()
    ```"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        model=model,
        temperature=0.3,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

def main():
    st.title("📊 Safe Data Analyst")
    
    # Model selection
    selected_model = st.selectbox("Choose Model", list(MODELS.keys()), index=0)
    
    # File upload and processing
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    df = None
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head(3))
            
            st.subheader("Data Types")
            st.write(df.dtypes.astype(str))
            
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return

    # Analysis request
    query = st.text_area("Analysis Request", "Show key insights through visualizations", height=100)

    if st.button("Analyze") and df is not None:
        with st.spinner("Generating analysis..."):
            try:
                # Generate analysis code
                code_response = generate_analysis_code(df, query, MODELS[selected_model])
                
                # Clean and validate code
                code = re.search(r'```python(.*?)```', code_response, re.DOTALL)
                if not code:
                    st.error("No valid code found in response")
                    return
                
                clean_code = code.group(1).strip()
                clean_code = re.sub(r'pd\.read_csv\(.*?\)', '# Removed file loading', clean_code)
                
                # Display generated code
                with st.expander("Generated Code"):
                    st.code(clean_code)
                
                # Execute code with safety features
                output, env = safe_execute_code(clean_code, df)
                
                # Display results
                st.subheader("Results")
                if 'plt' in env:
                    figures = [plt.figure(i) for i in plt.get_fignums()]
                    for fig in figures:
                        st.pyplot(fig)
                    plt.close('all')
                
                if output.strip():
                    st.subheader("Code Output")
                    st.text(output)
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("Common fixes: The AI might have tried to access forbidden operations")

if __name__ == "__main__":
    main()
