import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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

CHART_GUIDE = """## Chart Selection Guide
1. Temporal Data: Line, Area, Candlestick, Gantt
2. Comparisons: Bar, Grouped Bar, Radar
3. Distributions: Histogram, Box, Violin, Density
4. Relationships: Scatter, Bubble, Heatmap, Network
5. Hierarchical: Treemap, Sunburst, Sankey
6. Geospatial: Use coordinates for maps
7. Proportions: Pie, Donut, Funnel

Always:
- Check data types first
- Handle missing values
- Add titles and labels
- Use appropriate color schemes
- Include legends where needed
"""

def extract_code(response_text):
    """Extracts Python code from AI response more robustly"""
    # Try finding code inside triple backticks
    match = re.search(r'```python(.*?)```', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no backticks, try extracting after a phrase
    match = re.search(r'(Here is the Python code:|Code:|Python code:)\s*(.*)', response_text, re.DOTALL)
    return match.group(2).strip() if match else None

def safe_execute_code(code: str, df: pd.DataFrame):
    """Execute code safely with enhanced validation and Plotly support"""
    # Security checks
    forbidden = ['pd.read_', 'open(', 'os.', 'sys.', 'exec', 'eval']
    for keyword in forbidden:
        if keyword in code:
            raise ValueError(f"Forbidden operation detected: {keyword}")

    # Create execution environment
    env = {
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'px': px,
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
    """Generate analysis code with chart-specific instructions"""
    system_prompt = f"""Analyze DataFrame with columns: {list(df.columns)}
{CHART_GUIDE}
Respond **ONLY** with a Python script inside:

# Your Python code here

If a chart is required, use Plotly or Matplotlib.
“””

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate Python code for: {query}. Only return the code inside triple backticks."}
    ],
    model=model,
    temperature=0.2,
    max_tokens=1500
)

return response.choices[0].message.content

def display_results(env, output):
“”“Handle multiple visualization types”””
# Matplotlib figures
figures = [plt.figure(i) for i in plt.get_fignums()]
for fig in figures:
st.pyplot(fig)
plt.close(‘all’)

# Plotly figures from environment
for var in env:
    if isinstance(env[var], (plt.Figure, px._figure.Figure)):
        if 'plotly' in str(type(env[var])):
            st.plotly_chart(env[var])
        else:
            st.pyplot(env[var])

def main():
st.title(“📊 Smart Chart Generator”)

# Model selection
selected_model = st.sidebar.selectbox("Choose AI Model", list(MODELS.keys()), index=0)

# File upload and processing
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.subheader("Data Summary")
        st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.sidebar.write("Columns:", df.columns.tolist())
        
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        return

# Analysis request
query = st.text_area("Analysis Request", 
                   "Generate appropriate visualizations to show key insights",
                   height=100)

if st.button("Generate Visualizations") and df is not None:
    with st.spinner("Creating analysis..."):
        try:
            # Generate analysis code
            code_response = generate_analysis_code(df, query, MODELS[selected_model])
            
            # Extract code block
            clean_code = extract_code(code_response)
            if not clean_code:
                st.error("No valid code block found in response")
                return
            
            # Display generated code
            with st.expander("Generated Analysis Code"):
                st.code(clean_code)
            
            # Execute code with safety features
            output, env = safe_execute_code(clean_code, df)
            
            # Display results
            st.subheader("Visualization Results")
            display_results(env, output)
            
            if output.strip():
                st.subheader("Execution Logs")
                st.text(output)
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.info("Common fixes: Check data types, column names, and sample data")

if name == “main”:
main()

### Steps to Use:

1. **Install Required Libraries:**
   - If you haven't installed the libraries required by this script yet, run:
     ```
     pip install streamlit pandas matplotlib seaborn plotly groq
     ```

2. **Run Streamlit:**
   - Save this script as a Python file (e.g., `app.py`).
   - Run the Streamlit app:
     ```
     streamlit run app.py
     ```

3. **Upload CSV and Query:**
   - Upload a CSV file in the sidebar and provide a query in the text area.
   - Click "Generate Visualizations" to see the visualizations generated by the AI.

---

This code should now work as intended with a more robust method for extracting Python code and a safer execution environment for generating and displaying the visualizations. Let me know if you need any further tweaks!
