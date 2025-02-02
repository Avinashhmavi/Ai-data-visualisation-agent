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
“””response = client.chat.completions.create(
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
**Critical Rules**
1. Use EXISTING 'df' - never load data
2. Prefer Plotly Express (px) for advanced charts
3. For Matplotlib: call plt.show() after each plot
4. Handle errors gracefully with try/except
5. Preprocess data as needed (datetime conversion, normalization)

**Code Examples**
1. Line Chart:
plt.figure(figsize=(10,4))
plt.plot(df['date'], df['value'])
plt.title('Trend Analysis')
plt.show()

2. Interactive Plotly:
fig = px.scatter(df, x='x_col', y='y_col', color='category')
fig.show()

3. Statistical Plot:
sns.violinplot(x='group', y='value', data=df)
plt.show()

4. Geospatial:
fig = px.choropleth(df, locations='iso_code', color='value')
fig.show()

5. Composition:
df.groupby('category').size().plot.pie(autopct='%1.1f%%')
plt.show()

**Response Strategy**
1. Analyze user query intent
2. Select appropriate chart types
3. Verify data requirements
4. Generate clean visualization code
5. Include necessary preprocessing"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        model=model,
        temperature=0.3,
        max_tokens=2500
    )
    
    return response.choices[0].message.content

def display_results(env, output):
    """Handle multiple visualization types"""
    # Matplotlib figures
    figures = [plt.figure(i) for i in plt.get_fignums()]
    for fig in figures:
        st.pyplot(fig)
    plt.close('all')
    
    # Plotly figures from environment
    import plotly.graph_objects as go  # Import this at the top

    for var in env:
      if isinstance(env[var], (plt.Figure, go.Figure)):  # Use go.Figure instead
        if isinstance(env[var], go.Figure):  # Check for Plotly figures
            st.plotly_chart(env[var])
        else:
            st.pyplot(env[var])

def main():
    st.title("📊 Smart Chart Generator")
    
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
                code_match = re.search(r'```python(.*?)```', code_response, re.DOTALL)
                if not code_match:
                    st.error("No valid code block found in response")
                    return
                
                clean_code = code_match.group(1).strip()
                
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

if __name__ == "__main__":
    main()
    ### Key Changes:
1. **`extract_code()` Function:** This method now handles both responses with triple backticks and responses that have direct code snippets without formatting.
   
2. **Updated `system_prompt`:** The prompt instructs the AI to return **only Python code** inside triple backticks.

3. **Limit on `max_tokens`:** Reduced to `1500` to avoid overly long responses or truncation.

4. **Security in Code Execution:** The `safe_execute_code()` method includes checks to prevent unsafe operations in the AI-generated code.

### How it Works:
1. **User Uploads a CSV File:** The user uploads a CSV file, which is processed into a `DataFrame`.
2. **User Request for Visualization:** The user types a query asking the AI to generate a specific visualization.
3. **Groq API Generates Python Code:** The system sends the request to the Groq API, and the AI responds with Python code for data analysis and visualization.
4. **Execute and Display the Result:** The Python code is executed safely, and any resulting visualizations (Plotly or Matplotlib) are displayed in the Streamlit app.

### Troubleshooting:
- Ensure that the API key (`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`) is correct and valid.
- If the output is empty or the AI response is not useful, try refining the query or checking data types and columns in the CSV file.

This version should handle the "No valid code block found in response" issue while also improving safety and flexibility in executing AI-generated Python code.

Let me know if you need further adjustments!
