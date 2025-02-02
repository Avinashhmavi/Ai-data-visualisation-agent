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

# Chart Templates
CHART_TEMPLATES = {
    "Bar Chart": """
plt.figure(figsize=(10, 6))
sns.barplot(x='{x_column}', y='{y_column}', data=df)
plt.title('Bar Chart')
plt.show()
""",
    "Stacked Bar Chart": """
df_grouped = df.groupby(['{x_column}', '{stack_column}']).size().unstack()
df_grouped.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Stacked Bar Chart')
plt.show()
""",
    "Line Chart": """
plt.figure(figsize=(10, 6))
sns.lineplot(x='{x_column}', y='{y_column}', data=df)
plt.title('Line Chart')
plt.show()
""",
    "Scatter Plot": """
plt.figure(figsize=(10, 6))
sns.scatterplot(x='{x_column}', y='{y_column}', data=df)
plt.title('Scatter Plot')
plt.show()
""",
    "Heatmap": """
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap')
plt.show()
""",
    "Histogram": """
plt.figure(figsize=(10, 6))
sns.histplot(df['{column}'], kde=False, bins=20)
plt.title('Histogram')
plt.show()
""",
    "Box Plot": """
plt.figure(figsize=(10, 6))
sns.boxplot(x='{x_column}', y='{y_column}', data=df)
plt.title('Box Plot')
plt.show()
""",
    "Pie Chart": """
plt.figure(figsize=(8, 8))
df['{column}'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()
""",
    "Violin Plot": """
plt.figure(figsize=(10, 6))
sns.violinplot(x='{x_column}', y='{y_column}', data=df)
plt.title('Violin Plot')
plt.show()
""",
    "Density Plot": """
plt.figure(figsize=(10, 6))
sns.kdeplot(df['{column}'], shade=True)
plt.title('Density Plot')
plt.show()
""",
    # Add more templates here...
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


def parse_query(query: str, df: pd.DataFrame):
    """Parse user query to identify chart type and relevant columns"""
    query_lower = query.lower()
    chart_type = None
    x_column = None
    y_column = None
    column = None
    
    # Identify chart type
    for chart in CHART_TEMPLATES.keys():
        if chart.lower() in query_lower:
            chart_type = chart
            break
    
    if not chart_type:
        raise ValueError("Chart type not recognized in query.")
    
    # Identify relevant columns
    for col in df.columns:
        if col.lower() in query_lower:
            if not x_column:
                x_column = col
            elif not y_column:
                y_column = col
            else:
                break
    
    # Fallback: Use first two numeric columns if no match
    if not x_column or not y_column:
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        x_column = numeric_cols[0] if len(numeric_cols) > 0 else None
        y_column = numeric_cols[1] if len(numeric_cols) > 1 else None
    
    return chart_type, x_column, y_column


def generate_chart_code(chart_type: str, x_column: str, y_column: str, column: str):
    """Generate Python code for the requested chart"""
    if chart_type not in CHART_TEMPLATES:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    template = CHART_TEMPLATES[chart_type]
    code = template.format(x_column=x_column, y_column=y_column, column=column)
    return code


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
    query = st.text_area("Analysis Request", "Show me a bar chart of sales by category", height=100)
    if st.button("Analyze") and df is not None:
        with st.spinner("Generating analysis..."):
            try:
                # Parse query
                chart_type, x_column, y_column = parse_query(query, df)
                
                # Generate chart code
                code = generate_chart_code(chart_type, x_column, y_column, column=None)
                
                # Display generated code
                with st.expander("Generated Code"):
                    st.code(code)
                
                # Execute code with safety features
                output, env = safe_execute_code(code, df)
                
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
