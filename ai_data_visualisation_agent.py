import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from groq import Groq
import re

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
    """Extracts Python code from AI response with multiple fallback methods"""
    # Try finding code inside triple backticks (with or without language specifier)
    code_block = re.search(r'```(?:python)?(.*?)```', response_text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    # Try finding indented code blocks
    indented_code = re.search(r'(^|\n) +([^\n]+ *\n)+', response_text)
    if indented_code:
        return indented_code.group(0).strip()
    # Try finding code after specific markers
    markers = [
        r'Here(?: is| are) (?:the |a )?code(?: snippet)?:',
        r'Python code:',
        r'Generated code:',
        r'Solution:'
    ]
    for marker in markers:
        match = re.search(marker + r'\s*(.*)', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    # Final fallback
    return response_text.strip()

# Streamlit App Layout
st.title("Data Visualization Assistant")
st.write("Upload your dataset and get interactive visualizations!")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)
    st.write("Preview of your data:")
    st.dataframe(df.head())

    # Chart Type Selection
    chart_type = st.selectbox("Select a chart type", ["Line", "Bar", "Scatter", "Histogram", "Box"])

    # Generate Chart
    if st.button("Generate Chart"):
        try:
            if chart_type == "Line":
                fig = px.line(df, title="Line Chart")
            elif chart_type == "Bar":
                fig = px.bar(df, title="Bar Chart")
            elif chart_type == "Scatter":
                fig = px.scatter(df, title="Scatter Plot")
            elif chart_type == "Histogram":
                fig = px.histogram(df, title="Histogram")
            elif chart_type == "Box":
                fig = px.box(df, title="Box Plot")

            # Display the chart
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display Chart Guide
st.subheader("Need Help Selecting a Chart?")
st.markdown(CHART_GUIDE)
