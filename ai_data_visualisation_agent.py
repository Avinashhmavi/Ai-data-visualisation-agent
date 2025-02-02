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
    # Basic Charts
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
    "Grouped Bar Chart": """
df_grouped = df.groupby(['{x_column}', '{group_column}']).size().unstack()
df_grouped.plot(kind='bar', figsize=(10, 6))
plt.title('Grouped Bar Chart')
plt.show()
""",
    "Line Chart": """
plt.figure(figsize=(10, 6))
sns.lineplot(x='{x_column}', y='{y_column}', data=df)
plt.title('Line Chart')
plt.show()
""",
    "Area Chart": """
plt.figure(figsize=(10, 6))
df.plot.area(x='{x_column}', y='{y_column}', stacked=False, figsize=(10, 6))
plt.title('Area Chart')
plt.show()
""",
    "Stacked Area Chart": """
plt.figure(figsize=(10, 6))
df.plot.area(x='{x_column}', y='{y_column}', stacked=True, figsize=(10, 6))
plt.title('Stacked Area Chart')
plt.show()
""",
    "Pie Chart": """
plt.figure(figsize=(8, 8))
df['{column}'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()
""",
    "Donut Chart": """
plt.figure(figsize=(8, 8))
df['{column}'].value_counts().plot.pie(autopct='%1.1f%%', wedgeprops=dict(width=0.3))
plt.title('Donut Chart')
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

    # Advanced Statistical Charts
    "Scatter Plot": """
plt.figure(figsize=(10, 6))
sns.scatterplot(x='{x_column}', y='{y_column}', data=df)
plt.title('Scatter Plot')
plt.show()
""",
    "Bubble Chart": """
plt.figure(figsize=(10, 6))
sns.scatterplot(x='{x_column}', y='{y_column}', size='{size_column}', data=df)
plt.title('Bubble Chart')
plt.show()
""",
    "Heatmap": """
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap')
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
    "QQ Plot": """
import scipy.stats as stats
plt.figure(figsize=(10, 6))
stats.probplot(df['{column}'], dist="norm", plot=plt)
plt.title('QQ Plot')
plt.show()
""",
    "Error Bar Chart": """
plt.figure(figsize=(10, 6))
plt.errorbar(df['{x_column}'], df['{y_column}'], yerr=df['{error_column}'], fmt='o')
plt.title('Error Bar Chart')
plt.show()
""",
    "Radar Chart": """
import numpy as np
categories = list(df.columns)
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
values = df.iloc[0].values.flatten().tolist()
values += values[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='red', alpha=0.25)
plt.title('Radar Chart')
plt.show()
""",
    "Pareto Chart": """
import matplotlib.ticker as ticker
df_sorted = df['{column}'].value_counts().sort_values(ascending=False)
cumsum = df_sorted.cumsum() / df_sorted.sum() * 100

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(df_sorted.index, df_sorted, color='blue')
ax1.set_ylabel('Frequency')

ax2 = ax1.twinx()
ax2.plot(df_sorted.index, cumsum, color='orange', marker='o', linestyle='-')
ax2.set_ylabel('Cumulative Percentage')
ax2.yaxis.set_major_formatter(ticker.PercentFormatter())

plt.title('Pareto Chart')
plt.show()
""",
    "Waterfall Chart": """
import waterfall_chart
plt.figure(figsize=(10, 6))
waterfall_chart.plot(df['{x_column}'], df['{y_column}'])
plt.title('Waterfall Chart')
plt.show()
""",

    # Time-Series Charts
    "Candlestick Chart": """
import mplfinance as mpf
plt.figure(figsize=(10, 6))
mpf.plot(df, type='candle', style='charles', title='Candlestick Chart')
plt.show()
""",
    "OHLC Chart": """
import mplfinance as mpf
plt.figure(figsize=(10, 6))
mpf.plot(df, type='ohlc', style='charles', title='OHLC Chart')
plt.show()
""",
    "Gantt Chart": """
import plotly.express as px
fig = px.timeline(df, x_start='{start_column}', x_end='{end_column}', y='{task_column}')
fig.update_yaxes(categoryorder='total ascending')
fig.show()
""",
    "Calendar Heatmap": """
import calmap
plt.figure(figsize=(10, 6))
calmap.calendarplot(df['{date_column}'], daylabels='MTWTFSS', cmap='Reds')
plt.title('Calendar Heatmap')
plt.show()
""",
    "Streamgraph": """
import plotly.graph_objects as go
fig = go.Figure()
for col in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', stackgroup='one'))
fig.update_layout(title='Streamgraph')
fig.show()
""",
    "Horizon Chart": """
import matplotlib.colors as mcolors
plt.figure(figsize=(10, 6))
horizon_data = df['{column}'].values.reshape(-1, 10)
cmap = mcolors.LinearSegmentedColormap.from_list('custom', ['red', 'white', 'green'])
plt.imshow(horizon_data, aspect='auto', cmap=cmap)
plt.title('Horizon Chart')
plt.show()
""",
    "Cycle Plot": """
plt.figure(figsize=(10, 6))
sns.lineplot(x='{x_column}', y='{y_column}', hue='{hue_column}', data=df)
plt.title('Cycle Plot')
plt.show()
""",
    "Step Chart": """
plt.figure(figsize=(10, 6))
plt.step(df['{x_column}'], df['{y_column}'], where='post')
plt.title('Step Chart')
plt.show()
""",
    "Fan Chart": """
import matplotlib.patches as patches
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(len(df)):
    ax.add_patch(patches.Wedge((0, 0), df['{radius_column}'][i], df['{start_angle_column}'][i], df['{end_angle_column}'][i]))
plt.title('Fan Chart')
plt.show()
""",
    "Sunburst Chart": """
import plotly.express as px
fig = px.sunburst(df, path=['{level1_column}', '{level2_column}'], values='{value_column}')
fig.update_layout(title='Sunburst Chart')
fig.show()
""",

    # Geospatial Charts
    "Choropleth Map": """
import plotly.express as px
fig = px.choropleth(df, locations='{location_column}', locationmode='country names', color='{color_column}')
fig.update_layout(title='Choropleth Map')
fig.show()
""",
    "Heatmap Overlay": """
import folium
from folium.plugins import HeatMap
m = folium.Map(location=[df['{lat_column}'].mean(), df['{lon_column}'].mean()], zoom_start=10)
HeatMap(data=df[['{lat_column}', '{lon_column}']].values.tolist()).add_to(m)
m
""",
    "Dot Density Map": """
import geopandas as gpd
import matplotlib.pyplot as plt
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['{lon_column}'], df['{lat_column}']))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
base = world.plot(color='white', edgecolor='black')
gdf.plot(ax=base, color='red', markersize=5)
plt.title('Dot Density Map')
plt.show()
""",
    "Flow Map": """
import geopandas as gpd
import matplotlib.pyplot as plt
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['{lon_column}'], df['{lat_column}']))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
base = world.plot(color='white', edgecolor='black')
gdf.plot(ax=base, color='blue', markersize=5)
plt.title('Flow Map')
plt.show()
""",
    "Cartogram": """
import geopandas as gpd
import matplotlib.pyplot as plt
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['{lon_column}'], df['{lat_column}']))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
base = world.plot(color='white', edgecolor='black')
gdf.plot(ax=base, color='green', markersize=5)
plt.title('Cartogram')
plt.show()
""",
    "Hexbin Map": """
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hexbin(df['{lon_column}'], df['{lat_column}'], gridsize=30, cmap='Blues')
plt.title('Hexbin Map')
plt.show()
""",
    "3D Globe": """
import pyvista as pv
sphere = pv.Sphere(radius=1.0)
sphere.plot(texture='earth.jpg', title='3D Globe')
""",
    "Topographic Map": """
import matplotlib.pyplot as plt
plt.contourf(df['{x_column}'], df['{y_column}'], df['{z_column}'], cmap='terrain')
plt.colorbar(label='Elevation')
plt.title('Topographic Map')
plt.show()
""",
    "Cluster Map": """
import geopandas as gpd
import matplotlib.pyplot as plt
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['{lon_column}'], df['{lat_column}']))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
base = world.plot(color='white', edgecolor='black')
gdf.plot(ax=base, color='purple', markersize=5)
plt.title('Cluster Map')
plt.show()
""",
    "Route Map": """
import geopandas as gpd
import matplotlib.pyplot as plt
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['{lon_column}'], df['{lat_column}']))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
base = world.plot(color='white', edgecolor='black')
gdf.plot(ax=base, color='orange', markersize=5)
plt.title('Route Map')
plt.show()
""",

    # Specialized Charts
    "Sankey Diagram": """
import plotly.graph_objects as go
fig = go.Figure(data=[go.Sankey(
    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=df['{label_column}']),
    link=dict(source=df['{source_column}'], target=df['{target_column}'], value=df['{value_column}'])
)])
fig.update_layout(title='Sankey Diagram')
fig.show()
""",
    "Treemap": """
import plotly.express as px
fig = px.treemap(df, path=['{path_column}'], values='{value_column}')
fig.update_layout(title='Treemap')
fig.show()
""",
    "Network Graph": """
import networkx as nx
G = nx.from_pandas_edgelist(df, source='{source_column}', target='{target_column}')
nx.draw(G, with_labels=True)
plt.title('Network Graph')
plt.show()
""",
    "Chord Diagram": """
import matplotlib.pyplot as plt
import numpy as np
matrix = df.values
fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(matrix, cmap='viridis')
plt.title('Chord Diagram')
plt.show()
""",
    "Parallel Coordinates": """
from pandas.plotting import parallel_coordinates
plt.figure(figsize=(10, 6))
parallel_coordinates(df, class_column='{class_column}', colormap='viridis')
plt.title('Parallel Coordinates')
plt.show()
""",
    "Word Cloud": """
from wordcloud import WordCloud
text = ' '.join(df['{text_column}'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')
plt.show()
""",
    "Funnel Chart": """
import plotly.graph_objects as go
fig = go.Figure(go.Funnel(y=df['{stage_column}'], x=df['{value_column}']))
fig.update_layout(title='Funnel Chart')
fig.show()
""",
    "Pyramid Chart": """
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(df['{category_column}'], df['{positive_column}'], color='blue', label='Positive')
plt.barh(df['{category_column}'], -df['{negative_column}'], color='red', label='Negative')
plt.title('Pyramid Chart')
plt.legend()
plt.show()
""",
    "Polar Chart": """
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.polar(df['{angle_column}'], df['{radius_column}'], marker='o')
plt.title('Polar Chart')
plt.show()
"""
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

    # Normalize chart names and map synonyms
    CHART_SYNONYMS = {
        "bar chart": ["bar chart", "bar graph", "barplot"],
        "line chart": ["line chart", "line graph", "lineplot"],
        "scatter plot": ["scatter plot", "scatter chart", "scatter graph"],
        "pie chart": ["pie chart", "pie graph", "pieplot"],
        "histogram": ["histogram", "distribution plot"],
        "box plot": ["box plot", "box chart", "boxplot"],
        "heatmap": ["heatmap", "correlation matrix"],
        "donut chart": ["donut chart", "donut plot"],
        "stacked bar chart": ["stacked bar chart", "stacked bar graph"],
        "grouped bar chart": ["grouped bar chart", "grouped bar graph"],
        # Add more synonyms as needed...
    }

    # Identify chart type
    for chart, synonyms in CHART_SYNONYMS.items():
        if any(synonym in query_lower for synonym in synonyms):
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
        categorical_cols = [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]

        if chart_type in ["bar chart", "pie chart", "donut chart"]:
            x_column = categorical_cols[0] if len(categorical_cols) > 0 else None
            y_column = numeric_cols[0] if len(numeric_cols) > 0 else None
        elif chart_type in ["scatter plot", "line chart", "box plot"]:
            x_column = numeric_cols[0] if len(numeric_cols) > 0 else None
            y_column = numeric_cols[1] if len(numeric_cols) > 1 else None
        else:
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
