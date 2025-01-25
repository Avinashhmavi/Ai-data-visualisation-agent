### AI Data Visualization Agent 🤖📊

An intelligent data analysis tool that leverages AI models to automatically generate visualizations and insights from uploaded CSV data.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-data-visualisation-agent.streamlit.app/)

**Live App:** [https://ai-data-visualisation-agent.streamlit.app/](https://ai-data-visualisation-agent.streamlit.app/)


## Features ✨
- Supports multiple AI models (Llama3-70B, Mixtral-8x7B, Gemma-7B)
- Automatic data type detection and conversion
- Secure code execution environment
- Interactive visualizations with Matplotlib/Seaborn
- Real-time code generation and execution
- Streamlit-based web interface

## Prerequisites 📦
- Python 3.8+
- Streamlit Account (for deployment)
- [Groq API Key](https://console.groq.com/) (free tier available)

## Installation 🛠️

1. **Clone Repository**
```bash
git clone https://github.com/<your-repo>/ai-data-visualization-agent.git
cd ai-data-visualization-agent
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Set Environment Variables**  
Create `.env` file:
```env
GROQ_API_KEY=your_api_key_here
```

4. **Run Locally**
```bash
streamlit run ai_data_visualisation_agent.py
```

## Configuration ⚙️
1. Get Groq API Key:
   - Sign up at [Groq Cloud](https://console.groq.com/)
   - Create new API key in dashboard
   - Add to `.env` file

2. Supported Models:
   - `llama3-70b-8192`
   - `mixtral-8x7b-32768` 
   - `gemma-7b-it`

## Usage Guide 🖥️
1. Upload CSV file (≤200MB)
2. Select AI model from dropdown
3. Enter analysis request (e.g., "Show sales trends")
4. View generated visualizations
5. Inspect auto-generated code in expandable section


## Deployment ☁️
Deployed on Streamlit Cloud using:
```bash
requirements.txt
seaborn==0.13.2
matplotlib==3.8.4
pandas==2.2.2
streamlit==1.35.0
groq==0.3.0
python-dotenv==1.0.1
```

## Data Privacy 🔒
- All processing happens in-memory
- No data stored after session ends
- Groq API interactions are encrypted

---

**Need Help?**  
Contact: avi.hm24@gmail.com
