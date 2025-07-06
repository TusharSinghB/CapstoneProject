# AI Data Analytics

A Streamlit app that uses Google Gemini AI to analyze your CSV data and create visualizations through natural language commands.

## Quick Start

1. **Install dependencies:**

```bash
pip install streamlit pandas matplotlib seaborn plotly langchain-experimental langchain-google-genai
```

2. **Get Google Gemini API key:**

    - Go to [Google AI Studio](https://aistudio.google.com/apikey)
    - Create an API key
    - Replace the API key in the code

3. **Run the app:**

```bash
streamlit run app.py
```

4. **Upload CSV and start analyzing!**

## Features

-   **Ask Questions**: "What's the average price by brand?"
-   **Generate Visualizations**: "Create a bar chart of sales by region"
-   **Data Analysis**: "Find correlation between price and mileage"

## Example Requests

### Visualizations

-   "Create a scatter plot of price vs year"
-   "Make a bar chart showing count by category"
-   "Generate a histogram of prices"

### Analysis

-   "Calculate average price by brand"
-   "Show summary statistics"
-   "Find cars with price above average"

## How It Works

1. Upload your CSV file
2. Choose analysis type (Questions, Visualizations, or Analysis)
3. Describe what you want in plain English
4. AI generates and runs Python code automatically
5. View results instantly

## Important Notes

-   **API Costs**: Monitor your Google Gemini API usage
-   **Data Privacy**: Your data is sent to Google's API for processing
-   **Security**: Use environment variables for API keys in production

## Troubleshooting

-   **API Error**: Check your Gemini API key
-   **Import Error**: Install all required packages
-   **Code Fails**: Verify column names match your data

---

**Simple AI-powered data analysis with natural language commands**
