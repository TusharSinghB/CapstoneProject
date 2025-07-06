import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import re
import traceback

# LangChain imports for AI-powered data analysis
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Google Gemini API key configuration
# Note: In production, use environment variables or Streamlit secrets for API keys
GEMINI_API_KEY = "Paste your API Key Here"

# Main application title
st.title("AI Data Analytics - Code Generation & Execution")

# File upload widget - accepts CSV files only
uploaded_file = st.file_uploader("Upload your file", type=["csv"])

# Main application logic - only runs when a file is uploaded
if uploaded_file is not None:
    # Load the uploaded CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display preview of the data
    st.write("First 5 rows of DataFrame")
    st.write(df.head())
    
    # Initialize the Large Language Model (LLM)
    @st.cache_resource  # Cache the LLM instance to avoid repeated initialization
    def get_llm():
        """
        Initialize and return a Google Gemini LLM instance.
        Uses caching to prevent unnecessary re-initialization.
        """
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY
        )
    
    # Get the cached LLM instance
    llm = get_llm()
    
    # Analysis type selection dropdown
    analysis_type = st.selectbox(
        "Choose analysis type:",
        ["Ask a question", "Generate visualization code", "Generate analysis code"]
    )
    
    # ========================================
    # QUESTION ANSWERING MODE
    # ========================================
    if analysis_type == "Ask a question":
        # Text input for user questions
        question = st.text_input("Enter your Question")
        
        if question:
            # Create a pandas DataFrame agent that can answer questions about the data
            agent = create_pandas_dataframe_agent(
                llm, df, verbose=True, allow_dangerous_code=True
            )
            
            # Process the question with loading spinner
            with st.spinner("Analyzing...."):
                try:
                    # Run the agent to get an answer
                    answer = agent.run(question)
                    st.write("**Answer:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # ========================================
    # VISUALIZATION CODE GENERATION MODE
    # ========================================
    elif analysis_type == "Generate visualization code":
        st.subheader("AI Code Generation for Visualizations")
        
        # Display dataset metadata to help users make informed requests
        st.write("**Available columns:**", ", ".join(df.columns.tolist()))
        st.write("**Dataset shape:**", df.shape)
        
        # Text area for visualization requests
        viz_request = st.text_area(
            "Describe the visualization you want:",
            placeholder="e.g., Create a bar chart showing average price by brand with colors"
        )
        
        if viz_request:
            # Create a detailed prompt template for code generation
            code_prompt = PromptTemplate(
                input_variables=["request", "columns", "sample_data"],
                template="""
                Generate Python code to create a visualization based on this request: {request}
                
                Available columns: {columns}
                Sample data (first 3 rows): {sample_data}
                
                Requirements:
                1. Use the variable 'df' for the dataframe
                2. Use matplotlib (plt) or plotly.express (px) for visualization
                3. Include proper titles, labels, and styling
                4. For matplotlib: use plt.figure(figsize=(10, 6)) and plt.show()
                5. For plotly: create the figure 
                6. Handle any potential errors (missing columns, data types)
                7. Return ONLY the Python code, no explanations
                
                Example output format:
                ```python
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                # your visualization code here
                plt.title('Your Title')
                plt.xlabel('X Label')
                plt.ylabel('Y Label')
                plt.show()
                ```

                
                Code:
                """
            )
            
            # Create LLM chain for code generation
            chain = LLMChain(llm=llm, prompt=code_prompt)
            
            # Generate code with loading spinner
            with st.spinner("Generating code..."):
                try:
                    # Run the chain to generate code
                    result = chain.run(
                        request=viz_request,
                        columns=", ".join(df.columns.tolist()),
                        sample_data=df.head(3).to_string()
                    )
                    
                    # Extract Python code from the LLM response
                    # Look for code blocks wrapped in ```python ... ```
                    code_match = re.search(r'```python\n(.*?)```', result, re.DOTALL)
                    if code_match:
                        generated_code = code_match.group(1).strip()
                    else:
                        # If no code block found, use the entire response
                        generated_code = result.strip()
                    
                    # Auto-execute the generated code (comment out these lines to show code first)
                    # st.subheader("Generated Code:")
                    # st.code(generated_code, language='python')
                    
                    # Execute the code automatically
                    # if st.button("🚀 Execute Code"):
                    if True:  # Auto-execute without button click
                        # st.subheader("Execution Result:")
                        
                        # Create a safe execution environment with required libraries
                        exec_globals = {
                            'df': df,
                            'pd': pd,
                            'plt': plt,
                            'sns': sns,
                            'px': px,
                            'go': go,
                            'st': st
                        }
                        
                        try:
                            # Execute the generated code in the safe environment
                            exec(generated_code, exec_globals)
                            
                            # Handle matplotlib plots
                            if 'plt.show()' in generated_code:
                                st.pyplot(plt.gcf())  # Display the current figure
                                plt.close()  # Close the figure to free memory
                            
                            # Handle plotly plots (if a 'fig' variable was created)
                            if 'fig' in exec_globals:
                                st.plotly_chart(exec_globals['fig'], use_container_width=True)
                            
                            st.success("Code executed successfully!")
                            
                        except Exception as e:
                            # Handle execution errors gracefully
                            st.error(f"Error executing code: {str(e)}")
                            st.write("**Error details:**")
                            st.code(traceback.format_exc())
                            
                            # Provide helpful suggestions for common issues
                            st.write("**Possible fixes:**")
                            st.write("1. Check if all column names are correct")
                            st.write("2. Verify data types are compatible")
                            st.write("3. Try a simpler visualization request")
                    
                except Exception as e:
                    st.error(f"Error generating code: {e}")
    
    # ========================================
    # DATA ANALYSIS CODE GENERATION MODE
    # ========================================
    elif analysis_type == "Generate analysis code":
        st.subheader("AI Code Generation for Data Analysis")
        
        # Text area for analysis requests
        analysis_request = st.text_area(
            "Describe the analysis you want:",
            placeholder="e.g., Find correlation between price and year, group by brand and calculate statistics"
        )
        
        if analysis_request:
            # Create prompt template for analysis code generation
            analysis_prompt = PromptTemplate(
                input_variables=["request", "columns", "sample_data"],
                template="""
                Generate Python code for data analysis based on this request: {request}
                
                Available columns: {columns}
                Sample data: {sample_data}
                
                Requirements:
                1. Use the variable 'df' for the dataframe
                2. Use pandas operations for analysis
                3. Print results clearly with descriptive text
                4. Handle missing values if necessary
                5. Return ONLY the Python code, no explanations
                
                Code:
                """
            )
            
            # Create LLM chain for analysis code generation
            chain = LLMChain(llm=llm, prompt=analysis_prompt)
            
            # Generate analysis code with loading spinner
            with st.spinner("Generating analysis code..."):
                try:
                    # Run the chain to generate analysis code
                    result = chain.run(
                        request=analysis_request,
                        columns=", ".join(df.columns.tolist()),
                        sample_data=df.head(3).to_string()
                    )
                    
                    # Clean the generated code
                    generated_code = result.strip()
                    
                    # Display the generated code for review
                    st.subheader("Generated Analysis Code:")
                    st.code(generated_code, language='python')
                    
                    # Execute analysis code when button is clicked
                    if st.button("🔍 Run Analysis"):
                        st.subheader("Analysis Results:")
                        
                        # Capture print output from the executed code
                        import io
                        import sys
                        
                        # Redirect stdout to capture print statements
                        old_stdout = sys.stdout
                        sys.stdout = buffer = io.StringIO()
                        
                        # Create execution environment for analysis code
                        exec_globals = {
                            'df': df,
                            'pd': pd,
                            'print': print
                        }
                        
                        try:
                            # Execute the analysis code
                            exec(generated_code, exec_globals)
                            
                            # Get the captured output
                            output = buffer.getvalue()
                            sys.stdout = old_stdout  # Restore stdout
                            
                            # Display the output
                            if output:
                                st.text(output)
                            else:
                                st.info("Analysis completed (no output to display)")
                            
                            st.success("Analysis executed successfully!")
                            
                        except Exception as e:
                            # Restore stdout and handle errors
                            sys.stdout = old_stdout
                            st.error(f"Error executing analysis: {str(e)}")
                            st.code(traceback.format_exc())
                
                except Exception as e:
                    st.error(f"Error generating analysis code: {e}")
    
    # ========================================
    # HELP SECTION - EXAMPLE REQUESTS
    # ========================================
    with st.expander("💡 Example Requests"):
        st.write("**Visualization Examples:**")
        st.write("• Create a scatter plot of price vs year colored by brand")
        st.write("• Make a bar chart showing count of cars by brand")
        st.write("• Generate a histogram of car prices with 20 bins")
        st.write("• Create a box plot of prices grouped by condition")
        st.write("• Show a correlation heatmap of numeric columns")
        
        st.write("**Analysis Examples:**")
        st.write("• Calculate average price by brand and year")
        st.write("• Find the correlation between price and mileage")
        st.write("• Show summary statistics for each brand")
        st.write("• Identify cars with price above average")
        st.write("• Group by condition and show count and average price")
    
    # ========================================
    # DATASET INFORMATION SECTION
    # ========================================
    with st.expander("📊 Dataset Information"):
        st.write("**Columns and Data Types:**")
        st.write(df.dtypes)
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())
        st.write("**Basic Statistics:**")
        st.write(df.describe())