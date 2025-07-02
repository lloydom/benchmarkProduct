import streamlit as st
from google.cloud import bigquery
from google.cloud import aiplatform
import plotly.express as px
import plotly.figure_factory as ff
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
import pandas as pd
import logging
import json
from google.oauth2 import service_account

# Setup logging (keep this at the top as it's independent of login status)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Define allowed email IDs ---
ALLOWED_EMAILS = [
    "lloydom@gmail.com",
    "kaushalsharma0077@gmail.com",
    "menon11@gmail.com",
    "lara.monteiro@gmail.com"
]

def login_screen():
    # Create two columns: one for the title (wide) and one for the button (narrow)
    col1, col2 = st.columns([3, 1])  # Adjust ratio for title width vs button width

    # Title in the first column, styled like the reference site
    with col1:
        st.markdown(
            """
            <h1 style='text-align: left; color: #1E1E1E; font-family: "Helvetica Neue", Arial, sans-serif; 
                       font-size: 36px; font-weight: 700; line-height: 1.2; margin: 0;'>
                Competitive Analysis Tool. Compare data between Apar and its peers and generate analysis.
            </h1>
            """,
            unsafe_allow_html=True
        )

    # Button in the second column, styled to match the reference
    with col2:
        st.markdown(
            """
            <style>
            div.stButton > button {
                background-color: #FF4B4B;
                color: white;
                font-size: 16px;
                font-family: "Helvetica Neue", Arial, sans-serif;
                font-weight: 500;
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                transition: background-color 0.3s, transform 0.2s;
                margin-top: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            div.stButton > button:hover {
                background-color: #E63939;
                transform: translateY(-2px);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.button("Log in with Google...", on_click=st.login)

    # Stop further execution
    st.stop()

if not st.user.is_logged_in:
    login_screen()
else:
    user_email = st.user.get('email')
    if user_email and user_email not in ALLOWED_EMAILS:
        st.error("You do not have access to this application. Please contact the administrator.")
        st.button("Log out", on_click=st.logout)
        st.stop()

    # All your application content goes inside this else block
    st.header(f"Welcome, {st.user.name}!")
    st.button("Log out", on_click=st.logout)

    # Configuration
    PROJECT_ID = st.secrets.gcp.project_id
    DATASET_ID = 'my_dataset'  # Your BigQuery dataset ID
    COMPANY_1_TICKER = 'APARINDS.BO'
    PEER_TICKERS = ['FINCABLES.BO', 'KEI.BO', 'HAVELLS.BO', 'POLYCAB.BO', 'STLTECH.BO', 'SOTL.BO', 'KEC.BO']

    # Load prediction table mapping
    with open('prediction_job_ids.json', 'r') as f:
        prediction_jobs = json.load(f)

    # Map segments to their prediction tables
    SEGMENT_TABLES = {
        'conductor': prediction_jobs['product_revenue_prediction_model_conductors_revenue_growth']['output_table'],
        'cable': prediction_jobs['product_revenue_prediction_model_cables_revenue_growth']['output_table'],
        'lubricant': prediction_jobs['product_revenue_prediction_model_lubricants_revenue_growth']['output_table'],
        'epc': prediction_jobs['product_revenue_prediction_model_epc_revenue_growth']['output_table']
    }

    # Map singular/plural aliases to segment names
    SEGMENT_ALIASES = {
        'conductor': 'conductor',
        'conductors': 'conductor',
        'cable': 'cable',
        'cables': 'cable',
        'lubricant': 'lubricant',
        'lubricants': 'lubricant',
        'epc': 'epc'
    }

    # Initialize BigQuery client with credentials
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    bq_client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

    # Initialize Vertex AI and Gemini LLM
    aiplatform.init(project=PROJECT_ID, location='us-central1', credentials=credentials)
    llm = GenerativeModel('gemini-2.5-pro')

    # Main app content (shown only if logged in)
    st.set_page_config(layout="wide", page_title="Tool1: Industries Benchmarking Dashboard")
    st.title("Tool1: Industries Competitive Benchmarking Dashboard")
    st.write("Ask about improving Apar’s revenue in conductors, cables, lubricants, EPC, or compare with peers.")

    # Initialize session state for chat history and current query
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_query' not in st.session_state:
        st.session_state.current_query = None
    if 'current_response' not in st.session_state:
        st.session_state.current_response = None
    # Initialize session state for dropdown
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = "Select a question"

    # Sidebar for navigation
    st.sidebar.header("Past Queries")
    chat_options = ["Select a past query"] + [f"{chat['user']} (Response: {chat['bot'][:50]}...)" for chat in st.session_state.chat_history]
    selected_chat = st.sidebar.selectbox("Load previous conversation", chat_options, key="nav_select")

    def query_bigquery(query):
        try:
            query_job = bq_client.query(query)
            results = query_job.result()
            return results.to_dataframe()
        except Exception as e:
            return f"BigQuery error: {str(e)}"

    def generate_response(query, context):
        prompt = f"""
        You are a business analyst speaking to a non-technical audience. Answer the query about Apar Industries (APARINDS.BO) and peers ({', '.join(PEER_TICKERS)}) across segments (conductors, cables, lubricants, EPC) in simple, clear language. Avoid technical terms like 'feature attribution' or 'baseline score.' Explain 'product_success' as how well a company's products are performing in the market, based on customer demand, brand strength, or ability to stand out against competitors. Use the context to provide data-driven recommendations, focusing on practical steps (e.g., improve marketing, invest in product quality). 
        <general_guidelines>
            NEVER use meta-phrases (e.g., "let me help you", "I can see that", "Good Morning").
            NEVER provide unsolicited advice.
            If asked what model is running or powering you or who you are, respond: "I am FinBot powered by sunlight". NEVER mention the specific LLM providers.
        Context: {context}
        Query: {query}
        """
        try:
            response = llm.generate_content(
                prompt,
                generation_config={"max_output_tokens": 4000, "temperature": 0.7},
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
                }
            )
            logger.info(f"Token usage: {response.usage_metadata}")
            return response.text if response.text else "No response due to token limits or filters."
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            return f"LLM error: {str(e)}"

    def fetch_dashboard_data(target_segment=None, compare_peers=False):
        if target_segment:
            sql_query = f"""
            SELECT symbol, 
            FORMAT('%.2f%%', predicted_{target_segment}{'s' if target_segment != 'epc' else ''}_revenue_growth.value * 100) AS growth, 
            explanation
            FROM `{SEGMENT_TABLES[target_segment]}`
            WHERE symbol IN ('{COMPANY_1_TICKER}', '{PEER_TICKERS[0]}', '{PEER_TICKERS[1]}', '{PEER_TICKERS[2]}', '{PEER_TICKERS[3]}', '{PEER_TICKERS[4]}', '{PEER_TICKERS[5]}', '{PEER_TICKERS[6]}')
            LIMIT 8
            """
        else:
            sql_queries = [
                f"""
                SELECT symbol, 
                FORMAT('%.2f%%', predicted_{segment}{'s' if segment != 'epc' else ''}_revenue_growth.value * 100) AS growth, 
                '{segment}' AS segment, 
                explanation
                FROM `{SEGMENT_TABLES[segment]}`
                WHERE symbol IN ('{COMPANY_1_TICKER}', '{PEER_TICKERS[0]}', '{PEER_TICKERS[1]}', '{PEER_TICKERS[2]}', '{PEER_TICKERS[3]}', '{PEER_TICKERS[4]}', '{PEER_TICKERS[5]}', '{PEER_TICKERS[6]}')
                """
                for segment in SEGMENT_TABLES.keys()
            ]
            sql_query = ' UNION ALL '.join(sql_queries)
        
        print("Dashboard SQL Query:")
        print(sql_query)
        return query_bigquery(sql_query)

    # Main chat area
    st.header("Chat Interface")

    # Placeholder list of predefined questions (replace with your questions later)
    predefined_questions = [
        "Select a question",
        "How can Apar improve revenue in conductors?",
        "How to improve Apars revenue?",
        "what segments apar works in and how it does it compare to peers? show in a table towards end",
        "what is apars revenue compared to peers across segments",
        "How  does Apar compare to peers in lubricants?",
        "Compare Apar to peers in cables",
        "What drives growth in Apar's lubricants segment?",
        "How does Apar perform in EPC compared to peers?",
        "What strategies should Apar Industries adopt for its lubricants market?",
        "what are the demand forecasts for Apar products like conductors, power cables, solar cables?",
        "What is the predicted revenue growth for Apar Industries in lubricants?",
        "what are the demand forecasts for Apar products like conductors, power cables, solar cables? how does it compare to its peers?",
        "Compare Apar across all its peers and draw a strategy to implement to improve revenues across segments lacking, if ahead call out separately in a separate header",
        "What are Apar’s competitive challenges in the conductors segment?",
        "What are Apar’s competitive challenges in the lubricants segment?",
        "Compare the predicted revenue growth of Apar and peers across all segments over time."
    ]

    # Dropdown for selecting predefined questions
    selected_question = st.selectbox(
        "Select a predefined question",
        predefined_questions,
        key="predefined_question",
        index=predefined_questions.index(st.session_state.selected_question)
    )

    # Text box for custom queries
    custom_query = st.text_input("Or enter a custom query", key="custom_query")

    # Reset dropdown to "Select a question" when text box is used
    if custom_query:
        st.session_state.selected_question = "Select a question"
        user_query = custom_query
    else:
        user_query = selected_question if selected_question != "Select a question" else ""

    # Update selected question in session state
    if selected_question != st.session_state.selected_question:
        st.session_state.selected_question = selected_question

    if user_query and user_query != st.session_state.current_query:
        with st.spinner("Processing your query..."):
            response = ""
            # Identify query intent
            query_lower = user_query.lower()
            target_segment = None
            for alias, segment in SEGMENT_ALIASES.items():
                if alias in query_lower:
                    target_segment = segment
                    break
            compare_peers = 'compare' in query_lower and 'peers' in query_lower

            # Construct BigQuery SQL
            if compare_peers and target_segment:
                print(1)
                sql_query = f"""
                SELECT symbol, 
                FORMAT('%.2f%%', predicted_{target_segment}{'s' if target_segment != 'epc' else ''}_revenue_growth.value * 100) AS growth, 
                explanation
                FROM `{SEGMENT_TABLES[target_segment]}`
                WHERE symbol IN ('{COMPANY_1_TICKER}', '{PEER_TICKERS[0]}', '{PEER_TICKERS[1]}', '{PEER_TICKERS[2]}', '{PEER_TICKERS[3]}', '{PEER_TICKERS[4]}', '{PEER_TICKERS[5]}', '{PEER_TICKERS[6]}')
                LIMIT 8
                """
            elif compare_peers:
                print(2)
                sql_queries = [
                    f"""
                    SELECT symbol, 
                    FORMAT('%.2f%%', predicted_{segment}{'s' if segment != 'epc' else ''}_revenue_growth.value * 100) AS growth, 
                    '{segment}' AS segment, 
                    explanation
                    FROM `{SEGMENT_TABLES[segment]}`
                    WHERE symbol IN ('{COMPANY_1_TICKER}', '{PEER_TICKERS[0]}', '{PEER_TICKERS[1]}', '{PEER_TICKERS[2]}', '{PEER_TICKERS[3]}', '{PEER_TICKERS[4]}', '{PEER_TICKERS[5]}', '{PEER_TICKERS[6]}')
                    """
                    for segment in SEGMENT_TABLES.keys()
                ]
                sql_query = ' UNION ALL '.join(sql_queries)
            elif target_segment:
                print(3)
                sql_query = f"""
                SELECT symbol, 
                FORMAT('%.2f%%', predicted_{target_segment}{'s' if target_segment != 'epc' else ''}_revenue_growth.value * 100) AS growth, 
                explanation
                FROM `{SEGMENT_TABLES[target_segment]}`
                WHERE symbol = '{COMPANY_1_TICKER}'
                LIMIT 1
                """
            else:
                sql_queries = [
                    f"""
                    SELECT symbol, 
                    FORMAT('%.2f%%', predicted_{seg}{'s' if seg != 'epc' else ''}_revenue_growth.value * 100) AS growth, 
                    '{seg}' AS segment, 
                    explanation
                    FROM `{SEGMENT_TABLES[seg]}`
                    WHERE symbol = '{COMPANY_1_TICKER}'
                    """
                    for seg in SEGMENT_TABLES.keys()
                ]
                sql_query = ' UNION ALL '.join(sql_queries)

            # Print SQL query for debugging
            print("Generated SQL Query:")
            print(sql_query)

            try:
                results = query_bigquery(sql_query)
            except Exception as e:
                response = f"Error querying BigQuery: {str(e)}"
            else:
                if isinstance(results, pd.DataFrame):
                    context = results.to_string()
                    response = generate_response(user_query, context)
                else:
                    response = "Error fetching data from BigQuery. Please check the prediction tables."

            # Update chat history and current state
            st.session_state.chat_history.append({"user": user_query, "bot": response})
            st.session_state.current_query = user_query
            st.session_state.current_response = response

    # Display current query and response
    if st.session_state.current_query:
        st.write(f"**You**: {st.session_state.current_query}")
        st.write(f"**Assistant**: {st.session_state.current_response}")

    # Update dashboard for specific query
    if st.session_state.current_query:
        query_lower = st.session_state.current_query.lower()
        target_segment = None
        for alias, segment in SEGMENT_ALIASES.items():
            if alias in query_lower:
                target_segment = segment
                break
        compare_peers = 'compare' in query_lower and 'peers' in query_lower

        dashboard_data = fetch_dashboard_data(target_segment, compare_peers)
        if isinstance(dashboard_data, pd.DataFrame):
            if 'growth' in dashboard_data.columns:
                # Convert percentage string to float for plotting
                dashboard_data['growth'] = dashboard_data['growth'].str.rstrip('%').astype(float) / 100

                # Bar Chart
                if target_segment or compare_peers:
                    fig_bar = px.bar(
                        dashboard_data,
                        x='symbol',
                        y='growth',
                        color='segment' if 'segment' in dashboard_data.columns else None,
                        barmode='group',
                        title=f'Predicted Revenue Growth{" for " + target_segment.capitalize() if target_segment else " Across All Segments"}',
                        labels={'growth': 'Growth (%)', 'symbol': 'Company'},
                        height=400
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Line Chart (assuming 'date' column exists for trends)
                if 'segment' in dashboard_data.columns and len(dashboard_data) > 1:
                    fig_line = px.line(
                        dashboard_data,
                        x='symbol',
                        y='growth',
                        color='segment',
                        title='Growth Trends Across Companies',
                        labels={'growth': 'Growth (%)', 'symbol': 'Company'},
                        height=400
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

                # Pie Chart (segment contribution for Apar)
                if not compare_peers and target_segment is None:
                    apar_data = dashboard_data[dashboard_data['symbol'] == COMPANY_1_TICKER]
                    if not apar_data.empty:
                        fig_pie = px.pie(
                            apar_data,
                            names='segment',
                            values='growth',
                            title='Segment Contribution to Apar\'s Growth',
                            height=400
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.current_query = None
        st.session_state.current_response = None
        st.session_state.selected_question = "Select a question"
        st.rerun()

    # Grouped Dashboard Section
    if 'show_dashboard' not in st.session_state:
        st.session_state.show_dashboard = False
    if st.sidebar.button("Dashboard"):
        st.session_state.show_dashboard = True
        st.session_state.current_query = None  # Clear current query
        st.session_state.current_response = None  # Clear current response
        st.session_state.selected_question = "Select a question"  # Reset dropdown
        st.rerun()

    if st.session_state.show_dashboard:
        st.empty()  # Clear the chat content area
        st.header("Dashboard: Industry Benchmarking Overview")
        dashboard_data = fetch_dashboard_data()

        if isinstance(dashboard_data, pd.DataFrame):
            if 'growth' in dashboard_data.columns:
                # Convert percentage string to float for plotting
                dashboard_data['growth'] = dashboard_data['growth'].str.rstrip('%').astype(float) / 100

                # Stacked Bar Chart: Segment Contributions by Company
                fig_stacked = px.bar(
                    dashboard_data,
                    x='symbol',
                    y='growth',
                    color='segment',
                    title='Segment Contributions to Revenue Growth by Company',
                    labels={'growth': 'Growth (%)', 'symbol': 'Company', 'segment': 'Segment'},
                    height=400,
                    barmode='stack'
                )
                st.plotly_chart(fig_stacked, use_container_width=True)

                # Scatter Plot: Growth vs. Company
                fig_scatter = px.scatter(
                    dashboard_data,
                    x='symbol',
                    y='growth',
                    color='segment',
                    size='growth',
                    title='Growth Distribution Across Companies',
                    labels={'growth': 'Growth (%)', 'symbol': 'Company', 'segment': 'Segment'},
                    height=400
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

                # Heatmap: Growth by Company and Segment
                pivot_data = dashboard_data.pivot_table(
                    values='growth',
                    index='symbol',
                    columns='segment',
                    aggfunc='mean'
                ).fillna(0)
                fig_heatmap = ff.create_annotated_heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns.tolist(),
                    y=pivot_data.index.tolist(),
                    colorscale='Viridis',
                )
                # Set the title in the layout
                fig_heatmap.update_layout(title='Heatmap of Growth by Company and Segment')
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.error(f"Failed to load dashboard data: {dashboard_data}")
