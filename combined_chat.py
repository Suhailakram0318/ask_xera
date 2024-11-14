import streamlit as st
import mysql.connector
import google.generativeai as genai
import pickle
from typing import Any, Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import sys
import getpass
import os
from typing import Optional
import re

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
GEMINI_API_URL = os.getenv('GEMINI_API_URL')
URLS_TO_CRAWL = [url.strip() for url in os.getenv('URLS_TO_CRAWL', '').split(',') if url.strip()]

# Database Configuration
DB_CONFIG = {
    "host": os.getenv('DB_HOST'),
    "database": os.getenv('DB_NAME'),
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PASSWORD'),
    "port": int(os.getenv('DB_PORT', 3306))
}

# Leadership team data
LEADERSHIP_TEAM = {
    "leadership": [
        {"name": "Ganesh N Mandalam", "position": "CEO"},
        {"name": "Gene Bindi", "position": "President"},
        {"name": "Selvakumar", "position": "Revenue"},
        {"name": "Patric Charles", "position": "Sales"},
        {"name": "Harish Padmanabhan", "position": "Partnerships"},
        {"name": "Rajasekar", "position": "Account Management"},
        {"name": "Ram Prabhakaran", "position": "Solutions"},
        {"name": "Prem Kumar", "position": "Technology"},
        {"name": "Vishnu", "position": "Innovation"},
        {"name": "Senthil Kumar", "position": "Delivery"},
        {"name": "Anandh Rajan", "position": "Experience"},
        {"name": "Masilamani", "position": "Marketing"},
        {"name": "Buvaraha Murthy", "position": "Finance"},
        {"name": "Srinivasan", "position": "Campaign"}
    ]
}

# Create data directory for storing JSON files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'website_data')
MAIN_JSON_FILE = os.path.join(DATA_DIR, 'xerago_data.json')

# Create directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

def save_data(data):
    """Save data to the main JSON file"""
    try:
        with open(MAIN_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving data: {e}")

def load_data():
    """Load data from the main JSON file"""
    try:
        if os.path.exists(MAIN_JSON_FILE):
            with open(MAIN_JSON_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading data: {e}")
    return None

def initialize_session_state():
    """Initialize all session state variables"""
    if 'pdf_chat_history' not in st.session_state:
        st.session_state.pdf_chat_history = []
    if 'website_chat_history' not in st.session_state:
        st.session_state.website_chat_history = []
    if 'sources' not in st.session_state:
        st.session_state.sources = []
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = load_qa_chain()
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "Database Query"
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'response_count' not in st.session_state:
        st.session_state.response_count = 0
    if 'chat_ended' not in st.session_state:
        st.session_state.chat_ended = False
    if 'asking_continue' not in st.session_state:
        st.session_state.asking_continue = False

import streamlit as st
import mysql.connector
import google.generativeai as genai
import pickle
from typing import Any, Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
import datetime
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import sys

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
GEMINI_API_URL = os.getenv('GEMINI_API_URL')
URLS_TO_CRAWL = [url.strip() for url in os.getenv('URLS_TO_CRAWL', '').split(',') if url.strip()]

# Database Configuration
DB_CONFIG = {
    "host": os.getenv('DB_HOST'),
    "database": os.getenv('DB_NAME'),
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PASSWORD'),
    "port": int(os.getenv('DB_PORT', 3306))
}

# Leadership team data
LEADERSHIP_TEAM = {
    "leadership": [
        {"name": "Ganesh N Mandalam", "position": "CEO"},
        {"name": "Gene Bindi", "position": "President"},
        {"name": "Selvakumar", "position": "Revenue"},
        {"name": "Patric Charles", "position": "Sales"},
        {"name": "Harish Padmanabhan", "position": "Partnerships"},
        {"name": "Rajasekar", "position": "Account Management"},
        {"name": "Ram Prabhakaran", "position": "Solutions"},
        {"name": "Prem Kumar", "position": "Technology"},
        {"name": "Vishnu", "position": "Innovation"},
        {"name": "Senthil Kumar", "position": "Delivery"},
        {"name": "Anandh Rajan", "position": "Experience"},
        {"name": "Masilamani", "position": "Marketing"},
        {"name": "Buvaraha Murthy", "position": "Finance"},
        {"name": "Srinivasan", "position": "Campaign"}
    ]
}

# Create data directory for storing JSON files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'website_data')
MAIN_JSON_FILE = os.path.join(DATA_DIR, 'xerago_data.json')

# Create directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

def save_data(data):
    """Save data to the main JSON file"""
    try:
        with open(MAIN_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving data: {e}")

def load_data():
    """Load data from the main JSON file"""
    try:
        if os.path.exists(MAIN_JSON_FILE):
            with open(MAIN_JSON_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading data: {e}")
    return None

import streamlit as st
import mysql.connector
import google.generativeai as genai
import pickle
from typing import Any, Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
import datetime
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import sys

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
GEMINI_API_URL = os.getenv('GEMINI_API_URL')
URLS_TO_CRAWL = [url.strip() for url in os.getenv('URLS_TO_CRAWL', '').split(',') if url.strip()]

# Database Configuration
DB_CONFIG = {
    "host": os.getenv('DB_HOST'),
    "database": os.getenv('DB_NAME'),
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PASSWORD'),
    "port": int(os.getenv('DB_PORT', 3306))
}

# Leadership team data
LEADERSHIP_TEAM = {
    "leadership": [
        {"name": "Ganesh N Mandalam", "position": "CEO"},
        {"name": "Gene Bindi", "position": "President"},
        {"name": "Selvakumar", "position": "Revenue"},
        {"name": "Patric Charles", "position": "Sales"},
        {"name": "Harish Padmanabhan", "position": "Partnerships"},
        {"name": "Rajasekar", "position": "Account Management"},
        {"name": "Ram Prabhakaran", "position": "Solutions"},
        {"name": "Prem Kumar", "position": "Technology"},
        {"name": "Vishnu", "position": "Innovation"},
        {"name": "Senthil Kumar", "position": "Delivery"},
        {"name": "Anandh Rajan", "position": "Experience"},
        {"name": "Masilamani", "position": "Marketing"},
        {"name": "Buvaraha Murthy", "position": "Finance"},
        {"name": "Srinivasan", "position": "Campaign"}
    ]
}

# Create data directory for storing JSON files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'website_data')
MAIN_JSON_FILE = os.path.join(DATA_DIR, 'xerago_data.json')

# Create directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

def save_data(data):
    """Save data to the main JSON file"""
    try:
        with open(MAIN_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving data: {e}")

def load_data():
    """Load data from the main JSON file"""
    try:
        if os.path.exists(MAIN_JSON_FILE):
            with open(MAIN_JSON_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading data: {e}")
    return None

def fetch_info_from_website(url):
    """Fetch and process information from a website"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        scraped_data = {
            "timestamp": datetime.now().isoformat(),
            "url": url,
            "metadata": {
                "title": soup.title.string if soup.title else '',
                "headings": {
                    "h1": [h.get_text(strip=True) for h in soup.find_all('h1')],
                    "h2": [h.get_text(strip=True) for h in soup.find_all('h2')],
                    "h3": [h.get_text(strip=True) for h in soup.find_all('h3')]
                }
            },
            "content": {
                "body_text": soup.find('body').get_text(separator=' ', strip=True),
                "paragraphs": [p.get_text(strip=True) for p in soup.find_all('p')],
                "links": [{"text": a.get_text(strip=True), "href": a.get('href')} 
                         for a in soup.find_all('a', href=True)]
            }
        }
        return scraped_data

    except requests.exceptions.RequestException as e:
        st.error(f"Error scraping website {url}: {str(e)}")
        return None
    
def get_website_response(question: str) -> str:
    """Get response from Gemini API based on website content"""
    try:
        # Load or create data
        all_data = load_data()
        if not all_data:
            all_data = {
                "websites": [],
                "leadership": LEADERSHIP_TEAM["leadership"],
                "interactions": []
            }
            
            # Fetch data from all URLs
            for url in URLS_TO_CRAWL:
                scraped_data = fetch_info_from_website(url)
                if scraped_data:
                    all_data["websites"].append(scraped_data)
            
            if not all_data["websites"]:
                return "Error: Unable to fetch website data."
                
            save_data(all_data)

        # Combine context from all websites and leadership information
        website_context = " ".join([
            f"{site['metadata']['title']} {' '.join(site['metadata']['headings']['h1'])} {site['content']['body_text']}"
            for site in all_data["websites"]
        ])
        
        leadership_context = "Leadership Team: "
        for leader in all_data["leadership"]:
            leadership_context += f"{leader['name']} is the {leader['position']}, "
        
        context = f"{website_context} {leadership_context}"

        # Prepare prompt for Gemini
        prompt = f"""
        Based on the following context from the Xerago website, about-us page, and leadership information, please answer this question:
        
        Question: {question}
        
        Website Context: {context}
        
        If the question is about leadership or team members, please provide information about their roles and positions.
        If you cannot find relevant information in the context to answer the question, 
        please respond with: "I apologize, but I don't have enough information from the website to answer that question."
        
        Answer:
        """

        headers = {
            'Content-Type': 'application/json',
        }

        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }

        response = requests.post(
            f"{GEMINI_API_URL}?key={os.getenv('GEMINI_API_KEY')}",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            response_data = response.json()
            answer = response_data['candidates'][0]['content']['parts'][0]['text']
            
            # Add to interactions and save
            all_data['interactions'].append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "response": answer
            })
            save_data(all_data)
            
            return answer
        else:
            return f"Error: Unable to get response (Status code: {response.status_code})"

    except Exception as e:
        return f"Error generating response: {str(e)}"

def load_qa_chain(model_path='qa_chain.pkl'):
    """Load the saved QA chain"""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading PDF model: {str(e)}")
        return None

@st.cache_resource
def init_connection():
    """Initialize database connection"""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

def get_user_details():
    """Get current user's details from database"""
    try:
        conn = init_connection()
        if not conn:
            return None
        
        cursor = conn.cursor(dictionary=True)
        username = getpass.getuser()
        
        query = "SELECT * FROM dim_employee WHERE login_name = %s"
        cursor.execute(query, (username,))
        user_details = cursor.fetchone()
        cursor.close()
        return user_details
    except Exception as e:
        st.error(f"Error getting user details: {e}")
        return None

def get_table_columns(table_name: str) -> List[str]:
    """Get column names for a table"""
    try:
        conn = init_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT TABLE_TYPE 
            FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = %s
        """, (os.getenv('DB_NAME'), table_name))
        result = cursor.fetchone()
        if not result or result[0] != 'BASE TABLE':
            return []
            
        cursor.execute(f"DESCRIBE {table_name}")
        columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return columns
    except Exception as e:
        return []

def find_related_tables(user_details: Dict) -> Dict[str, List[str]]:
    """Find tables that share columns with dim_employee"""
    try:
        conn = init_connection()
        if not conn:
            return {}
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT TABLE_NAME 
            FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = %s
            AND TABLE_TYPE = 'BASE TABLE'
        """, (os.getenv('DB_NAME'),))
        tables = [table[0] for table in cursor.fetchall()]
        
        related_tables = {}
        emp_columns = set(user_details.keys())
        
        for table in tables:
            if table != 'dim_employee':
                table_columns = set(get_table_columns(table))
                common_columns = emp_columns.intersection(table_columns)
                if common_columns:
                    related_tables[table] = list(common_columns)
        
        cursor.close()
        return related_tables
    except Exception as e:
        st.error(f"Error finding related tables: {e}")
        return {}

def clean_sql_query(query: str) -> str:
    """Clean SQL query"""
    if not query:
        return query
    query = re.sub(r'```sql|```', '', query)
    query = query.replace('`', '')
    query = ' '.join(query.split())
    return query.strip()

def run_query(query: str, params: Optional[Dict] = None) -> Tuple[Optional[List[str]], Optional[List[Tuple]]]:
    """Execute SQL query"""
    conn = init_connection()
    if not conn:
        return None, None
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(clean_sql_query(query), params)
        else:
            cursor.execute(clean_sql_query(query))
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()
        return columns, rows
    except Exception as e:
        st.error(f"Query Error: {e}")
        st.error(f"Query being executed: {query}")
        return None, None

def format_chat_history(history: List) -> List[str]:
    """Format chat history for display"""
    formatted = []
    for msg in history:
        if isinstance(msg, dict):
            formatted.append(f"{msg['role'].capitalize()}: {msg['content']}")
        else:
            user_msg, bot_msg = msg
            formatted.append(f"User: {user_msg}")
            formatted.append(f"Assistant: {bot_msg}")
    return formatted

def conversation(message: str, history: List[Dict[str, str]], qa_chain: Any, mode: str) -> tuple:
    """Handle conversation for both modes"""
    try:
        if mode == "PDF":
            if not qa_chain:
                st.error("PDF model not loaded!")
                return None, None, None
            
            formatted_history = format_chat_history(history)
            response = qa_chain.invoke({
                "question": message,
                "chat_history": formatted_history
            })
            
            answer = response["answer"]
            sources = response["source_documents"]
            
            source_texts = []
            source_pages = []
            for i in range(min(3, len(sources))):
                source_texts.append(sources[i].page_content.strip())
                source_pages.append(sources[i].metadata["page"] + 1)
            
            while len(source_texts) < 3:
                source_texts.append("")
                source_pages.append(0)
                
            return answer, source_texts, source_pages
        else:
            answer = get_website_response(message)
            return answer, [""] * 3, [None] * 3
            
    except Exception as e:
        st.error(f"Error in conversation: {str(e)}")
        return None, None, None

def get_predefined_queries(user_details: Dict, related_tables: Dict[str, List[str]]) -> Dict[str, str]:
    """Get predefined queries for all tables"""
    queries = {
        "My Basic Details": """
            SELECT employee_name, employee_id, login_name, emp_email, grade
            FROM dim_employee
            WHERE login_name = %(login_name)s
        """
    }
    
    # Add all related table queries
    for table, columns in related_tables.items():
        query_name = f"My Records from {table}"
        query = f"""
            SELECT t.*
            FROM {table} t
            WHERE t.{columns[0]} = %(employee_id)s
        """
        queries[query_name] = query
    
    return queries

def generate_summary(columns: List[str], rows: List[tuple]) -> str:
    """Generate a summary description based on the table data."""
    if not columns or not rows:
        return "No data available to summarize."
    
    try:
        # Create data preview
        data_preview = []
        for row in rows[:5]:  # Use first 5 rows for the summary
            row_dict = dict(zip(columns, row))
            data_preview.append(row_dict)
        
        total_records = len(rows)
        numeric_stats = {}
        
        # Calculate statistics for numeric columns
        for i, col in enumerate(columns):
            try:
                values = [float(row[i]) for row in rows if row[i] is not None]
                if values:
                    numeric_stats[col] = {
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
            except (ValueError, TypeError):
                continue
        
        # Create prompt for Gemini
        prompt = f"""
        Analyze this table data and provide a concise summary including:
        - Total records: {total_records}
        - Key columns present: {', '.join(columns)}
        - Numeric column statistics: {numeric_stats if numeric_stats else 'None identified'}
        
        Table preview (first {min(5, len(rows))} rows):
        {data_preview}
        
        Please provide a 2-3 sentence summary focusing on:
        1. The overall scope and content of the data
        2. Any notable patterns or key insights
        3. Important statistics or trends (if applicable)
        
        Keep the summary concise and business-focused.
        """
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text
        return "Unable to generate summary for this data."
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Summary generation failed."

def handle_database_interface(user_details: Dict):
    """Handle database interface"""
    related_tables = find_related_tables(user_details)
    
    st.write("### Available Related Tables")
    for table, columns in related_tables.items():
        st.write(f"- {table} (linked via: {', '.join(columns)})")
    
    query_type = st.radio(
        "Select query type:",
        ["Predefined Queries", "Custom Join Query"]
    )

    if query_type == "Predefined Queries":
        predefined_queries = get_predefined_queries(user_details, related_tables)
        selected_query = st.selectbox(
            "Select a query:",
            list(predefined_queries.keys())
        )
        
        if selected_query:
            st.code(predefined_queries[selected_query], language="sql")
            
            if st.button("Execute Query"):
                with st.spinner("Executing query..."):
                    columns, rows = run_query(
                        predefined_queries[selected_query], 
                        user_details
                    )
                    if columns and rows:
                        # Display summary in expandable section
                        with st.expander("📊 View Data Summary", expanded=True):
                            st.write("### Data Summary")
                            with st.spinner("Generating summary..."):
                                summary = generate_summary(columns, rows)
                            st.write(summary)
                        
                        # Display detailed results
                        st.write("### Detailed Results")
                        results = [dict(zip(columns, row)) for row in rows]
                        st.table(results)
                    else:
                        st.warning("No results found for this query.")
    
    else:
        st.write("### Build Custom Join Query")
        selected_tables = st.multiselect(
            "Select tables to join",
            list(related_tables.keys())
        )
        
        if selected_tables:
            base_query = "SELECT * FROM dim_employee e"
            joins = []
            for table in selected_tables:
                common_cols = related_tables[table]
                joins.append(f"""
                    LEFT JOIN {table} t{len(joins)} 
                    ON e.{common_cols[0]} = t{len(joins)}.{common_cols[0]}
                """)
            
            where_clause = "WHERE e.login_name = %(login_name)s"
            query = base_query + "\n".join(joins) + where_clause
            
            if st.button("Execute Join Query"):
                with st.spinner("Executing query..."):
                    columns, rows = run_query(query, user_details)
                    if columns and rows:
                        # Display summary in expandable section
                        with st.expander("📊 View Data Summary", expanded=True):
                            st.write("### Data Summary")
                            with st.spinner("Generating summary..."):
                                summary = generate_summary(columns, rows)
                            st.write(summary)
                        
                        # Display detailed results
                        st.write("### Detailed Results")
                        results = [dict(zip(columns, row)) for row in rows]
                        st.table(results)
                    else:
                        st.warning("No results found for this query.")

def handle_chat_interface(mode: str):
    """Handle chat interface"""
    st.markdown("### Chat")
    
    # Get current mode's chat history
    current_chat_history = (st.session_state.pdf_chat_history if mode == "PDF Chat" 
                          else st.session_state.website_chat_history)
    
    # Display chat history
    for message in current_chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question here..."):
        current_chat_history.append({"role": "user", "content": prompt})
        
        if mode == "PDF Chat":
            answer, sources, pages = conversation(
                prompt, 
                current_chat_history,
                st.session_state.qa_chain,
                "PDF"
            )
            if answer:
                current_chat_history.append({"role": "assistant", "content": answer})
                st.session_state.sources = list(zip(sources, pages))
        else:
            answer = get_website_response(prompt)
            if answer:
                current_chat_history.append({"role": "assistant", "content": answer})
        
        # Update appropriate history
        if mode == "PDF Chat":
            st.session_state.pdf_chat_history = current_chat_history
        else:
            st.session_state.website_chat_history = current_chat_history
        
        st.rerun()

    # Clear chat button
    if st.button(f"Clear {mode} Chat"):
        if mode == "PDF Chat":
            st.session_state.pdf_chat_history = []
            st.session_state.sources = []
        else:
            st.session_state.website_chat_history = []
        st.rerun()


def detect_query_type(prompt: str) -> str:
    """Updated query type detection"""
    prompt_lower = prompt.lower()
    
    # Company info keywords
    company_keywords = ['xerago', 'digital impact', 'services', 'client', 'benefits', 'approach']
    
    if any(keyword in prompt_lower for keyword in company_keywords):
        return "Website"
    elif 'records from' in prompt_lower or any(keyword in prompt_lower for keyword in ['show', 'get', 'fetch', 'find', 'list', 'my', 'details']):
        return "Database Query"
    elif any(keyword in prompt_lower for keyword in ['policy', 'policies', 'EEO', 'security']):
        return "PDF"
    else:
        return "Website"

def format_policy_query(prompt: str) -> str:
    """Format policy-related queries to improve PDF search"""
    prompt_lower = prompt.lower()
    
    # Map common policy queries to more specific search terms
    policy_mappings = {
        'information security policy': [
            'information security policy',
            'information security guidelines',
            'security policy',
            'data security',
            'IT security'
        ],
        'equal employemt opportunity policy': [
            'policy statement',
            'responsibilities leadership',
            'compliance and reporting',
            'review and revision'
        ],
        'leave policy': [
            'leave policy',
            'leave guidelines',
            'leave rules',
            'vacation policy'
        ]
    }
    
    for policy_type, search_terms in policy_mappings.items():
        if any(term in prompt_lower for term in search_terms):
            return f"Please provide detailed information about the {policy_type}, including its key points and guidelines"
    
    return prompt

def format_response_in_points(response: str, min_points: int = 2, max_points: int = 6) -> str:
    """Format any response into bullet points"""
    try:
        # Remove any existing bullet points or numbers
        cleaned_text = re.sub(r'^\s*[-•*\d.]\s*', '', response, flags=re.MULTILINE)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we have too few sentences, try to split on semicolons and commas
        if len(sentences) < min_points:
            sentences = []
            for part in cleaned_text.split(';'):
                for subpart in part.split(','):
                    if len(subpart.strip()) > 20:  # Minimum length for a meaningful point
                        sentences.append(subpart.strip())
        
        # Ensure we have at least minimum points
        while len(sentences) < min_points and len(sentences) > 0:
            # Split longest sentence if possible
            longest = max(sentences, key=len)
            if len(longest) > 100:  # Only split if sentence is long enough
                split_idx = longest.find('. ', 50, -50)
                if split_idx != -1:
                    sentences.remove(longest)
                    sentences.extend([longest[:split_idx+1], longest[split_idx+2:]])
            else:
                break
        
        # Limit to maximum points
        sentences = sentences[:max_points]
        
        # Format as bullet points
        if len(sentences) >= min_points:
            formatted_points = []
            for idx, sentence in enumerate(sentences, 1):
                # Clean and capitalize the sentence
                clean_sentence = sentence.strip()
                if clean_sentence:
                    if not clean_sentence[0].isupper():
                        clean_sentence = clean_sentence[0].upper() + clean_sentence[1:]
                    if not clean_sentence.endswith(('.', '!', '?')):
                        clean_sentence += '.'
                    formatted_points.append(f"• {clean_sentence}")
            
            return "\n\n".join(formatted_points)
        
        return response
    except Exception as e:
        st.error(f"Error formatting response: {str(e)}")
        return response


def try_get_response(prompt: str, mode: str, history: List, qa_chain: Any) -> Tuple[Optional[str], List, List]:
    """Try to get response from specified mode with improved policy handling and structured points"""
    try:
        if mode == "PDF":
            if not qa_chain:
                st.error("PDF model not loaded!")
                return None, [], []
            
            formatted_prompt = format_policy_query(prompt)
            formatted_history = format_chat_history(history)
            
            response = qa_chain.invoke({
                "question": formatted_prompt,
                "chat_history": formatted_history
            })
            
            answer = response["answer"]
            sources = response["source_documents"]
            
            if answer:
                # Format the answer in points
                formatted_answer = format_response_in_points(answer)
                final_answer = f"Here's what I found about {prompt}:\n\n{formatted_answer}"
                
                source_texts = []
                source_pages = []
                for i in range(min(3, len(sources))):
                    source_texts.append(sources[i].page_content.strip())
                    source_pages.append(sources[i].metadata["page"] + 1)
                
                while len(source_texts) < 3:
                    source_texts.append("")
                    source_pages.append(0)
                
                return final_answer, source_texts, source_pages
            
            return None, [], []
            
        else:  # Website mode
            answer = get_website_response(prompt)
            if answer and answer.strip() and "no relevant information" not in answer.lower():
                formatted_answer = format_response_in_points(answer)
                return formatted_answer, [], []
            return None, [], []
            
    except Exception as e:
        st.error(f"Error in {mode} response: {str(e)}")
        return None, [], []

def display_follow_up_questions(topic: str, is_policy: bool = False):
    """Display follow-up questions after a response with better formatting"""
    questions = get_relevant_questions(topic)
    if questions:
        # Create columns for better layout
        cols = st.columns(3)
        for idx, q in enumerate(questions):
            with cols[idx]:
                if st.button(f"📌 {q['text']}", key=f"{topic}_{q['text'][:20]}"):
                    if is_policy:
                        handle_question(q["prompt"])
                    else:
                        handle_company_info_question(q["prompt"])
                    st.rerun()

def get_relevant_questions(topic: str) -> List[Dict[str, str]]:
    """Get relevant follow-up questions based on topic"""
    questions = {
        "leave policy": [
            {"text": "Types of Leaves Available", "prompt": "Explain the types of leaves available in the leave policy"},
            {"text": "Leave Application Process", "prompt": "What is the process to apply for leave according to the policy"},
            {"text": "Leave Encashment Rules", "prompt": "Explain the leave encashment rules in the policy"}
        ],
        "security policy": [
            {"text": "Password Requirements", "prompt": "Explain the password requirements in the security policy"},
            {"text": "Security Incident Reporting", "prompt": "What is the process to report security incidents"},
            {"text": "Data Protection Guidelines", "prompt": "Explain the data protection guidelines in the security policy"}
        ],
        "digital impact": [
            {"text": "Digital Transformation Services", "prompt": "What digital transformation services does Xerago offer"},
            {"text": "Impact Measurement Approach", "prompt": "How does Xerago measure digital impact for clients"},
            {"text": "Success Stories", "prompt": "What are Xerago's digital impact success stories"}
        ],
        "core services": [
            {"text": "Technology Services", "prompt": "What technology services does Xerago provide"},
            {"text": "Industry Expertise", "prompt": "Which industries does Xerago specialize in"},
            {"text": "Service Delivery Model", "prompt": "Explain Xerago's service delivery model"}
        ]
    }
    return questions.get(topic.lower(), [])

def handle_company_info_question(prompt: str):
    """Handle company information questions with follow-ups"""
    if 'unified_chat_history' not in st.session_state:
        st.session_state.unified_chat_history = []
    
    # Add user question to history
    st.session_state.unified_chat_history.append({
        "role": "user",
        "content": prompt
    })
    
    # Determine the topic for follow-up questions
    topic = ""
    if "digital impact" in prompt.lower():
        topic = "digital impact"
    elif "core services" in prompt.lower():
        topic = "core services"
    
    # Get response from website
    answer = get_website_response(prompt)
    if answer:
        formatted_answer = format_response_in_points(answer)
        
        # Create response message with follow-up questions
        response_content = formatted_answer
        st.session_state.unified_chat_history.append({
            "role": "assistant",
            "content": response_content
        })
        
        # Display follow-up questions if applicable
        if topic:
            display_follow_up_questions(topic, is_policy=False)
    else:
        st.session_state.unified_chat_history.append({
            "role": "assistant",
            "content": "I couldn't find information about this on our website. Please try rephrasing your question or contact support for more details."
        })

def handle_question(prompt: str):
    """Handle questions with follow-ups"""
    if 'unified_chat_history' not in st.session_state:
        st.session_state.unified_chat_history = []
    
    st.session_state.unified_chat_history.append({
        "role": "user",
        "content": prompt
    })
    
    # Determine topic for follow-up questions
    topic = ""
    if "leave policy" in prompt.lower():
        topic = "leave policy"
    elif "security policy" in prompt.lower():
        topic = "security policy"

    query_type = detect_query_type(prompt)
    
    if query_type == "PDF":
        answer, sources, pages = try_get_response(
            prompt, 
            "PDF", 
            st.session_state.unified_chat_history,
            st.session_state.qa_chain
        )
        
        if answer:
            # Add main response
            response_data = {
                "role": "assistant",
                "content": answer,
                "data": {"sources": list(zip(sources, pages))} if sources else {}
            }
            st.session_state.unified_chat_history.append(response_data)
            
            # Display follow-up questions if applicable
            if topic:
                display_follow_up_questions(topic, is_policy=True)
        else:
            st.session_state.unified_chat_history.append({
                "role": "assistant",
                "content": "I couldn't find specific information about this policy. Please try rephrasing your question."
            })
    
    elif query_type == "Website":
        web_answer, _, _ = try_get_response(
            prompt,
            "Website",
            st.session_state.unified_chat_history,
            None
        )
        
        if web_answer:
            st.session_state.unified_chat_history.append({
                "role": "assistant",
                "content": web_answer
            })
            # Display follow-up questions if applicable
            if topic:
                display_follow_up_questions(topic, is_policy=False)
        else:
            st.session_state.unified_chat_history.append({
                "role": "assistant",
                "content": "I couldn't find information about this. Please try rephrasing your question."
            })
    
def display_default_questions():
    """Display questions with separate handling for company information"""
    st.markdown("### Quick Questions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("📄 **Policy Information**")
        if st.button("What is the Leave Policy?"):
            handle_question("Explain the complete leave policy and its guidelines")
            st.rerun()
        if st.button("What is the Security Policy?"):
            handle_question("Explain the complete security policy and its key points")
            st.rerun()

    with col2:
        st.markdown("📊 **My Information**")
        user_details = get_user_details()
        if user_details:
            predefined_queries = {
                "My Basic Details": """
                    SELECT employee_name, employee_id, login_name, emp_email, grade
                    FROM dim_employee
                    WHERE login_name = %(login_name)s
                """
            }
            
            # Add related table queries
            related_tables = find_related_tables(user_details)
            for table, columns in related_tables.items():
                query_name = f"My Records from {table}"
                query = f"""
                    SELECT t.*
                    FROM {table} t
                    WHERE t.{columns[0]} = %(dim_employee_id)s
                """
                predefined_queries[query_name] = query
            
            selected_query = st.selectbox(
                "Select a query:",
                list(predefined_queries.keys())
            )
            
            if selected_query and st.button("Execute Query"):
                with st.spinner("Executing query..."):
                    columns, rows = run_query(
                        predefined_queries[selected_query], 
                        user_details
                    )
                    if columns and rows:
                        summary = generate_summary(columns, rows)
                        results = [dict(zip(columns, row)) for row in rows]
                        st.session_state.unified_chat_history.append({
                            "role": "assistant",
                            "content": f"Here's what I found for {selected_query}:\n\n{summary}",
                            "data": {"table": results}
                        })
                        st.rerun()
                    else:
                        st.session_state.unified_chat_history.append({
                            "role": "assistant",
                            "content": f"No records found for {selected_query}. Please verify the data exists."
                        })
                        st.rerun()

    with col3:
        st.markdown("🌐 **Company Information**")
        if st.button("Digital Impact Approach"):
            handle_company_info_question("What is Xerago's approach to Digital Impact?")
            st.rerun()
        if st.button("Core Services"):
            handle_company_info_question("What are Xerago's core services?")
            st.rerun()

def handle_database_response(columns, rows, title: str):
    """Handle database query responses with summary"""
    if columns and rows:
        # Generate summary
        with st.spinner("Generating summary..."):
            summary = generate_summary(columns, rows)
        
        # Format results
        results = [dict(zip(columns, row)) for row in rows]
        
        # Add to chat history
        st.session_state.unified_chat_history.append({
            "role": "assistant",
            "content": f"Here's what I found for {title}:\n\n{summary}",
            "data": {"table": results}
        })
    else:
        st.session_state.unified_chat_history.append({
            "role": "assistant",
            "content": f"No records found for {title}."
        })

            
def display_greeting(user_details: Dict):
    """Display personalized greeting message with concise information guide"""
    current_hour = datetime.now().hour  # Changed from datetime.datetime.now()
    
    # Get time-based greeting
    if current_hour < 12:
        greeting = "Good Morning"
    elif current_hour < 16:
        greeting = "Good Afternoon"
    else:
        greeting = "Good Evening"
    
    # Rest of the greeting function remains the same
    greeting_message = f"{greeting}, {user_details['employee_name']}! 👋\n\n"
    greeting_message += "I'm Xera, your AI assistant. I can help you with:\n\n"
    
    # Employee Information with all options in comma-separated format
    greeting_message += "• **Accessing your employee information and records:** \n  Basic profile details, task assignments, leave balance & history, "
    greeting_message += "project assignments, time tracking records, performance ratings, department details, attendance logs, and work hour entries\n\n"
    
    # Policy Information
    greeting_message += "• **Providing information about company policies:**\n  Leave policies, security guidelines, and company procedures\n\n"
    
    # Company Information
    greeting_message += "• **Answering questions about Xerago's services and approach:**\n  Digital impact solutions, core services, and company expertise\n\n"
    
    greeting_message += "Please feel free to ask questions or use the quick access buttons below."
    
    # Display greeting in chat format
    with st.chat_message("assistant"):
        st.write(greeting_message)

def handle_continue_chat():
    """Handle the continuation question after 4 responses"""
    st.markdown("---")
    st.markdown("### Would you like help with anything else?")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Yes", key="continue_yes"):
            st.session_state.asking_continue = False
            st.session_state.response_count = 0
            st.rerun()
    
    with col2:
        if st.button("No", key="continue_no"):
            st.session_state.chat_ended = True
            st.session_state.unified_chat_history.append({
                "role": "assistant",
                "content": "Thank you for using Ask Xera! Have a great day! 👋"
            })
            st.rerun()

def handle_unified_chat():
    """Handle unified chat interface for all modes with session ending"""
    if 'unified_chat_history' not in st.session_state:
        st.session_state.unified_chat_history = []
        
    # Check if chat has ended
    if st.session_state.chat_ended:
        st.markdown("### Chat Ended")
        # Display the last message (thank you message)
        with st.chat_message("assistant"):
            st.write(st.session_state.unified_chat_history[-1]["content"])
        return

    st.markdown("### Ask Me Anything")
    
    # Display greeting if not shown yet
    if not st.session_state.get('greeting_displayed', False):
        user_details = get_user_details()
        if user_details:
            display_greeting(user_details)
            st.session_state.greeting_displayed = True
    
    # Display default questions
    display_default_questions()
    
    st.markdown("---")
    
    # Display chat history with follow-up questions
    for i, message in enumerate(st.session_state.unified_chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "data" in message:
                if "table" in message["data"]:
                    st.table(message["data"]["table"])
                if "sources" in message["data"]:
                    with st.expander("📚 Reference Sources"):
                        for idx, (source, page) in enumerate(message["data"]["sources"], 1):
                            if source and page:
                                st.text(f"Source {idx} - Page {page}:\n{source}")
            
            # Display follow-up questions after assistant's response
            if message["role"] == "assistant" and i > 0:
                prev_message = st.session_state.unified_chat_history[i-1]
                query = prev_message["content"].lower()
                
                # Determine topic from previous query
                topic = None
                if "leave policy" in query:
                    topic = "leave policy"
                elif "security policy" in query:
                    topic = "security policy"
                elif "digital impact" in query:
                    topic = "digital impact"
                elif "core services" in query:
                    topic = "core services"
                
                if topic:
                    st.markdown("---")
                    st.markdown("### 🔍 You might also be interested in:")
                    
                    questions = get_relevant_questions(topic)
                    cols = st.columns(3)
                    for idx, q in enumerate(questions):
                        with cols[idx]:
                            if st.button(f"📌 {q['text']}", 
                                       key=f"{topic}_{i}_{q['text'][:20]}"):
                                if topic in ["leave policy", "security policy"]:
                                    handle_question(q["prompt"])
                                else:
                                    handle_company_info_question(q["prompt"])
                                st.session_state.response_count += 1
                                st.rerun()
    
    # Check if we should ask about continuing
    if st.session_state.response_count >= 4 and not st.session_state.chat_ended:
        st.markdown("---")
        st.markdown("### Would you like help with anything else?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes", key="continue_yes"):
                st.session_state.asking_continue = False
                st.session_state.response_count = 0
                st.rerun()
        with col2:
            if st.button("No", key="continue_no"):
                st.session_state.chat_ended = True
                st.session_state.unified_chat_history.append({
                    "role": "assistant",
                    "content": "Thank you for using Ask Xera! Have a great day! 👋"
                })
                st.rerun()
        return
    
    # Chat input for custom questions (only show if not ended and not asking continue)
    if not st.session_state.chat_ended and not st.session_state.asking_continue:
        if prompt := st.chat_input("Ask your own question..."):
            handle_question(prompt)
            st.session_state.response_count += 1
            st.rerun()

        # Clear chat button
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("Clear Chat"):
                st.session_state.unified_chat_history = []
                st.session_state.sources = []
                st.session_state.greeting_displayed = False
                st.session_state.response_count = 0
                st.session_state.asking_continue = False
                st.session_state.chat_ended = False
                st.rerun()

def main():
    st.set_page_config(page_title="Ask Xera", layout="wide")
    
    # Initialize session state at the start
    initialize_session_state()
    
    st.title("Ask Xera")
    
    user_details = get_user_details()
    
    if user_details:
        st.sidebar.title("Your Information")
        st.sidebar.write(f"Name: {user_details['employee_name']}")
        st.sidebar.write(f"Employee ID: {user_details['employee_id']}")
        st.sidebar.write(f"Login: {user_details['login_name']}")
        
        # Main chat interface
        handle_unified_chat()
        
        # Debug information
        if st.sidebar.checkbox("Show Debug Info"):
            st.sidebar.write("User Details:", user_details)
            st.sidebar.markdown("### Chat Statistics")
            st.sidebar.write(f"Total Messages: {len(st.session_state.unified_chat_history)}")
    else:
        st.error("Unable to retrieve user details. Please check your connection and permissions.")

if __name__ == "__main__":
    main()