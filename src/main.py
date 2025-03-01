import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import pm4py
import google.generativeai as genai
from google.cloud import texttospeech
import tempfile
import logging
import uuid
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO

# Import our custom modules
from process_mining.discovery import ProcessDiscovery
from process_mining.performance import PerformanceAnalyzer
from process_mining.statistics import ProcessStatistics
from ai.gemini import GeminiInterface
from ai.insights import InsightGenerator
from visualization.process_maps import ProcessMapVisualizer
from visualization.charts import ChartGenerator
from utils.data_processing import EventLogProcessor
from utils.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)

def initialize_session_state():
    """Initialize session state variables"""
    if 'event_log' not in st.session_state:
        st.session_state.event_log = None
    if 'process_model' not in st.session_state:
        st.session_state.process_model = None
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'audio_file_path' not in st.session_state:
        st.session_state.audio_file_path = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_type' not in st.session_state:
        st.session_state.analysis_type = None
    if 'podcast_script' not in st.session_state:
        st.session_state.podcast_script = None
    if 'mining_df' not in st.session_state:
        st.session_state.mining_df = None

class AudioGenerator:
    def __init__(self):
        # Set up Google Cloud TTS client
        credentials_path = 'google_tts_key.json'
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        # Check for credentials file
        if not os.path.exists(credentials_path):
            self.tts_client = None
            logging.error(f"TTS credentials file not found: {credentials_path}")
            return
            
        try:
            # Initialize the TTS client with more detailed error handling
            self.tts_client = texttospeech.TextToSpeechClient()
            
            # Test the client with a simple request to verify credentials
            voices = self.tts_client.list_voices()
            logging.info(f"TTS client initialized successfully with {len(voices.voices)} available voices")
            
            # Define voice profiles
            self.voice_profiles = [
                {
                    'name': 'en-US-Studio-M',  # Male voice for Alex
                    'language_code': 'en-US',
                    'ssml_gender': texttospeech.SsmlVoiceGender.MALE
                },
                {
                    'name': 'en-US-Studio-O',  # Female voice for Sarah
                    'language_code': 'en-US',
                    'ssml_gender': texttospeech.SsmlVoiceGender.FEMALE
                }
            ]
        except Exception as e:
            self.tts_client = None
            error_message = f"Error initializing TTS client: {e}"
            logging.error(error_message, exc_info=True)
    def generate_audio_segment(self, text, voice_profile):
        """Generate audio for a single segment of text."""
        if self.tts_client is None:
            logging.error("TTS client is not initialized")
            return None
            
        try:
            # Make sure text is not too long for the API
            if len(text) > 5000:  # Google TTS has a limit of 5000 characters
                text = text[:4990] + "..."
                
            voice = texttospeech.VoiceSelectionParams(
                language_code=voice_profile['language_code'],
                name=voice_profile['name'],
                ssml_gender=voice_profile['ssml_gender']
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=0.95,
                pitch=0.0,
                volume_gain_db=1.0,
                effects_profile_id=['headphone-class-device']
            )

            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            logging.info(f"Generating audio for text: {text[:50]}...")
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            logging.info(f"Successfully generated audio of size: {len(response.audio_content)} bytes")
            return response.audio_content
        except Exception as e:
            logging.error(f"Error generating audio segment: {e}", exc_info=True)
            return None


    def generate_audio(self, script):
        """Generate full audio from podcast script."""
        if self.tts_client is None:
            logging.error("Cannot generate audio - TTS client not initialized")
            return None
            
        segments = script.split('\n')
        audio_segments = []
        segment_count = 0
        error_count = 0

        for segment in segments:
            if not segment.strip() or ':' not in segment:
                continue
                
            try:
                speaker, dialogue = segment.split(':', 1)
                dialogue = dialogue.strip()
                
                if not dialogue:
                    continue

                # Select voice profile based on speaker
                voice_profile = self.voice_profiles[0] if 'Alex' in speaker else self.voice_profiles[1]
                
                audio_content = self.generate_audio_segment(dialogue, voice_profile)
                if audio_content:
                    audio_segments.append(audio_content)
                    segment_count += 1
                    logging.debug(f"Generated audio segment {segment_count}: {dialogue[:30]}...")
                else:
                    error_count += 1
                    logging.warning(f"Failed to generate audio for segment: {dialogue[:50]}...")
            except Exception as e:
                error_count += 1
                logging.error(f"Error processing segment: {e}")

        if not audio_segments:
            logging.error(f"No audio segments were generated. Total errors: {error_count}")
            return None

        logging.info(f"Generated {segment_count} audio segments with {error_count} errors")
            
        try:
            # Combine all audio segments
            combined_audio = b''.join(audio_segments)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                temp_audio.write(combined_audio)
                logging.info(f"Saved audio to temporary file: {temp_audio.name}")
                return temp_audio.name
        except Exception as e:
            logging.error(f"Error saving audio file: {e}", exc_info=True)
            return None
def render_upload_page():
    """Render the file upload page"""
    st.header("Upload Event Log")
    
    # Initialize EventLogProcessor
    processor = EventLogProcessor()
    
    uploaded_file = st.file_uploader("Choose a CSV or XES file", type=["csv", "xes"])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "csv":
            try:
                # Read CSV and show column mapping
                df = pd.read_csv(uploaded_file)
                st.subheader("CSV Column Mapping")
                
                # Display raw data sample
                st.write("Raw Data Sample:")
                st.write(df.head())
                
                # Get column mappings from user
                col1, col2 = st.columns(2)
                with col1:
                    case_id_col = st.selectbox("Select Case ID column", df.columns)
                    activity_col = st.selectbox("Select Activity column", df.columns)
                    timestamp_col = st.selectbox("Select Timestamp column", df.columns)
                
                with col2:
                    st.write("Optional Columns:")
                    resource_col = st.selectbox("Select Resource column (optional)", ["None"] + list(df.columns))
                    cost_col = st.selectbox("Select Cost column (optional)", ["None"] + list(df.columns))
                
                if st.button("Process CSV"):
                    # Create column mapping
                    column_mapping = {
                        case_id_col: 'case:concept:name',
                        activity_col: 'concept:name',
                        timestamp_col: 'time:timestamp'
                    }
                    
                    # Add optional columns if selected
                    if resource_col != "None":
                        column_mapping[resource_col] = 'org:resource'
                    if cost_col != "None":
                        column_mapping[cost_col] = 'cost'
                    
                    # Rename columns to PM4Py format
                    df = df.rename(columns=column_mapping)
                    
                    try:
                        # Process the event log
                        event_log = processor.convert_csv_to_event_log(df)
                        st.session_state.event_log = event_log
                        
                        # Show success message and processed data
                        st.success("CSV file successfully processed!")
                        st.subheader("Processed Event Log Sample")
                        
                        # Convert event log back to DataFrame for display
                        processed_df = pm4py.convert_to_dataframe(event_log)
                        st.write(processed_df.head())
                        
                        # Show event log statistics
                        st.subheader("Event Log Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Cases", len(set(processed_df['case:concept:name'])))
                        with col2:
                            st.metric("Total Events", len(processed_df))
                        with col3:
                            st.metric("Unique Activities", len(set(processed_df['concept:name'])))
                            
                    except Exception as e:
                        st.error(f"Error converting to event log: {str(e)}")
                        st.error("Please ensure your data is in the correct format and try again.")
            except Exception as e:
                st.error(f"Error loading CSV file: {e}")
        
        elif file_extension == "xes":
            try:
                log = pm4py.read_xes(uploaded_file)
                st.session_state.event_log = log
                st.success("XES file successfully loaded!")
                
                # Show sample of loaded data
                st.subheader("Sample of Loaded Data")
                st.write(pd.DataFrame(pm4py.convert_to_dataframe(log)[0:5]))
            except Exception as e:
                st.error(f"Error loading XES file: {e}")
        else:
            st.error("Unsupported file format. Please upload a CSV or XES file.")

def perform_eda(data):
    """Perform enhanced EDA using the Gemini API."""
    # Initialize Gemini interface
    gemini = GeminiInterface()
    
    eda_prompt = f"""
    Perform detailed exploratory data analysis on this dataset and identify key patterns: 
    {data.head(10).to_json()}
    
    Dataset Info:
    - Shape: {data.shape}
    - Columns: {', '.join(data.columns)}
    - Data Types: {data.dtypes.to_dict()}
    - Summary Statistics: {data.describe().to_json()}
    - Missing Values: {data.isna().sum().to_dict()}
    
    Consider:
    1. Data distribution and patterns
    2. Missing value analysis
    3. Correlations between variables
    4. Outlier detection
    5. Trends and seasonality (if time series)
    6. Key business metrics and KPIs
    """
    response = gemini.generate_response(eda_prompt)
    return response

def determine_analysis_type(eda_result, data):
    """
    Determine the most appropriate type of analysis based on EDA results.
    Returns both the analysis type and the reasoning behind the choice.
    """
    # Initialize Gemini interface
    gemini = GeminiInterface()
    
    analysis_prompt = f"""
    Based on the EDA results below, determine which type of data analysis would be most valuable for this dataset.
    Choose from these types and provide detailed reasoning:

    1. Descriptive Analytics: What happened? (Focus on summarizing historical data patterns)
    2. Diagnostic Analytics: Why did it happen? (Focus on cause-and-effect relationships)
    3. Predictive Analytics: What might happen? (Focus on forecasting future trends)
    4. Prescriptive Analytics: What should we do? (Focus on recommended actions)

    EDA Results:
    {eda_result}

    Dataset Summary:
    - Shape: {data.shape}
    - Columns: {', '.join(data.columns)}
    - Data Sample: {data.head(3).to_json()}
    
    Return your response in this format:
    Analysis Type: [type]
    Reasoning: [detailed explanation]
    """
    response = gemini.generate_response(analysis_prompt)
    return response

def perform_specific_analysis(data, analysis_type, eda_result):
    """
    Perform the specific type of analysis determined to be most appropriate.
    """
    # Initialize Gemini interface
    gemini = GeminiInterface()
    
    analysis_prompts = {
        "Descriptive Analytics": f"""
        Perform comprehensive descriptive analytics focusing on business impact. Address:
        
        1. Performance Metrics
           - Key business KPIs and their trends
           - Segment-wise performance analysis
           - Comparative analysis (YoY, MoM, etc.)
           - Market share and positioning data
        
        2. Pattern Analysis
           - Seasonal trends and cycles
           - Customer behavior patterns
           - Operational efficiency metrics
           - Revenue and cost patterns
        
        3. Distribution Analysis
           - Customer segmentation insights
           - Product/service performance distribution
           - Geographic or demographic patterns
           - Resource utilization distribution
        
        4. Relationship Analysis
           - Cross-product correlations
           - Customer lifetime value factors
           - Cost-revenue relationships
           - Operational dependencies
        
        Dataset Context:
        {eda_result}
        
        Data Sample:
        {data.head(5).to_json()}
        
        Provide specific numbers, percentages, and growth rates where applicable.
        Focus on insights that drive business value.
        """,
        
        "Diagnostic Analytics": f"""
        Perform in-depth diagnostic analytics to understand business drivers. Focus on:
        
        1. Root Cause Analysis
           - Primary factors affecting key metrics
           - Underlying causes of performance changes
           - Impact of business decisions
           - Market influence factors
        
        2. Performance Attribution
           - Contributors to success/failure
           - Impact of different business strategies
           - Resource allocation effectiveness
           - Customer response factors
        
        3. Variance Analysis
           - Deviation from targets/forecasts
           - Unusual patterns or anomalies
           - Performance gaps
           - Market comparison deltas
        
        4. Impact Assessment
           - Effect of external factors
           - Internal policy impacts
           - Competition influence
           - Technology adoption effects
        
        Dataset Context:
        {eda_result}
        
        Data Sample:
        {data.head(5).to_json()}
        
        Include quantitative evidence for each finding.
        Link causes to specific business outcomes.
        """,
        
        "Predictive Analytics": f"""
        Perform detailed predictive analytics with business growth focus. Address:
        
        1. Future Trends
           - Revenue/profit projections
           - Market share predictions
           - Customer behavior forecasts
           - Operational needs forecast
        
        2. Risk Assessment
           - Potential market changes
           - Customer churn probability
           - Resource requirement predictions
           - Competitive threat analysis
        
        3. Opportunity Identification
           - Growth potential areas
           - Market expansion possibilities
           - Product development opportunities
           - Efficiency improvement potential
        
        4. Scenario Analysis
           - Best/worst case projections
           - Market condition impacts
           - Resource requirement scenarios
           - ROI predictions
        
        Dataset Context:
        {eda_result}
        
        Data Sample:
        {data.head(5).to_json()}
        
        Include confidence levels and assumptions.
        Provide timeframe-specific predictions.
        """,
        
        "Prescriptive Analytics": f"""
        Perform actionable prescriptive analytics with implementation focus. Address:
        
        1. Strategic Recommendations
           - Specific action items
           - Resource allocation suggestions
           - Timeline recommendations
           - Priority rankings
        
        2. Implementation Planning
           - Required resources
           - Cost estimates
           - Risk mitigation strategies
           - Success metrics
        
        3. Impact Projections
           - Expected ROI
           - Performance improvements
           - Resource savings
           - Market position changes
        
        4. Optimization Strategies
           - Process improvement recommendations
           - Resource utilization optimization
           - Cost reduction opportunities
           - Revenue maximization approaches
        
        Dataset Context:
        {eda_result}
        
        Data Sample:
        {data.head(5).to_json()}
        
        Include specific metrics for success measurement.
        Provide detailed implementation steps.
        """
    }
    
    analysis_prompt = analysis_prompts.get(
            analysis_type,
            f"""
            Perform comprehensive data analysis with focus on actionable business insights.
            Consider all aspects of the data provided.
            
            Dataset Context:
            {eda_result}
            
            Data Sample:
            {data.head(5).to_json()}
            
            Provide in-depth analysis and actionable recommendations.
            """
        )
        
    response = gemini.generate_response(analysis_prompt)
    return response

def create_podcast_script(analysis_type, analysis_results):
    """Create a detailed, business-focused conversation that bridges data analysis with decision-making."""
    # Initialize Gemini interface
    gemini = GeminiInterface()
    
    script_prompt = f"""
    Create a detailed, strategic conversation between Alex (a senior data analyst) and Sarah (a business strategy consultant) 
    discussing the {analysis_type} results. Make it comprehensive and actionable for business decision-makers.
    
    Conversation Structure:
    1. Introduction
       - Alex introduces the analysis type and methodology
       - Sarah frames why this analysis matters for business decisions
    
    2. Key Findings (Most Detailed Section)
       - Present 3-4 major insights, each with:
         * Data-backed evidence
         * Business implications
         * Potential impact on different business areas (revenue, costs, operations, etc.)
       - Sarah asks probing questions about business relevance
       - Alex provides detailed explanations with specific numbers and trends
    
    3. Risk Analysis
       - Discuss potential limitations or risks
       - Address data reliability and confidence levels
       - Consider market context and external factors
    
    4. Strategic Recommendations
       - 2-3 concrete, actionable recommendations
       - Expected outcomes and timeline
       - Resource requirements or constraints
       - Implementation considerations
    
    5. ROI Discussion
       - Potential costs and benefits
       - Expected timeframe for results
       - Success metrics to track
    
    6. Next Steps
       - Specific action items
       - Monitoring and adjustment strategies
       - Follow-up analysis needs
    
    Guidelines:
    - Include specific metrics, percentages, and numbers from the analysis
    - Use industry-specific context where relevant
    - Break down complex concepts but maintain professional depth
    - Include brief pauses for reflection ("That's interesting...") to maintain natural flow
    - Reference real-world business scenarios and examples
    - End with a clear, actionable summary
    
    Analysis Results:
    {analysis_results}
    
    Format as:
    Alex: [dialogue]
    Sarah: [dialogue]
    
    Keep each response focused but detailed enough for meaningful business insights.
    """
    response = gemini.generate_response(script_prompt)
    return response

def create_process_podcast(process_insights):
    """Create a podcast script about process mining insights."""
    # Initialize Gemini interface
    gemini = GeminiInterface()
    
    script_prompt = f"""
    Create an informative conversation between Alex (a process mining expert) and Sarah (a business analyst)
    discussing the results of a process mining analysis. Focus on making the insights accessible and actionable.
    
    Conversation Structure:
    1. Introduction
       - Alex introduces the process mining analysis performed
       - Sarah asks about the key findings and significance
    
    2. Key Process Insights
       - Alex explains the main process patterns discovered
       - Discussion of bottlenecks, variations, and compliance issues
       - Identification of efficient vs. inefficient process paths
    
    3. Performance Analysis
       - Detailed discussion of throughput times and waiting times
       - Resource allocation insights
       - Conformance issues
    
    4. Business Impact
       - Translation of process insights into business implications
       - Cost implications and operational efficiency
       - Customer experience effects
    
    5. Recommendations
       - Specific actionable recommendations based on the insights
       - Potential process improvements
       - Monitoring suggestions
    
    Process Insights:
    {process_insights}
    
    Format as:
    Alex: [dialogue]
    Sarah: [dialogue]
    
    Keep each response focused but detailed enough for meaningful business insights.
    """
    
    response = gemini.generate_response(script_prompt)
    return response

def render_data_mining_page():
    """Render the Data Mining & Analytics page with podcast generation."""
    st.header("Data Mining & Business Analytics")
    
    # Store data mining dataframe in session state
    if 'mining_df' not in st.session_state:
        st.session_state.mining_df = None
    
    # Add separate file upload for data mining
    st.subheader("Upload Data for Mining")
    uploaded_file = st.file_uploader("Choose a CSV file for data mining", type=["csv"], key="data_mining_upload")
    
    # Process the uploaded file
    if uploaded_file is not None:
        try:
            # Read CSV file and store in session state
            df = pd.read_csv(uploaded_file)
            st.session_state.mining_df = df
            
            # Display data summary
            st.subheader("Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Data Shape:", df.shape)
                st.write("Missing Values:", df.isna().sum().sum())
            with col2:
                st.write("Data Types:")
                st.write(df.dtypes)
            
            # Display data sample
            with st.expander("Review Data Sample"):
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            st.error("Please check your CSV file format and try again.")
    
    # If no data is uploaded yet, show a message and return
    if st.session_state.mining_df is None:
        st.info("Please upload a CSV file to perform data mining and generate insights.")
        return
        
    # Now we have data in session state, proceed with analysis
    df = st.session_state.mining_df
    st.subheader("Perform Advanced Data Mining")
    
    # Step 1: Exploratory Data Analysis
    if st.button("1. Perform Exploratory Data Analysis"):
        with st.spinner("Performing detailed exploratory data analysis..."):
            # Pass actual dataframe to perform_eda
            eda_result = perform_eda(df)
            st.session_state.eda_result = eda_result
            st.success("EDA completed!")
            
        st.subheader("EDA Results")
        st.write(eda_result)
    
    # Check if EDA has been run
    if 'eda_result' in st.session_state:
        # Step 2: Determine Analysis Type
        if st.button("2. Determine Optimal Analysis Type"):
            with st.spinner("Determining the best analysis approach..."):
                # Pass actual dataframe and EDA results
                analysis_decision = determine_analysis_type(st.session_state.eda_result, df)
                st.session_state.analysis_decision = analysis_decision
                
                # Extract analysis type from the response
                try:
                    analysis_type = analysis_decision.split('\n')[0].replace('Analysis Type:', '').strip()
                    st.session_state.analysis_type = analysis_type
                except Exception as e:
                    st.warning(f"Could not automatically extract analysis type: {e}")
                    st.session_state.analysis_type = "Descriptive Analytics"
            
            st.subheader("Analysis Strategy")
            st.write(analysis_decision)
        
        # Check if analysis type has been determined
        if 'analysis_type' in st.session_state and 'analysis_decision' in st.session_state:
            # Step 3: Perform Specific Analysis
            if st.button(f"3. Perform {st.session_state.analysis_type}"):
                with st.spinner(f"Performing detailed {st.session_state.analysis_type.lower()}..."):
                    # Pass actual dataframe and context
                    analysis_results = perform_specific_analysis(
                        df, 
                        st.session_state.analysis_type, 
                        st.session_state.eda_result
                    )
                    st.session_state.analysis_results = analysis_results
                
                st.subheader("Analysis Results")
                st.write(analysis_results)
            
            # Check if specific analysis has been performed
            if 'analysis_results' in st.session_state:
                # Step 4: Generate Podcast Script
                if st.button("4. Generate Business Podcast Script"):
                    with st.spinner("Creating comprehensive business insights podcast script..."):
                        podcast_script = create_podcast_script(
                            st.session_state.analysis_type, 
                            st.session_state.analysis_results
                        )
                        st.session_state.podcast_script = podcast_script
                    
                    st.subheader("Business Insights Podcast Script")
                    st.text_area("Podcast Script", podcast_script, height=300)
                    
                    # Offer script download
                    script_bytes = podcast_script.encode()
                    script_filename = f"business_podcast_{uuid.uuid4().hex[:8]}.txt"
                    st.download_button(
                        label="Download Podcast Script",
                        data=script_bytes,
                        file_name=script_filename,
                        mime="text/plain"
                    )
                
                # Step 5: Generate Audio Podcast
                if 'podcast_script' in st.session_state:
                    if st.button("5. Generate Audio Podcast"):
                        with st.spinner("Generating professional audio presentation..."):
                            audio_generator = AudioGenerator()
                            audio_path = audio_generator.generate_audio(st.session_state.podcast_script)
                            
                            if audio_path:
                                st.session_state.audio_file_path = audio_path
                                st.success("Audio podcast generated successfully!")
                            else:
                                st.error("Error generating audio. Please check TTS configuration.")
                    
                    # Display audio if generated
                    if 'audio_file_path' in st.session_state and st.session_state.audio_file_path:
                        st.subheader("Business Insights Audio Podcast")
                        
                        try:
                            with open(st.session_state.audio_file_path, "rb") as audio_file:
                                audio_bytes = audio_file.read()
                                
                            st.audio(audio_bytes, format="audio/mp3")
                            
                            # Create download button for the audio file
                            audio_filename = f"business_podcast_{uuid.uuid4().hex[:8]}.mp3"
                            st.download_button(
                                label="Download Audio Podcast",
                                data=audio_bytes,
                                file_name=audio_filename,
                                mime="audio/mp3"
                            )
                        except Exception as e:
                            st.error(f"Error displaying audio: {e}")

def render_ai_insights_page():
    """Render AI Insights page with audio podcast generation."""
    if st.session_state.event_log is None:
        st.warning("Please upload an event log first.")
        return
        
    st.header("AI Process Insights")
    
    try:
        # Initialize AI components
        gemini = GeminiInterface()
        insights = InsightGenerator(gemini)
        
        # Generate insights
        st.subheader("Process Analysis")
        if st.button("Generate Process Insights"):
            with st.spinner("Analyzing process data..."):
                # Convert event log to dataframe for context
                df = pm4py.convert_to_dataframe(st.session_state.event_log)
                process_context = f"""
                Event Log Summary:
                - Total Cases: {len(df['case:concept:name'].unique())}
                - Total Events: {len(df)}
                - Activities: {', '.join(df['concept:name'].unique())}
                - Timeframe: {pd.to_datetime(df['time:timestamp']).min()} to {pd.to_datetime(df['time:timestamp']).max()}
                """
                
                process_insights = insights.generate_process_insights(
                    process_context,
                    "Process model analysis"
                )
                st.session_state.process_insights = process_insights
            
            # Display insights
            st.write(process_insights)
            
            # Option to create podcast from insights
            if st.button("Create Audio Podcast from Insights"):
                with st.spinner("Creating podcast script..."):
                    # Generate conversation script about process insights
                    podcast_script = create_process_podcast(process_insights)
                    st.session_state.process_podcast_script = podcast_script
                
                st.subheader("Process Analysis Podcast Script")
                st.text_area("Script", podcast_script, height=250)
                
                # Generate audio
                if st.button("Generate Process Audio Podcast"):
                    with st.spinner("Creating audio..."):
                        audio_generator = AudioGenerator()
                        audio_path = audio_generator.generate_audio(podcast_script)
                        
                        if audio_path:
                            st.session_state.process_audio_path = audio_path
                            st.success("Audio podcast generated successfully!")
                        else:
                            st.error("Error generating audio. Please check TTS configuration.")
                
                # Display audio if generated
                if 'process_audio_path' in st.session_state and st.session_state.process_audio_path:
                    st.subheader("Process Analysis Audio Podcast")
                    
                    try:
                        with open(st.session_state.process_audio_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            
                        st.audio(audio_bytes, format="audio/mp3")
                        
                        # Create download button for the audio file
                        audio_filename = f"process_podcast_{uuid.uuid4().hex[:8]}.mp3"
                        st.download_button(
                            label="Download Process Audio Podcast",
                            data=audio_bytes,
                            file_name=audio_filename,
                            mime="audio/mp3"
                        )
                    except Exception as e:
                        st.error(f"Error displaying audio: {e}")
        
        # Interactive query section
        st.subheader("Ask Questions")
        
        user_query = st.text_input("Ask a question about the process:")
        
        if user_query:
            with st.spinner("Analyzing your question..."):
                answer = insights.generate_conversational_analysis(
                    user_query,
                    str(st.session_state.event_log)
                )
                st.session_state.last_query = user_query
                st.session_state.last_answer = answer
            
            st.write("Answer:", answer)
            
            # Option to create audio response
            if st.button("Create Audio Response"):
                with st.spinner("Generating audio response..."):
                    # Create a conversational audio script
                    convo_script = f"""
                    Alex: I was asked: {user_query}
                    
                    Sarah: That's an interesting question about the process. Let me help analyze that.
                    
                    Alex: {answer}
                    
                    Sarah: Thank you for that thorough explanation. That gives us good insights into the process.
                    """
                    
                    # Generate audio
                    audio_generator = AudioGenerator()
                    audio_path = audio_generator.generate_audio(convo_script)
                    
                    if audio_path:
                        st.session_state.query_audio_path = audio_path
                        st.success("Audio response generated!")
                    else:
                        st.error("Error generating audio response.")
            
            # Display audio if generated
            if 'query_audio_path' in st.session_state and st.session_state.query_audio_path:
                try:
                    with open(st.session_state.query_audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        
                    st.audio(audio_bytes, format="audio/mp3")
                    
                    audio_filename = f"query_response_{uuid.uuid4().hex[:8]}.mp3"
                    st.download_button(
                        label="Download Audio Response",
                        data=audio_bytes,
                        file_name=audio_filename,
                        mime="audio/mp3"
                    )
                except Exception as e:
                    st.error(f"Error displaying audio: {e}")
        
    except Exception as e:
        st.error(f"Error generating insights: {e}")

def fix_dfg_visualization():
    """Apply fixes for DFG visualization issues"""
    try:
        logging.info("Applying DFG visualization fixes...")
    except Exception as e:
        logging.error(f"Error applying DFG visualization fixes: {e}")

def fix_bpmn_visualization():
    """Apply fixes for BPMN visualization issues"""
    try:
        logging.info("Applying BPMN visualization fixes...")
    except Exception as e:
        logging.error(f"Error applying BPMN visualization fixes: {e}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Set page config
    st.set_page_config(
        page_title="Process Mining + AI Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
        # Apply visualizer fixes
        fix_dfg_visualization()
        fix_bpmn_visualization()
        
        # Initialize session state
        initialize_session_state()
        
        # Main title
        st.title("Process Mining + AI Analytics Platform")
        
        # Sidebar
        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Choose a page",
                ["Upload & Process", "Process Discovery", "Performance Analysis", 
                 "Statistical Analysis", "AI Insights", "Data Mining & Podcast"]
            )
        
        # Main content based on selected page
        if page == "Upload & Process":
            render_upload_page()
        elif page == "Process Discovery":
            if st.session_state.event_log is not None:
                st.header("Process Discovery")
                
                # Initialize components
                discovery = ProcessDiscovery()
                visualizer = ProcessMapVisualizer()
                
                # Process model selection
                model_type = st.selectbox(
                    "Select Process Model Type",
                    ["Directly-Follows Graph", "Petri Net", "BPMN Model"]
                )
                
                try:
                    if model_type == "Directly-Follows Graph":
                        st.subheader("Directly-Follows Graph Visualization")
                        
                        # Discover DFG
                        dfg, start_activities, end_activities = discovery.discover_dfg(st.session_state.event_log)
                        
                        # Visualize
                        visualizer.visualize_dfg(dfg, start_activities, end_activities)
                        
                    elif model_type == "Petri Net":
                        st.subheader("Petri Net Model")
                        
                        # Discover Petri net
                        net, initial_marking, final_marking = discovery.discover_process_map(st.session_state.event_log)
                        
                        # Visualize
                        visualizer.visualize_process_map(net, initial_marking, final_marking)
                        
                    elif model_type == "BPMN Model":
                        st.subheader("BPMN Model")
                        
                        # Discover BPMN model
                        bpmn_model = discovery.discover_bpmn_model(st.session_state.event_log)
                        
                        # Visualize
                        visualizer.visualize_bpmn(bpmn_model)
                except Exception as e:
                    st.error(f"Error in model discovery/visualization: {e}")
                    
                    # Fallback to basic DFG visualization
                    try:
                        dfg = pm4py.discover_directly_follows_graph(st.session_state.event_log)
                        parameters = {"format": "png", "bgcolor": "white", "rankdir": "LR"}
                        gviz = pm4py.visualization.dfg.visualizer.apply(dfg, parameters=parameters)
                        st.graphviz_chart(gviz)
                    except Exception as e2:
                        st.error(f"Basic visualization also failed: {e2}")
            else:
                st.warning("Please upload an event log first")

        elif page == "Performance Analysis":
            if st.session_state.event_log is not None:
                st.header("Performance Analysis")
                
                # Initialize components
                performance = PerformanceAnalyzer()
                charts = ChartGenerator()
                
                try:
                    # Calculate performance metrics
                    cycle_time = performance.calculate_cycle_time(st.session_state.event_log)
                    waiting_time = performance.calculate_waiting_time(st.session_state.event_log)
                    sojourn_time = performance.calculate_sojourn_time(st.session_state.event_log)
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Cycle Time Analysis")
                        charts.create_cycle_time_chart(cycle_time)
                    
                    with col2:
                        st.subheader("Activity Waiting Times")
                        st.write(waiting_time)
                    
                    st.subheader("Process Timeline")
                    charts.create_performance_timeline(st.session_state.event_log)
                
                except Exception as e:
                    st.error(f"Error in performance analysis: {e}")
            else:
                st.warning("Please upload an event log first")

        elif page == "Statistical Analysis":
            if st.session_state.event_log is not None:
                st.header("Statistical Analysis")
                
                # Initialize components
                stats = ProcessStatistics()
                charts = ChartGenerator()
                
                try:
                    # Get all statistics
                    case_stats = stats.get_case_statistics(st.session_state.event_log)
                    activity_stats = stats.get_activity_statistics(st.session_state.event_log)
                    resource_stats = stats.get_resource_statistics(st.session_state.event_log)
                    process_kpis = stats.get_process_kpis(st.session_state.event_log)
                    
                    # Display Process Overview
                    st.subheader("Process Overview")
                    kpi_cols = st.columns(4)
                    with kpi_cols[0]:
                        st.metric("Total Cases", process_kpis['process']['total_cases'])
                    with kpi_cols[1]:
                        st.metric("Total Events", process_kpis['process']['total_events'])
                    with kpi_cols[2]:
                        st.metric("Avg Duration (hrs)", f"{process_kpis['time']['avg_case_duration']:.1f}")
                    with kpi_cols[3]:
                        st.metric("Events per Case", f"{process_kpis['process']['events_per_case']:.1f}")
                    
                    # Case Analysis
                    st.subheader("Case Analysis")
                    case_tabs = st.tabs(["Overview", "Performance", "Business Metrics"])
                    
                    with case_tabs[0]:
                        st.write("Case Duration Distribution")
                        case_durations = [(case, stats['temporal']['duration_hours']) 
                                        for case, stats in case_stats.items()]
                        charts.create_cycle_time_chart(case_durations)
                    
                    with case_tabs[1]:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Resource Utilization")
                            resource_data = pd.DataFrame(
                                [(r, stats['workload']['total_activities']) 
                                 for r, stats in resource_stats.items()],
                                columns=['Resource', 'Activities']
                            )
                            st.bar_chart(resource_data.set_index('Resource'))
                        
                        with col2:
                            st.write("Activity Distribution")
                            charts.create_activity_frequency_chart(
                                {act: stats['frequency']['total_occurrences'] 
                                 for act, stats in activity_stats.items()}
                            )
                    
                    with case_tabs[2]:
                        if 'business' in process_kpis:
                            business_cols = st.columns(2)
                            with business_cols[0]:
                                st.metric("Total Claim Value", 
                                        f"${process_kpis['business']['total_claim_value']:,.2f}")
                                st.metric("Avg Claim Value",
                                        f"${process_kpis['business']['avg_claim_value']:,.2f}")
                            with business_cols[1]:
                                st.metric("Total Process Cost",
                                        f"${process_kpis['business']['total_process_cost']:,.2f}")
                                st.metric("Avg Process Cost",
                                        f"${process_kpis['business']['avg_process_cost']:,.2f}")
                        else:
                            st.info("No business metrics available in the event log")
                    
                    # Resource Analysis
                    st.subheader("Resource Analysis")
                    resource_tabs = st.tabs(["Workload", "Performance"])
                    
                    with resource_tabs[0]:
                        for resource, stats in resource_stats.items():
                            with st.expander(f"Resource: {resource}"):
                                rcol1, rcol2 = st.columns(2)
                                with rcol1:
                                    st.metric("Total Activities", stats['workload']['total_activities'])
                                    st.metric("Unique Cases", stats['workload']['unique_cases'])
                                with rcol2:
                                    st.metric("Active Hours", f"{stats['time']['active_hours']:.1f}")
                                    if 'performance' in stats and 'total_cost' in stats['performance']:
                                        st.metric("Total Cost", f"${stats['performance']['total_cost']:,.2f}")
                    
                    with resource_tabs[1]:
                        if any('costs' in stats['performance'] for stats in resource_stats.values()):
                            performance_data = pd.DataFrame([
                                {
                                    'Resource': r,
                                    'Total Cost': stats['performance'].get('total_cost', 0),
                                    'Avg Cost': stats['performance'].get('avg_cost_per_activity', 0)
                                }
                                for r, stats in resource_stats.items()
                            ])
                            st.write("Resource Cost Analysis")
                            st.dataframe(performance_data)
                        else:
                            st.info("No cost information available for resources")
                    
                except Exception as e:
                    st.error(f"Error in statistical analysis: {e}")
                    st.error("Details:", str(e))
            else:
                st.warning("Please upload an event log first")

        elif page == "AI Insights":
            render_ai_insights_page()
                
        elif page == "Data Mining & Podcast":
            render_data_mining_page()
            
    except Exception as e:
        st.error(f"An error occurred in the application: {e}")
        logging.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()