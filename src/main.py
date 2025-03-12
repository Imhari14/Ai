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
from process_mining import (
    ProcessDiscovery, 
    PerformanceAnalyzer, 
    ProcessStatistics,
    calculate_process_variants
)
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
    if 'process_insights' not in st.session_state:
        st.session_state.process_insights = None  # Store process insights
    if 'process_podcast_script' not in st.session_state:
        st.session_state.process_podcast_script = None
    if 'process_audio_path' not in st.session_state:
        st.session_state.process_audio_path = None  # Store audio path
    if 'last_query' not in st.session_state:
        st.session_state.last_query = None
    if 'last_answer' not in st.session_state:
        st.session_state.last_answer = None
    if 'query_audio_path' not in st.session_state:
        st.session_state.query_audio_path = None
    if 'eda_result' not in st.session_state:
        st.session_state.eda_result = None
    if 'analysis_decision' not in st.session_state:
        st.session_state.analysis_decision = None

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
            self.tts_client = texttospeech.TextToSpeechClient()
            voices = self.tts_client.list_voices()
            logging.info(f"TTS client initialized successfully with {len(voices.voices)} available voices")
            
            self.voice_profiles = [
                {
                    'name': 'en-US-Studio-M',
                    'language_code': 'en-US',
                    'ssml_gender': texttospeech.SsmlVoiceGender.MALE
                },
                {
                    'name': 'en-US-Studio-O',
                    'language_code': 'en-US',
                    'ssml_gender': texttospeech.SsmlVoiceGender.FEMALE
                }
            ]
        except Exception as e:
            self.tts_client = None
            logging.error(f"Error initializing TTS client: {e}", exc_info=True)

    def generate_audio_segment(self, text, voice_profile):
        if self.tts_client is None:
            logging.error("TTS client is not initialized")
            return None
            
        try:
            if len(text) > 5000:
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
            
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            return response.audio_content
        except Exception as e:
            logging.error(f"Error generating audio segment: {e}", exc_info=True)
            return None

    def generate_audio(self, script):
        if self.tts_client is None:
            logging.error("Cannot generate audio - TTS client not initialized")
            return None
            
        segments = script.split('\n')
        audio_segments = []
        
        for segment in segments:
            if not segment.strip() or ':' not in segment:
                continue
                
            try:
                speaker, dialogue = segment.split(':', 1)
                dialogue = dialogue.strip()
                
                if not dialogue:
                    continue

                voice_profile = self.voice_profiles[0] if 'Alex' in speaker else self.voice_profiles[1]
                
                audio_content = self.generate_audio_segment(dialogue, voice_profile)
                if audio_content:
                    audio_segments.append(audio_content)
            except Exception as e:
                logging.error(f"Error processing segment: {e}")

        if not audio_segments:
            return None

        try:
            combined_audio = b''.join(audio_segments)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                temp_audio.write(combined_audio)
                return temp_audio.name
        except Exception as e:
            logging.error(f"Error saving audio file: {e}", exc_info=True)
            return None

def render_upload_page():
    """Render the file upload page"""
    st.header("Upload Event Log")
    
    processor = EventLogProcessor()
    
    uploaded_file = st.file_uploader("Choose a CSV or XES file", type=["csv", "xes"])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "csv":
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("CSV Column Mapping")
                
                st.write("Raw Data Sample:")
                st.write(df.head())
                
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
                    column_mapping = {
                        case_id_col: 'case:concept:name',
                        activity_col: 'concept:name',
                        timestamp_col: 'time:timestamp'
                    }
                    
                    if resource_col != "None":
                        column_mapping[resource_col] = 'org:resource'
                    if cost_col != "None":
                        column_mapping[cost_col] = 'cost'
                    
                    df = df.rename(columns=column_mapping)
                    
                    try:
                        event_log = processor.convert_csv_to_event_log(df)
                        st.session_state.event_log = event_log
                        
                        st.success("CSV file successfully processed!")
                        st.subheader("Processed Event Log Sample")
                        
                        processed_df = pm4py.convert_to_dataframe(event_log)
                        st.write(processed_df.head())
                        
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
                
                st.subheader("Sample of Loaded Data")
                st.write(pd.DataFrame(pm4py.convert_to_dataframe(log)[0:5]))
            except Exception as e:
                st.error(f"Error loading XES file: {e}")
        else:
            st.error("Unsupported file format. Please upload a CSV or XES file.")

def perform_eda(data):
    """Perform enhanced EDA using the Gemini API."""
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
    """
    response = gemini.generate_response(eda_prompt)
    return response

def determine_analysis_type(eda_result, data):
    """Determine the most appropriate type of analysis based on EDA results."""
    gemini = GeminiInterface()
    
    analysis_prompt = f"""
    Based on the EDA results below, determine which type of data analysis would be most valuable for this dataset.
    Choose from: Descriptive, Diagnostic, Predictive, or Prescriptive Analytics.
    
    EDA Results:
    {eda_result}
    
    Dataset Summary:
    - Shape: {data.shape}
    - Columns: {', '.join(data.columns)}
    - Data Sample: {data.head(3).to_json()}
    """
    response = gemini.generate_response(analysis_prompt)
    return response

def perform_specific_analysis(data, analysis_type, eda_result):
    """Perform the specific type of analysis determined to be most appropriate."""
    gemini = GeminiInterface()
    
    analysis_prompt = f"""
    Perform {analysis_type} on the dataset with focus on actionable business insights.
    
    Dataset Context:
    {eda_result}
    
    Data Sample:
    {data.head(5).to_json()}
    """
    response = gemini.generate_response(analysis_prompt)
    return response

def create_podcast_script(analysis_type, analysis_results):
    """Create a detailed, business-focused conversation script."""
    gemini = GeminiInterface()
    
    script_prompt = f"""
    Create a conversation between Alex (data analyst) and Sarah (business consultant) 
    discussing the {analysis_type} results with actionable insights.
    
    Analysis Results:
    {analysis_results}
    """
    response = gemini.generate_response(script_prompt)
    return response

def create_process_podcast(process_insights):
    """Create a podcast script about process mining insights."""
    gemini = GeminiInterface()
    
    script_prompt = f"""
    Create a conversation between Alex (process mining expert) and Sarah (business analyst)
    discussing the process mining analysis results.
    
    Process Insights:
    {process_insights}
    """
    response = gemini.generate_response(script_prompt)
    return response

def render_process_discovery_page(event_log):
    """Render the Process Discovery page with enhanced error handling"""
    st.header("Process Discovery")
    
    discovery = ProcessDiscovery()
    visualizer = ProcessMapVisualizer()
    
    model_type = st.selectbox(
        "Select Process Model Type",
        ["Directly-Follows Graph", "Petri Net", "BPMN Model"]
    )
    
    try:
        if model_type == "Directly-Follows Graph":
            st.subheader("Directly-Follows Graph Visualization")
            
            try:
                dfg, start_activities, end_activities = discovery.discover_dfg(event_log)
                gviz = visualizer.visualize_dfg(dfg, start_activities, end_activities)
                if gviz is None:
                    raise ValueError("DFG visualization failed")
                    
            except Exception as e:
                logging.error(f"Error in DFG discovery/visualization: {e}")
                st.error("Primary DFG visualization failed. Attempting basic visualization...")
                
                try:
                    dfg = pm4py.discover_directly_follows_graph(event_log)
                    if isinstance(dfg, list):
                        dfg_dict = {(item[0][0], item[0][1]): item[1] for item in dfg}
                    else:
                        dfg_dict = dfg
                    
                    parameters = {
                        "format": "png",
                        "bgcolor": "white",
                        "rankdir": "LR"
                    }
                    
                    gviz = pm4py.visualization.dfg.visualizer.apply(dfg_dict, parameters=parameters)
                    st.graphviz_chart(gviz)
                    
                except Exception as e2:
                    st.error(f"Basic DFG visualization also failed: {e2}")
                    st.info("Please check your event log format and try again.")
                
        elif model_type == "Petri Net":
            st.subheader("Petri Net Model")
            
            try:
                net, initial_marking, final_marking = discovery.discover_process_map(event_log)
                gviz = visualizer.visualize_process_map(net, initial_marking, final_marking)
                if gviz is None:
                    raise ValueError("Petri net visualization failed")
                    
            except Exception as e:
                st.error(f"Error in Petri net discovery/visualization: {e}")
                st.info("Please check your event log format and try again.")
            
        elif model_type == "BPMN Model":
            st.subheader("BPMN Model")
            
            try:
                bpmn_model = discovery.discover_bpmn_model(event_log)
                gviz = visualizer.visualize_bpmn(bpmn_model)
                if gviz is None:
                    raise ValueError("BPMN visualization failed")
                    
            except Exception as e:
                st.error(f"Error in BPMN discovery/visualization: {e}")
                st.info("Please check your event log format and try again.")
                
    except Exception as e:
        st.error(f"Error in process discovery: {e}")
        st.info("Please ensure your event log is in the correct format and contains valid process data.")

def render_performance_page(event_log):
    """Render the Performance Analysis page"""
    st.header("Performance Analysis")
    
    performance = PerformanceAnalyzer()
    charts = ChartGenerator()
    
    try:
        cycle_time = performance.calculate_cycle_time(event_log)
        waiting_time = performance.calculate_waiting_time(event_log)
        sojourn_time = performance.calculate_sojourn_time(event_log)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cycle Time Analysis")
            charts.create_cycle_time_chart(cycle_time)
        
        with col2:
            st.subheader("Activity Waiting Times")
            st.write(waiting_time)
        
        st.subheader("Process Timeline")
        charts.create_performance_timeline(event_log)
    
    except Exception as e:
        st.error(f"Error in performance analysis: {e}")

def render_statistics_page(event_log):
    """Render the Statistical Analysis page"""
    st.header("Statistical Analysis")
    
    stats = ProcessStatistics()
    charts = ChartGenerator()
    
    try:
        case_stats = stats.get_case_statistics(event_log)
        activity_stats = stats.get_activity_statistics(event_log)
        resource_stats = stats.get_resource_statistics(event_log)
        process_kpis = stats.get_process_kpis(event_log)
        
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
        
        st.subheader("Activity Statistics")
        st.write(activity_stats)
        
        st.subheader("Resource Analysis")
        st.write(resource_stats)
        
    except Exception as e:
        st.error(f"Error in statistical analysis: {e}")

def render_ai_insights_page():
    """Render AI Insights page with enhanced process analysis"""
    if st.session_state.event_log is None:
        st.warning("Please upload an event log first.")
        return
        
    st.header("AI Process Insights")
    
    try:
        # Initialize components
        gemini = GeminiInterface()
        insights = InsightGenerator(gemini)
        discovery = ProcessDiscovery()
        performance = PerformanceAnalyzer()
        stats = ProcessStatistics()
        
        if st.session_state.process_insights is None: # Only generate insights if not already generated
            if st.button("Generate Process Insights"):
                with st.spinner("Analyzing process data..."):
                    # Convert event log to dataframe for analysis
                    df = pm4py.convert_to_dataframe(st.session_state.event_log)
                    
                    # Get process statistics
                    case_stats = stats.get_case_statistics(st.session_state.event_log)
                    activity_stats = stats.get_activity_statistics(st.session_state.event_log)
                    resource_stats = stats.get_resource_statistics(st.session_state.event_log)
                    
                    # Calculate performance metrics
                    cycle_time = performance.calculate_cycle_time(st.session_state.event_log)
                    waiting_time = performance.calculate_waiting_time(st.session_state.event_log)
                    
                    # Calculate process variants safely
                    total_variants, top_variants_str = calculate_process_variants(df)
                    
                    # Get process model information
                    try:
                        dfg, start_activities, end_activities = discovery.discover_dfg(st.session_state.event_log)
                        process_model = "Directly-Follows Graph (DFG)"
                    except Exception as e1:
                        try:
                            bpmn_model = discovery.discover_bpmn_model(st.session_state.event_log)
                            process_model = "BPMN Model"
                        except Exception as e2:
                            process_model = "Unable to generate process model"
                            start_activities = {}
                            end_activities = {}
                            dfg = {}
                    
                    # Create detailed process context
                    process_context = f"""
                    Process Analysis Details:

                    1. Process Overview:
                    - Total Cases: {len(df['case:concept:name'].unique())}
                    - Total Events: {len(df)}
                    - Unique Activities: {len(df['concept:name'].unique())}
                    - Process Model Type: {process_model}
                    - Timeframe: {pd.to_datetime(df['time:timestamp']).min()} to {pd.to_datetime(df['time:timestamp']).max()}

                    2. Activities Information:
                    - Complete Activity List: {', '.join(sorted(df['concept:name'].unique()))}
                    - Start Activities: {', '.join(start_activities.keys()) if start_activities else 'Not available'}
                    - End Activities: {', '.join(end_activities.keys()) if end_activities else 'Not available'}
                    
                    3. Resource Information:
                    - Total Resources: {len(df['org:resource'].unique()) if 'org:resource' in df.columns else 'Not available'}
                    - Resource List: {', '.join(sorted(df['org:resource'].unique())) if 'org:resource' in df.columns else 'Not available'}
                    
                    4. Performance Metrics:
                    - Average Case Duration: {sum(d for _, d in cycle_time)/len(cycle_time) if cycle_time else 'Not available'} hours
                    - Min Case Duration: {min(d for _, d in cycle_time) if cycle_time else 'Not available'} hours
                    - Max Case Duration: {max(d for _, d in cycle_time) if cycle_time else 'Not available'} hours
                    
                    5. Activity Analysis:
                    Activity Frequencies:
                    {pd.DataFrame(df['concept:name'].value_counts()).to_string() if not df.empty else 'Not available'}
                    
                    6. Process Flow Information:
                    - Direct Follow Relations: {len(dfg) if dfg else 'Not available'}
                    - Total Process Variants: {total_variants}
                    
                    7. Timing Analysis:
                    Average Waiting Times per Activity:
                    {pd.Series(waiting_time).to_string() if waiting_time else 'Not available'}
                    
                    8. Case Analysis:
                    - Average Events per Case: {len(df) / len(df['case:concept:name'].unique()):.2f}
                    - Case Duration Distribution:
                    {df.groupby('case:concept:name').size().describe().to_string() if not df.empty else 'Not available'}

                    9. Process Variants Analysis:
                    Total Unique Variants: {total_variants}
                    Most Common Process Paths:
                    {top_variants_str}
                    """
                    
                    # Generate insights with enhanced context
                    process_insights = insights.generate_process_insights(
                        process_context,
                        "Detailed process mining analysis with emphasis on performance metrics, bottlenecks, and improvement opportunities"
                    )
                    st.session_state.process_insights = process_insights

        # Display insights (always display if they exist)
        if st.session_state.process_insights:
            st.subheader("Process Analysis Results")
            st.write(st.session_state.process_insights)

            # Create podcast functionality
            if st.session_state.process_podcast_script is None: # Only generate if not already generated
                if st.button("Create Audio Podcast from Insights"):
                    with st.spinner("Creating podcast script..."):
                        podcast_script = create_process_podcast(st.session_state.process_insights)
                        st.session_state.process_podcast_script = podcast_script
            
            if st.session_state.process_podcast_script:
                st.subheader("Process Analysis Podcast Script")
                st.text_area("Script", st.session_state.process_podcast_script, height=250)
                
                # Generate audio if requested
                if st.button("Generate Process Audio Podcast"):
                    with st.spinner("Creating audio..."):
                        audio_generator = AudioGenerator()
                        audio_path = audio_generator.generate_audio(st.session_state.process_podcast_script)
                        
                        if audio_path:
                            st.session_state.process_audio_path = audio_path
                            st.success("Audio podcast generated successfully!")
                            
                            try:
                                with open(audio_path, "rb") as audio_file:
                                    audio_bytes = audio_file.read()
                                st.audio(audio_bytes, format="audio/mp3")
                                
                                audio_filename = f"process_podcast_{uuid.uuid4().hex[:8]}.mp3"
                                st.download_button(
                                    label="Download Process Audio Podcast",
                                    data=audio_bytes,
                                    file_name=audio_filename,
                                    mime="audio/mp3"
                                )
                            except Exception as e:
                                st.error(f"Error displaying audio: {e}")
                        else:
                            st.error("Error generating audio. Please check TTS configuration.")
        
        # Interactive query section
        st.subheader("Ask Questions")
        user_query = st.text_input("Ask a question about the process:")
        
        if user_query:
            with st.spinner("Analyzing your question..."):
                df = pm4py.convert_to_dataframe(st.session_state.event_log)
                # Create query context
                query_context = f"""
                Process Context:
                - Total Cases: {len(df['case:concept:name'].unique())}
                - Total Events: {len(df)}
                - Activities: {', '.join(sorted(df['concept:name'].unique()))}
                - Resources: {', '.join(sorted(df['org:resource'].unique())) if 'org:resource' in df.columns else 'Not available'}
                - Timeframe: {pd.to_datetime(df['time:timestamp']).min()} to {pd.to_datetime(df['time:timestamp']).max()}
                """
                
                answer = insights.generate_conversational_analysis(user_query, query_context)
                st.session_state.last_query = user_query
                st.session_state.last_answer = answer
            
            st.write("Answer:", answer)
            
            # Option to create audio response
            if st.button("Create Audio Response"):
                with st.spinner("Generating audio response..."):
                    convo_script = f"""
                    Alex: I was asked: {user_query}
                    
                    Sarah: That's an interesting question about the process. Let me help analyze that.
                    
                    Alex: {answer}
                    
                    Sarah: Thank you for that thorough explanation. That gives us good insights into the process.
                    """
                    
                    audio_generator = AudioGenerator()
                    audio_path = audio_generator.generate_audio(convo_script)
                    
                    if audio_path:
                        st.session_state.query_audio_path = audio_path
                        st.success("Audio response generated!")
                        
                        try:
                            with open(audio_path, "rb") as audio_file:
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
        logging.error(f"Error in AI insights generation: {e}", exc_info=True)

def render_data_mining_page():
    """Render the Data Mining & Analytics page"""
    st.header("Data Mining & Business Analytics")
    
    if st.session_state.mining_df is None:
        st.info("Please upload a CSV file to perform data mining and generate insights.")
        return
        
    df = st.session_state.mining_df
    st.subheader("Perform Advanced Data Mining")

    if st.session_state.eda_result is None: # Only perform EDA if it's not already done
        if st.button("1. Perform Exploratory Data Analysis"):
            with st.spinner("Performing detailed exploratory data analysis..."):
                eda_result = perform_eda(df)
                st.session_state.eda_result = eda_result
                st.success("EDA completed!")
                st.write(eda_result)

    if st.session_state.eda_result:
        # Determine Analysis Type
        if st.session_state.analysis_decision is None:
            if st.button("2. Determine Optimal Analysis Type"):
                with st.spinner("Determining the best analysis approach..."):
                    analysis_decision = determine_analysis_type(st.session_state.eda_result, df)
                    st.session_state.analysis_decision = analysis_decision
                    
                    try:
                        analysis_type = analysis_decision.split('\n')[0].replace('Analysis Type:', '').strip()
                        st.session_state.analysis_type = analysis_type
                    except Exception as e:
                        st.warning(f"Could not automatically extract analysis type: {e}")
                        st.session_state.analysis_type = "Descriptive Analytics"
                
                st.subheader("Analysis Strategy")
                st.write(analysis_decision)
            
        if st.session_state.analysis_type and st.session_state.analysis_results is None:  # Perform Analysis
            if st.button(f"3. Perform {st.session_state.analysis_type}"):
                with st.spinner(f"Performing detailed {st.session_state.analysis_type.lower()}..."):
                    analysis_results = perform_specific_analysis(
                        df,
                        st.session_state.analysis_type,
                        st.session_state.eda_result,
                    )
                    st.session_state.analysis_results = analysis_results
                
                st.subheader("Analysis Results")
                st.write(analysis_results)
        
        if st.session_state.analysis_results:  # Generate Podcast Script
            if st.session_state.podcast_script is None: # only generate if not already generated
                if st.button("4. Generate Business Podcast Script"):
                    with st.spinner("Creating business insights podcast script..."):
                        podcast_script = create_podcast_script(
                            st.session_state.analysis_type,
                            st.session_state.analysis_results
                        )
                        st.session_state.podcast_script = podcast_script
                
            if st.session_state.podcast_script:
                st.subheader("Business Insights Podcast Script")
                st.text_area("Podcast Script", st.session_state.podcast_script, height=300)
                
                # Offer script download
                script_bytes = st.session_state.podcast_script.encode()
                script_filename = f"business_podcast_{uuid.uuid4().hex[:8]}.txt"
                st.download_button(
                    label="Download Podcast Script",
                    data=script_bytes,
                    file_name=script_filename,
                    mime="text/plain"
                )
                
                # Option to generate audio
                if st.session_state.audio_file_path is None: # generate audio only if its not already generated
                     if st.button("5. Generate Audio Podcast"):
                        with st.spinner("Generating professional audio presentation..."):
                            audio_generator = AudioGenerator()
                            audio_path = audio_generator.generate_audio(st.session_state.podcast_script)
                            
                            if audio_path:
                                st.session_state.audio_file_path = audio_path
                                st.success("Audio podcast generated successfully!")
                                
                                try:
                                    with open(audio_path, "rb") as audio_file:
                                        audio_bytes = audio_file.read()
                                    
                                    st.audio(audio_bytes, format="audio/mp3")
                                    
                                    audio_filename = f"business_podcast_{uuid.uuid4().hex[:8]}.mp3"
                                    st.download_button(
                                        label="Download Audio Podcast",
                                        data=audio_bytes,
                                        file_name=audio_filename,
                                        mime="audio/mp3"
                                    )
                                except Exception as e:
                                    st.error(f"Error displaying audio: {e}")
                                else:
                                    st.error("Error generating audio. Please check TTS configuration.")
                elif st.session_state.audio_file_path:
                    try:
                         with open(st.session_state.audio_file_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()

                         st.audio(audio_bytes, format="audio/mp3")

                         audio_filename = f"business_podcast_{uuid.uuid4().hex[:8]}.mp3"
                                             
                         st.download_button(
                            label="Download Audio Podcast",
                            data=audio_bytes,
                            file_name=audio_filename,
                            mime="audio/mp3"
                        )
                    except Exception as e:
                        st.error(f"Error displaying audio: {e}")
                
            
def main():
    """Main application entry point"""
    load_dotenv()
    
    st.set_page_config(
        page_title="Process Mining + AI Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
        initialize_session_state()
        
        st.title("Process Mining + AI Analytics Platform")
        
        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Choose a page",
                ["Upload & Process", "Process Discovery", "Performance Analysis", 
                 "Statistical Analysis", "AI Insights", "Data Mining & Podcast"]
            )
        
        if page == "Upload & Process":
            render_upload_page()
        elif page == "Process Discovery":
            if st.session_state.event_log is not None:
                render_process_discovery_page(st.session_state.event_log)
            else:
                st.warning("Please upload an event log first")
        elif page == "Performance Analysis":
            if st.session_state.event_log is not None:
                render_performance_page(st.session_state.event_log)
            else:
                st.warning("Please upload an event log first")
        elif page == "Statistical Analysis":
            if st.session_state.event_log is not None:
                render_statistics_page(st.session_state.event_log)
            else:
                st.warning("Please upload an event log first")
        elif page == "AI Insights":
            render_ai_insights_page()
        elif page == "Data Mining & Podcast":
            uploaded_file = st.file_uploader("Upload CSV File for Data Mining", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.mining_df = df  # Store the dataframe
                    st.success("CSV file uploaded successfully!")
                    st.write("Sample Data:")
                    st.write(df.head())
                    render_data_mining_page()
                except Exception as e:
                    st.error(f"Error loading CSV file: {e}")
            else:
                st.info("Please upload a CSV file to start Data Mining")
            
    except Exception as e:
        st.error(f"An error occurred in the application: {e}")
        logging.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
