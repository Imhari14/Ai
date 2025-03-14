# Process Mining + AI Analytics Platform

An innovative application combining process mining capabilities with generative AI for advanced process analysis and insights.

![Process Mining Platform](https://via.placeholder.com/800x400?text=Process+Mining+Platform)

## Features

- **Event log analysis** (CSV/XES formats)
- **Process mining visualizations**
  - Process maps
  - BPMN diagrams
  - Petri nets
- **Performance analytics**
- **Natural language process querying**
- **Automated insights generation**
- **Interactive process analysis**

## Requirements

- Python 3.9+
- PM4Py 2.7.7+
- Streamlit
- Google Gemini 1.5 Flash API
- Additional dependencies listed in requirements.txt

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/Imhari14/process-mining.git
   cd process-mining
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Gemini API key:
   - Create a `.env` file in the project root
   - Add your Gemini API key: `GEMINI_API_KEY=your_key_here`

## Usage

1. Run the application:
   ```bash
   streamlit run src/main.py
   ```

2. **Testing with Sample Data**:
   - Load the provided `sample_event_log.csv` file which contains a process log for a ticket handling system
   - The sample log includes activities like:
     * Register request
     * Examine (thoroughly/casually)
     * Check ticket
     * Decide
     * Pay compensation/Reject request
   - The log contains timestamps, resources (staff), and costs

3. **Using the Application**:

   a. **Upload & Process**:
      - Upload CSV or XES files
      - Map columns for CSV files
      - View processed data sample
   
   b. **Process Discovery**:
      - View process maps as Petri nets
      - Explore BPMN diagrams
      - Analyze Directly-Follows Graphs (DFG)
   
   c. **Performance Analysis**:
      - Analyze cycle times
      - View waiting times
      - Explore process timelines
   
   d. **Statistical Analysis**:
      - Review case statistics
      - Analyze activity frequencies
      - Explore attribute distributions
   
   e. **AI Insights** (requires Gemini API key):
      - Get automated process insights
      - Ask questions about the process
      - Receive KPI recommendations

## Project Structure

```
.
├── src/
│   ├── main.py                # Main Streamlit application
│   ├── process_mining/
│   │   ├── __init__.py
│   │   ├── discovery.py       # Process discovery algorithms
│   │   ├── performance.py     # Performance analysis
│   │   └── statistics.py      # Statistical analysis
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── gemini.py          # Gemini API integration
│   │   └── insights.py        # AI-driven insights
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── process_maps.py    # Process map visualizations
│   │   └── charts.py          # Performance charts
│   └── utils/
│       ├── __init__.py
│       ├── data_processing.py # Data preprocessing
│       └── config.py          # Configuration management
├── requirements.txt
├── .env                       # Environment variables (not in repo)
└── README.md
```

## Components

### Process Mining Module
Handles core process mining functionality using PM4Py, including:
- Process discovery
- Conformance checking
- Performance analysis
- Statistical computations

### AI Module
Integrates with Gemini 1.5 Flash API for:
- Natural language processing
- Automated insights
- Process understanding
- KPI recommendations

### Visualization Module
Manages all visualization components:
- Process maps
- Performance dashboards
- Statistical charts
- Interactive displays

## License

MIT License

## Contact

- Created by: [Imhari14](https://github.com/Imhari14)
- Last Updated: 2025-02-27
