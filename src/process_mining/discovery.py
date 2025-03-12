import pm4py
import logging
from typing import Dict, Tuple, List, Union, Set
from collections import defaultdict

class ProcessDiscovery:
    def __init__(self):
        logging.info("Initializing ProcessDiscovery module")
    
    def discover_process_map(self, event_log):
        """
        Discover a process map from the event log using the Alpha algorithm.
        """
        try:
            net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(event_log)
            return net, initial_marking, final_marking
        except Exception as e:
            logging.error(f"Error in process map discovery: {str(e)}")
            raise ValueError(f"Error in process map discovery: {str(e)}")

    def discover_bpmn_model(self, event_log):
        """
        Discover a BPMN model from the event log using the Inductive Miner.
        """
        try:
            bpmn_model = pm4py.discover_bpmn_inductive(event_log)
            
            if bpmn_model is None:
                raise ValueError("Empty BPMN model generated")
            
            try:
                _ = [node for node in bpmn_model.get_nodes()]
            except Exception as node_error:
                logging.warning(f"BPMN model structure validation failed: {node_error}")
                raise ValueError("Invalid BPMN model structure")
                
            return bpmn_model
        except Exception as e:
            logging.error(f"Error in primary BPMN discovery method: {str(e)}")
            
            try:
                logging.info("Trying alternative BPMN discovery method")
                net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log)
                bpmn_model = pm4py.convert_to_bpmn(net, initial_marking, final_marking)
                return bpmn_model
            except Exception as e2:
                logging.error(f"Error in fallback BPMN discovery: {str(e2)}")
                raise ValueError(f"Error in BPMN discovery: {str(e)} | Fallback error: {str(e2)}")

    def _convert_dfg_to_dict(self, dfg_data: Union[Dict, List, Tuple]) -> Dict:
        """
        Convert various DFG formats to a standardized dictionary format.
        """
        if isinstance(dfg_data, dict):
            return {(k[0], k[1]) if isinstance(k, (list, tuple)) else k: v 
                   for k, v in dfg_data.items()}
        elif isinstance(dfg_data, list):
            return {(item[0][0], item[0][1]): item[1] for item in dfg_data}
        elif isinstance(dfg_data, tuple):
            # Handle PM4Py's tuple format (dfg, start_activities, end_activities)
            dfg_part = dfg_data[0] if len(dfg_data) > 0 else {}
            if isinstance(dfg_part, dict):
                return {(k[0], k[1]) if isinstance(k, (list, tuple)) else k: v 
                       for k, v in dfg_part.items()}
            elif isinstance(dfg_part, list):
                return {(item[0][0], item[0][1]): item[1] for item in dfg_part}
        
        raise ValueError(f"Unsupported DFG format: {type(dfg_data)}")

    def discover_dfg(self, event_log):
        """
        Discover a Directly-Follows Graph (DFG) from the event log.
        Enhanced implementation to handle different PM4Py versions and formats.
        """
        try:
            # Get activities for validation
            activities = pm4py.get_event_attribute_values(event_log, "concept:name")
            
            # Try different DFG discovery methods
            try:
                dfg_result = pm4py.discover_directly_follows_graph(event_log)
                
                # Handle different return formats
                if isinstance(dfg_result, tuple):
                    # Newer PM4Py versions might return (dfg, start_activities, end_activities)
                    dfg_dict = self._convert_dfg_to_dict(dfg_result)
                    start_activities = dfg_result[1] if len(dfg_result) > 1 else None
                    end_activities = dfg_result[2] if len(dfg_result) > 2 else None
                else:
                    # Single DFG result
                    dfg_dict = self._convert_dfg_to_dict(dfg_result)
                    start_activities = None
                    end_activities = None
                
            except Exception as e1:
                logging.warning(f"Primary DFG discovery failed: {e1}")
                try:
                    from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
                    dfg_result = dfg_discovery.apply(event_log)
                    dfg_dict = self._convert_dfg_to_dict(dfg_result)
                    start_activities = None
                    end_activities = None
                except Exception as e2:
                    logging.warning(f"Alternative DFG discovery failed: {e2}")
                    # Create minimum viable DFG
                    df = pm4py.convert_to_dataframe(event_log)
                    df = df.sort_values(['case:concept:name', 'time:timestamp'])
                    
                    dfg = defaultdict(int)
                    for case in df['case:concept:name'].unique():
                        case_events = df[df['case:concept:name'] == case]['concept:name'].tolist()
                        for i in range(len(case_events)-1):
                            dfg[(case_events[i], case_events[i+1])] += 1
                    dfg_dict = dict(dfg)
            
            # Get or infer start/end activities if not provided
            if start_activities is None:
                try:
                    start_activities = pm4py.get_start_activities(event_log)
                except Exception:
                    # Infer from DFG
                    all_activities = {act for pair in dfg_dict.keys() for act in pair}
                    incoming = {act2 for (act1, act2) in dfg_dict.keys()}
                    start_activities = {act: 1 for act in all_activities if act not in incoming}
                    if not start_activities:
                        start_activities = {list(all_activities)[0]: 1} if all_activities else {"start": 1}

            if end_activities is None:
                try:
                    end_activities = pm4py.get_end_activities(event_log)
                except Exception:
                    # Infer from DFG
                    all_activities = {act for pair in dfg_dict.keys() for act in pair}
                    outgoing = {act1 for (act1, act2) in dfg_dict.keys()}
                    end_activities = {act: 1 for act in all_activities if act not in outgoing}
                    if not end_activities:
                        end_activities = {list(all_activities)[-1]: 1} if all_activities else {"end": 1}

            # Ensure activities are in dictionary format
            if not isinstance(start_activities, dict):
                start_activities = {act: 1 for act in start_activities}
            if not isinstance(end_activities, dict):
                end_activities = {act: 1 for act in end_activities}
            
            return dfg_dict, start_activities, end_activities
            
        except Exception as e:
            logging.error(f"Error in DFG discovery: {str(e)}")
            raise ValueError(f"Error in DFG discovery: {str(e)}")
