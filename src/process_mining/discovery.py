import pm4py
import logging

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
        Fixed implementation to handle potential errors.
        """
        try:
            # Try the default inductive miner first
            bpmn_model = pm4py.discover_bpmn_inductive(event_log)
            
            # Verify the model is valid
            if bpmn_model is None or not hasattr(bpmn_model, 'get_nodes'):
                raise ValueError("Empty or invalid BPMN model generated")
                
            return bpmn_model
        except Exception as e:
            logging.error(f"Error in primary BPMN discovery method: {str(e)}")
            
            try:
                # Fallback to alternative discovery method
                logging.info("Trying alternative BPMN discovery method")
                
                # First discover a Petri net
                net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log)
                
                # Convert Petri net to BPMN
                bpmn_model = pm4py.convert_to_bpmn(net, initial_marking, final_marking)
                
                return bpmn_model
            except Exception as e2:
                logging.error(f"Error in fallback BPMN discovery: {str(e2)}")
                raise ValueError(f"Error in BPMN discovery: {str(e)} | Fallback error: {str(e2)}")

    def discover_dfg(self, event_log):
        """
        Discover a Directly-Follows Graph (DFG) from the event log.
        Fixed implementation to ensure consistent output format.
        """
        try:
            # Get start and end activities first
            start_activities = pm4py.get_start_activities(event_log)
            end_activities = pm4py.get_end_activities(event_log)
            
            # Get the DFG with frequency
            dfg = pm4py.discover_directly_follows_graph(event_log)
            
            # Ensure dfg is in the correct format - should be a dictionary
            if not isinstance(dfg, dict):
                # If it's a list of tuples, convert to dict
                if isinstance(dfg, list):
                    dfg_dict = {item[0]: item[1] for item in dfg}
                else:
                    # Try different conversion method
                    dfg_dict = {(k[0], k[1]): v for k, v in dfg.items()}
            else:
                dfg_dict = dfg
            
            # Validate start/end activities
            if not start_activities:
                logging.warning("No start activities found, using first activity")
                # Use the first activity in the dfg as start
                all_activities = set()
                for (act1, act2) in dfg_dict.keys():
                    all_activities.add(act1)
                    all_activities.add(act2)
                start_activities = {list(all_activities)[0]: 1} if all_activities else {"start": 1}
                
            if not end_activities:
                logging.warning("No end activities found, using last activity")
                # Use the last activity in the dfg as end
                all_activities = set()
                for (act1, act2) in dfg_dict.keys():
                    all_activities.add(act1)
                    all_activities.add(act2)
                end_activities = {list(all_activities)[-1]: 1} if all_activities else {"end": 1}
            
            return dfg_dict, start_activities, end_activities
        except Exception as e:
            logging.error(f"Error in DFG discovery: {str(e)}")
            raise ValueError(f"Error in DFG discovery: {str(e)}")