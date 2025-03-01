import pm4py
import matplotlib.pyplot as plt
import streamlit as st
import logging

class ProcessMapVisualizer:
    def __init__(self):
        logging.info("Initializing ProcessMapVisualizer module")

    def visualize_process_map(self, net, initial_marking, final_marking):
        """
        Visualize a process map using PM4Py and matplotlib.
        """
        try:
            # Set visualization parameters
            parameters = {
                "format": "png",
                "bgcolor": "white",
                "rankdir": "LR",  # Left to right layout
                "ranksep": "0.5",  # Space between ranks
                "fontsize": "12",  # Font size
                "nodesep": "0.5"   # Space between nodes
            }
            
            # Create Petri net visualization
            gviz = pm4py.visualization.petri_net.visualizer.apply(
                net, initial_marking, final_marking,
                parameters=parameters,
                variant=pm4py.visualization.petri_net.visualizer.Variants.FREQUENCY
            )
            
            # Display the visualization
            st.graphviz_chart(gviz)
            
            # Show additional information
            st.subheader("Process Model Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Places:", len(net.places))
                st.write("Transitions:", len(net.transitions))
            with col2:
                st.write("Arcs:", len(net.arcs))
                st.write("Initial Places:", len(initial_marking))
            
            return gviz
        except Exception as e:
            st.error(f"Error visualizing process map: {e}")
            return None

    def visualize_dfg(self, dfg, start_activities, end_activities):
        """
        Visualize a Directly-Follows Graph (DFG) using PM4Py and matplotlib.
        Enhanced to handle different formats and ensure proper visualization.
        """
        try:
            # Ensure dfg is in the correct format
            if isinstance(dfg, dict):
                # Check if it's already in the format (tuple, tuple): value
                if any(isinstance(k, tuple) for k in dfg.keys()):
                    # It's already in the right format
                    dfg_formatted = dfg
                else:
                    # Not sure of the format, try to force it
                    try:
                        dfg_formatted = {(k[0], k[1]): v for k, v in dfg.items()}
                    except Exception:
                        # Fallback
                        logging.warning("DFG format conversion failed, using as-is")
                        dfg_formatted = dfg
            else:
                # Not a dict, try to convert from list format if possible
                try:
                    dfg_formatted = {item[0]: item[1] for item in dfg}
                except Exception:
                    logging.warning("Unable to convert DFG to proper format")
                    dfg_formatted = dfg
            
            # Convert start/end activities to list if they're dictionaries
            start_acts = list(start_activities.keys()) if isinstance(start_activities, dict) else list(start_activities)
            end_acts = list(end_activities.keys()) if isinstance(end_activities, dict) else list(end_activities)
            
            # Create DFG visualization with enhanced parameters
            parameters = {
                "format": "png",
                "bgcolor": "white",
                "rankdir": "LR",
                "fontsize": "12",
                "ratio": "0.5",
                "min_edge_length": "2",
                "edge_penwidth": "1.5",
                "start_activities": start_acts,
                "end_activities": end_acts,
            }
            
            try:
                # Try using the recommended visualizer method first
                gviz = pm4py.visualization.dfg.visualizer.apply(
                    dfg_formatted,
                    log=None,
                    parameters=parameters
                )
            except Exception as e:
                logging.warning(f"Primary DFG visualization failed: {e}")
                # Fallback to direct variant call
                try:
                    from pm4py.visualization.dfg.variants import frequency
                    gviz = frequency.apply(
                        dfg_formatted,
                        parameters=parameters
                    )
                except Exception as e2:
                    logging.error(f"Fallback DFG visualization also failed: {e2}")
                    st.error("Could not visualize DFG due to formatting issues.")
                    # Create a minimal valid graphviz object as fallback
                    import graphviz
                    gviz = graphviz.Digraph('DFG')
                    gviz.attr(rankdir='LR', size='8,5', bgcolor='white')
                    
                    # Add basic nodes from start/end activities at least
                    for act in start_acts:
                        gviz.node(act, shape='box', style='filled', fillcolor='lightblue')
                    for act in end_acts:
                        gviz.node(act, shape='box', style='filled', fillcolor='lightpink')
            
            # Display the visualization
            st.graphviz_chart(gviz)
            
            # Show additional information
            st.subheader("DFG Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Start Activities:", len(start_activities))
                st.write("End Activities:", len(end_activities))
            with col2:
                # Calculate total activities more safely
                all_activities = set()
                try:
                    for k in dfg_formatted.keys():
                        if isinstance(k, tuple) and len(k) == 2:
                            all_activities.add(k[0])
                            all_activities.add(k[1])
                except Exception:
                    pass
                
                st.write("Total Activities:", len(all_activities) if all_activities else "N/A")
                st.write("Total Connections:", len(dfg_formatted) if isinstance(dfg_formatted, dict) else "N/A")
            
            # Show activity frequency details
            with st.expander("Activity Details"):
                # Create a frequency counter for activities
                activity_freq = {}
                try:
                    for (act1, act2), freq in dfg_formatted.items():
                        activity_freq[act1] = activity_freq.get(act1, 0) + freq
                        activity_freq[act2] = activity_freq.get(act2, 0) + freq
                    
                    # Display as a small table
                    if activity_freq:
                        import pandas as pd
                        activity_df = pd.DataFrame([
                            {"Activity": act, "Frequency": freq}
                            for act, freq in activity_freq.items()
                        ])
                        activity_df = activity_df.sort_values("Frequency", ascending=False)
                        st.dataframe(activity_df)
                except Exception as e:
                    st.write("Could not calculate activity frequencies")
            
            return gviz
        except Exception as e:
            logging.error(f"Error visualizing DFG: {str(e)}")
            st.error(f"Error visualizing DFG: {str(e)}")
            return None
            
    def visualize_bpmn(self, bpmn_model):
        """
        Visualize a BPMN model using PM4Py.
        """
        try:
            # Set visualization parameters
            parameters = {
                "format": "png",
                "bgcolor": "white",
                "rankdir": "LR"  # Left to right layout
            }
            
            # Create BPMN visualization
            gviz = pm4py.visualization.bpmn.visualizer.apply(bpmn_model, parameters=parameters)
            
            # Display the visualization
            st.graphviz_chart(gviz)
            
            # Show additional information if available
            if hasattr(bpmn_model, 'get_nodes') and hasattr(bpmn_model, 'get_flows'):
                tasks = [node for node in bpmn_model.get_nodes() if node.get_type() == "task"]
                gateways = [node for node in bpmn_model.get_nodes() if "gateway" in node.get_type()]
                events = [node for node in bpmn_model.get_nodes() if "event" in node.get_type()]
                
                st.subheader("BPMN Model Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Tasks:", len(tasks))
                    st.write("Gateways:", len(gateways))
                with col2:
                    st.write("Events:", len(events))
                    st.write("Flows:", len(bpmn_model.get_flows()))
                    
                # Show detailed information about tasks
                with st.expander("Task Details"):
                    for task in tasks:
                        st.write(f"Task: {task.get_name()}")
            
            return gviz
        except Exception as e:
            st.error(f"Error visualizing BPMN: {e}")
            return None