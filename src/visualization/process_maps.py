import pm4py
import matplotlib.pyplot as plt
import streamlit as st
import logging
from typing import Dict, Tuple, List, Union
import graphviz

class ProcessMapVisualizer:
    def __init__(self):
        logging.info("Initializing ProcessMapVisualizer module")

    def visualize_process_map(self, net, initial_marking, final_marking):
        """
        Visualize a process map using PM4Py and matplotlib.
        """
        try:
            parameters = {
                "format": "png",
                "bgcolor": "white",
                "rankdir": "LR",
                "ranksep": "0.5",
                "fontsize": "12",
                "nodesep": "0.5"
            }
            
            gviz = pm4py.visualization.petri_net.visualizer.apply(
                net, initial_marking, final_marking,
                parameters=parameters,
                variant=pm4py.visualization.petri_net.visualizer.Variants.FREQUENCY
            )
            
            st.graphviz_chart(gviz)
            
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
        Enhanced implementation with better error handling and multiple fallback options.
        """
        try:
            # Convert dfg to the correct format if needed
            if isinstance(dfg, list):
                dfg_formatted = {(item[0][0], item[0][1]): item[1] for item in dfg}
            elif isinstance(dfg, dict):
                dfg_formatted = {(k[0], k[1]) if isinstance(k, (tuple, list)) else k: v 
                               for k, v in dfg.items()}
            else:
                raise ValueError(f"Unexpected DFG format: {type(dfg)}")
            
            # Ensure start/end activities are dictionaries
            start_acts = (start_activities if isinstance(start_activities, dict) 
                        else {act: 1 for act in start_activities})
            end_acts = (end_activities if isinstance(end_activities, dict)
                       else {act: 1 for act in end_activities})
            
            # Create visualization using graphviz directly
            dot = graphviz.Digraph('DFG')
            dot.attr(rankdir='LR', size='8,5', bgcolor='white')
            
            # Calculate maximum frequency for edge thickness
            max_freq = max(dfg_formatted.values()) if dfg_formatted else 1
            
            # Add nodes
            all_activities = {act for pair in dfg_formatted.keys() for act in pair}
            for act in all_activities:
                node_attr = {
                    'shape': 'box',
                    'style': 'filled',
                    'fontsize': '12'
                }
                
                if act in start_acts:
                    node_attr['fillcolor'] = 'lightblue'
                elif act in end_acts:
                    node_attr['fillcolor'] = 'lightpink'
                else:
                    node_attr['fillcolor'] = 'white'
                    
                dot.node(str(act), str(act), **node_attr)
            
            # Add edges with frequencies
            for (act1, act2), freq in dfg_formatted.items():
                # Calculate edge thickness based on frequency
                thickness = 1 + (3 * freq / max_freq)
                dot.edge(str(act1), str(act2), 
                        label=str(freq),
                        penwidth=str(thickness))
            
            # Display the visualization
            st.graphviz_chart(dot)
            
            # Show additional information
            st.subheader("DFG Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Start Activities:", len(start_acts))
                st.write("End Activities:", len(end_acts))
            with col2:
                st.write("Total Activities:", len(all_activities))
                st.write("Total Connections:", len(dfg_formatted))
            
            # Show activity details
            with st.expander("Activity Details"):
                # Calculate activity frequencies
                activity_freq = {}
                for (act1, act2), freq in dfg_formatted.items():
                    activity_freq[act1] = activity_freq.get(act1, 0) + freq
                    activity_freq[act2] = activity_freq.get(act2, 0) + freq
                
                # Display as a table
                if activity_freq:
                    import pandas as pd
                    activity_df = pd.DataFrame([
                        {"Activity": act, "Frequency": freq}
                        for act, freq in activity_freq.items()
                    ])
                    activity_df = activity_df.sort_values("Frequency", ascending=False)
                    st.dataframe(activity_df)
            
            return dot
        except Exception as e:
            logging.error(f"Error visualizing DFG: {str(e)}")
            st.error(f"Error visualizing DFG: {str(e)}")
            return None
            
    def visualize_bpmn(self, bpmn_model):
        """
        Visualize a BPMN model using PM4Py.
        Updated to handle various BPMN object structures.
        """
        try:
            parameters = {
                "format": "png",
                "bgcolor": "white",
                "rankdir": "LR"
            }
            
            gviz = pm4py.visualization.bpmn.visualizer.apply(bpmn_model, parameters=parameters)
            
            st.graphviz_chart(gviz)
            
            try:
                if hasattr(bpmn_model, 'get_nodes'):
                    nodes = bpmn_model.get_nodes()
                    flows = bpmn_model.get_flows()
                else:
                    nodes = [node for node in bpmn_model.get_nodes()]
                    flows = [flow for flow in bpmn_model.get_flows()]
                
                tasks = []
                gateways = []
                events = []
                
                for node in nodes:
                    try:
                        node_type = (node.get_type() if hasattr(node, 'get_type') else
                                   node.get_gateway_type() if hasattr(node, 'get_gateway_type') else
                                   str(type(node).__name__).lower())
                                   
                        if 'task' in node_type.lower():
                            tasks.append(node)
                        elif 'gateway' in node_type.lower():
                            gateways.append(node)
                        elif 'event' in node_type.lower():
                            events.append(node)
                    except Exception as type_error:
                        logging.warning(f"Could not determine type for node: {type_error}")
                
                st.subheader("BPMN Model Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Tasks:", len(tasks))
                    st.write("Gateways:", len(gateways))
                with col2:
                    st.write("Events:", len(events))
                    st.write("Flows:", len(flows))
                    
                with st.expander("Task Details"):
                    for task in tasks:
                        try:
                            task_name = task.get_name() if hasattr(task, 'get_name') else str(task)
                            st.write(f"Task: {task_name}")
                        except Exception as task_error:
                            logging.warning(f"Could not get task name: {task_error}")
            
            except Exception as info_error:
                logging.warning(f"Could not display additional BPMN information: {info_error}")
            
            return gviz
        except Exception as e:
            st.error(f"Error visualizing BPMN: {e}")
            return None
