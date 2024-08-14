import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from textblob import TextBlob
import networkx as nx
import numpy as np

def load_and_preprocess_data(json_file):
    # Load the JSON file and preprocess the data
    with open(json_file, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame(data)
    df_exploded = df.explode('fallacy_type')

    # Filter out rows where fallacy_type is "None"
    df_filtered = df_exploded[df_exploded['fallacy_type'] != "None"]

    return df_filtered

def generate_word_cloud_image(df_filtered, output_path="wordcloud.png"):
    # Generate a word cloud from text segments
    text = " ".join(segment for segment in df_filtered['text_segment'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Save the WordCloud as a PNG image
    wordcloud.to_file(output_path)
    print(f"Word cloud saved to {output_path}")

def add_word_cloud_to_figure(fig, image_base64):
    # Add Word Cloud image to the figure as a separate subplot
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{image_base64}",
            xref="x domain", yref="y domain",
            x=0, y=1,
            sizex=1, sizey=1,
            xanchor="left", yanchor="top"
        ),
        row=7, col=1
    )

def perform_sentiment_analysis(df_filtered):
    # Perform sentiment analysis on text segments
    df_filtered['sentiment'] = df_filtered['text_segment'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df_filtered

def create_correlation_matrix(df_filtered):
    # Create a correlation matrix of speakers and fallacies
    speaker_fallacy_matrix = pd.crosstab(df_filtered['speaker'], df_filtered['fallacy_type'])
    return speaker_fallacy_matrix

def create_network_graph(df_filtered):
    # Create a network graph of speakers based on shared fallacy types
    G = nx.Graph()
    for speaker in df_filtered['speaker'].unique():
        G.add_node(speaker)

    for fallacy_type in df_filtered['fallacy_type'].unique():
        speakers = df_filtered[df_filtered['fallacy_type'] == fallacy_type]['speaker'].unique()
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                if G.has_edge(speakers[i], speakers[j]):
                    G[speakers[i]][speakers[j]]['weight'] += 1
                else:
                    G.add_edge(speakers[i], speakers[j], weight=1)
    
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Number of Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{node}: {len(adjacencies[1])} connections')
    
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    
    return edge_trace, node_trace

def create_plots(df_filtered):
    # Additional analysis
    df_filtered = perform_sentiment_analysis(df_filtered)
    fallacy_count = df_filtered['fallacy_type'].value_counts()
    speaker_fallacy_count = df_filtered.groupby(['speaker', 'fallacy_type']).size().unstack(fill_value=0)
    avg_fallacy_duration = df_filtered.groupby('fallacy_type')['end'].mean() - df_filtered.groupby('fallacy_type')['start'].mean()
    top_speakers = df_filtered['speaker'].value_counts().head(10)
    speaker_fallacy_matrix = create_correlation_matrix(df_filtered)

    # Create a combined plot with multiple subplots, specify the type for each subplot
    fig = make_subplots(
        rows=6, cols=1,  # Stack all subplots vertically
        specs=[[{"type": "scatter"}],
               [{"type": "table"}],
               [{"type": "bar"}],
               [{"type": "heatmap"}],
               [{"type": "bar"}],
               [{"type": "scatter"}]],
        subplot_titles=(
            "Fallacy Distribution Over Time", 
            "Example Segment",
            "Fallacy Type Frequency",
            "Fallacies by Speaker",
            "Average Duration of Fallacies",
            "Sentiment Analysis Over Time"
        ),
        vertical_spacing=0.03  # Decrease vertical spacing
    )

    # 1. Scatter plot: Fallacy Distribution Over Time
    fig.add_trace(go.Scatter(x=df_filtered['start'], y=df_filtered['fallacy_type'], mode='markers', 
                             text=df_filtered['text_segment'], name='Fallacy Type'), row=1, col=1)

    # 2. Table: Example Segment
    fig.add_trace(go.Table(header=dict(values=["Speaker", "Text Segment", "Fallacy Type"]),
                           cells=dict(values=[df_filtered['speaker'], df_filtered['text_segment'], df_filtered['fallacy_type']])), row=2, col=1)
    
    # 3. Bar chart: Fallacy Type Frequency
    fig.add_trace(go.Bar(x=fallacy_count.index, y=fallacy_count.values, name="Fallacy Type Frequency"), row=3, col=1)
    
    # 4. Heatmap: Fallacies by Speaker
    fig.add_trace(go.Heatmap(z=speaker_fallacy_count.values, 
                         x=speaker_fallacy_count.columns, 
                         y=speaker_fallacy_count.index, 
                         colorscale='Viridis', 
                         colorbar=dict(title="Count", len=0.5, x=1.05, y=0.5),  # Adjusted color bar position
                         name="Fallacies by Speaker"), row=4, col=1)
    
    # 5. Bar chart: Average Duration of Fallacies
    fig.add_trace(go.Bar(x=avg_fallacy_duration.index, y=avg_fallacy_duration.values, name="Avg. Fallacy Duration (s)"), row=5, col=1)

    # 6. Sentiment Analysis Over Time
    fig.add_trace(go.Scatter(x=df_filtered['start'], y=df_filtered['sentiment'], mode='lines+markers', 
                             text=df_filtered['text_segment'], name='Sentiment Over Time'), row=6, col=1)

    # Adjust layout
    fig.update_layout(height=2400)  # Increase height to accommodate all plots

    return fig


def save_dashboard(fig, dashboard_file, wordcloud_image_path="wordcloud.png"):
    # Save the Plotly figure to an HTML file
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Embed the word cloud image in the HTML and center it
    wordcloud_html = f'''
    <div style="text-align:center; margin-top:20px;">
        <h3>Word Cloud of Text Segments</h3>
        <img src="{wordcloud_image_path}" alt="Word Cloud" style="width:100%; max-width:800px;">
    </div>
    '''

    # Combine the word cloud and Plotly dashboard HTML
    full_html = f"""
    <html>
    <head>
        <title>Comprehensive Fallacy Analysis Dashboard</title>
    </head>
    <body>
        <h1>Comprehensive Fallacy Analysis Dashboard</h1>
        {fig_html}
        {wordcloud_html}
    </body>
    </html>
    """

    # Write the full HTML to a file
    with open(dashboard_file, 'w') as f:
        f.write(full_html)
    
    print(f"HTML dashboard generated: {dashboard_file}")


def create_dashboard(json_file, dashboard_file):
    df_filtered = load_and_preprocess_data(json_file)

    # Generate word cloud image
    wordcloud_image_path = dashboard_file[:-4] + ".png"
    generate_word_cloud_image(df_filtered, wordcloud_image_path)

    # Create the Plotly figure
    fig = create_plots(df_filtered)

    # Save the dashboard with the word cloud image
    save_dashboard(fig, dashboard_file, wordcloud_image_path)

# Example usage
# create_dashboard("fallacy_data.json", "fallacy_analysis_dashboard.html")
