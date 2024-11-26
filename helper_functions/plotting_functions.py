import plotly.graph_objects as go

def plot_metrics(rewards_per_episode, episode_numbers):
    """ Plot rewards over episodes using Plotly """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=episode_numbers, y=rewards_per_episode, mode='lines+markers', name='Rewards'))
    fig.update_layout(
        title="Rewards per Episode",
        xaxis_title="Episode Number",
        yaxis_title="Reward",
        template="plotly_dark"
    )
    return fig