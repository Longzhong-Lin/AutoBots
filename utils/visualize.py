import uuid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


####################################
#          Tool Functions
####################################

def create_figure_and_axes(size_pixels):
    """Initializes a unique figure and axes for plotting."""
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

    # Sets output image to pixel resolution.
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    fig.set_tight_layout(True)
    ax.grid(False)

    return fig, ax


def fig_canvas_image(fig):
    """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
    # Just enough margin in the figure to display xticks and yticks.
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0
    )
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_colormap(num_agents, seed=0):
    """Compute a color map array of shape [num_agents, 4]."""
    colors = plt.cm.get_cmap('jet', num_agents)
    colors = colors(range(num_agents))
    np.random.seed(seed)
    np.random.shuffle(colors)
    return colors


def get_viewport(all_states, all_states_mask):
    """Gets the region containing the data.

    Args:
        all_states: states of agents as an array of shape [num_agents, num_steps, 2]
        all_states_mask: binary mask of shape [num_agents, num_steps] for `all_states`

    Returns:
        center_y: float. y coordinate for center of data
        center_x: float. x coordinate for center of data
        width: float. Width of data
    """
    valid_states = all_states[all_states_mask]
    all_y = valid_states[..., 1]
    all_x = valid_states[..., 0]

    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)

    width = max(range_y, range_x)

    return center_y, center_x, width


def visualize_one_step(
    states, mask, roadgraph, title,
    center_y, center_x, width, color_map,
    size_pixels=1000
):
    """Generate visualization for a single step."""

    # Create figure and axes.
    fig, ax = create_figure_and_axes(size_pixels=size_pixels)

    # Plot roadgraph.
    rg_pts = roadgraph[:, :2].T
    ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=1, ms=2)

    masked_x = states[:, 0][mask]
    masked_y = states[:, 1][mask]
    colors = color_map[mask]

    # Plot agent current position.
    ax.scatter(
        masked_x, masked_y,
        marker='o', linewidths=3, color=colors,
    )

    # Title.
    ax.set_title(title)

    # Set axes.  Should be at least 10m on a side.
    size = max(10, width * 1.0)
    ax.axis([
        -size / 2 + center_x, size / 2 + center_x,
        -size / 2 + center_y, size / 2 + center_y
    ])
    ax.set_aspect('equal')

    image = fig_canvas_image(fig)
    plt.close(fig)

    return image


####################################
#         Visualization API
####################################

def visualize_all_agents_steps(
    agent_history: np.ndarray,
    agent_future: np.ndarray,
    roadgraph: np.ndarray,
    size_pixels=1000
):
    """Visualizes all agent predicted trajectories in a serie of images.

    Args:
    agent_history: [num_agents, num_history_steps, (valid,x,y)]
    agent_future: [num_agents, num_future_steps, (valid,x,y)]
    roadgraph: [num_points, (valid,x,y)]
    size_pixels: The size in pixels of the output image

    Returns:
    images: T of [H, W, 3] uint8 np.arrays of the drawn matplotlib's figure canvas
    """
    # prepare data
    history_mask = agent_history[..., 0] > 0.0 # [num_agents, num_history_steps]
    history_states = agent_history[..., 1:3] # [num_agents, num_history_steps, 2]

    future_mask = agent_future[..., 0] > 0.0 # [num_agents, num_future_steps]
    future_states = agent_future[..., 1:3] # [num_agents, num_future_steps, 2]

    roadgraph_mask = roadgraph[:, 0] > 0.0 # [num_points]
    roadgraph_xy = roadgraph[:, 1:3] # [num_points, 2]
    roadgraph_xy_valid = roadgraph_xy[roadgraph_mask] # [num_valid_points, 2]

    # get dimensions
    num_agents, num_history_steps, _ = history_states.shape
    num_future_steps = future_states.shape[1]

    # get colormap for agents
    color_map = get_colormap(num_agents)

    # compute viewport
    all_states = np.concatenate([history_states, future_states], axis=1) # [num_agents, num_steps, 2]
    all_states_mask = np.concatenate([history_mask,  future_mask], axis=1) # [num_agents, num_steps]
    center_y, center_x, width = get_viewport(all_states, all_states_mask)

    # generate images for each step
    images = []
    for i, (s, m) in enumerate(
        zip(
            np.split(history_states, num_history_steps, 1),
            np.split(history_mask, num_history_steps, 1)
        )
    ):
        im = visualize_one_step(
            s[:, 0], m[:, 0], roadgraph_xy_valid,
            '-%d' % (num_history_steps - 1 - i),
            center_y, center_x, width, color_map, size_pixels
        )
        images.append(im)

    for i, (s, m) in enumerate(
        zip(
            np.split(future_states, num_future_steps, 1),
            np.split(future_mask, num_future_steps, 1)
        )
    ):
        im = visualize_one_step(
            s[:, 0], m[:, 0], roadgraph_xy_valid,
            '%d' % (i + 1),
            center_y, center_x, width, color_map, size_pixels
        )
        images.append(im)

    return images


def create_animation(images, interval=100):
    """ Creates a Matplotlib animation of the given images.

    Args:
        images: A list of numpy arrays representing the images.
        interval: Delay between frames in milliseconds.

    Returns:
        A matplotlib.animation.Animation.

    Usage:
        anim = create_animation(images)
        anim.save('/tmp/animation.avi')
        HTML(anim.to_html5_video())
    """

    plt.ioff()
    fig, ax = plt.subplots()
    dpi = 100
    size_inches = 1000 / dpi
    fig.set_size_inches([size_inches, size_inches])
    plt.ion()

    def animate_func(i):
        ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid('off')

    anim = animation.FuncAnimation(fig, animate_func, frames=len(images), interval=interval)
    plt.close(fig)
    return anim


def visualize_all_agents_traj(
    agent_history: np.ndarray,
    agent_future: np.ndarray,
    roadgraph: np.ndarray,
    topic: str,
    center_x=0.0,
    center_y=0.0,
    width=200.0,
    size_pixels=1000
):
    """Visualizes all agent predicted trajectories in a image.

    Args:
    agent_history: [num_agents, num_history_steps, (valid,x,y)]
    agent_future: [num_agents, num_future_steps, (valid,x,y)]
    roadgraph: [num_points, (valid,x,y)]
    size_pixels: The size in pixels of the output image

    Returns:
    image: a [H, W, 3] uint8 np.arrays of trajectory image
    """
    ## prepare data
    history_mask = agent_history[..., 0] > 0.0 # [num_agents, num_history_steps]
    history_states = agent_history[..., 1:3] # [num_agents, num_history_steps, 2]

    future_mask = agent_future[..., 0] > 0.0 # [num_agents, num_future_steps]
    future_states = agent_future[..., 1:3] # [num_agents, num_future_steps, 2]

    # all_states_mask = np.concatenate([history_mask,  future_mask], axis=1) # [num_agents, num_steps]
    # all_states = np.concatenate([history_states, future_states], axis=1) # [num_agents, num_steps, 2]
    
    roadgraph_mask = roadgraph[:, 0] > 0.0 # [num_points]
    roadgraph_xy = roadgraph[:, 1:3] # [num_points, 2]
    roadgraph_xy_valid = roadgraph_xy[roadgraph_mask] # [num_valid_points, 2]

    ## get dimensions
    num_agents, num_history_steps, _ = history_states.shape
    num_future_steps = future_states.shape[1]

    ## get colormap for agents
    color_map = get_colormap(num_agents)

    ## generate trajectory image
    # Create figure and axes.
    fig, ax = create_figure_and_axes(size_pixels=size_pixels)
    # Plot roadgraph.
    rg_pts = roadgraph_xy_valid[:, :2].T
    ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=1, ms=2)
    # Plot agents.
    for t in range(num_history_steps):
        masked_x = history_states[:, t, 0][history_mask[:, t]]
        masked_y = history_states[:, t, 1][history_mask[:, t]]
        colors = color_map[history_mask[:, t]]
        ax.scatter(
            masked_x, masked_y,
            marker='.', linewidths=3, color=colors,
            alpha=1-t/(num_history_steps+num_future_steps)/1.2
        )
    for t in range(num_future_steps):
        masked_x = future_states[:, t, 0][future_mask[:, t]]
        masked_y = future_states[:, t, 1][future_mask[:, t]]
        colors = color_map[future_mask[:, t]]
        ax.scatter(
            masked_x, masked_y,
            marker='x', linewidths=3, color=colors,
            alpha=1-(t+num_history_steps)/(num_history_steps+num_future_steps)/1.2
        )
    # Title.
    ax.set_title(topic)
    # Set axes.  Should be at least 10m on a side.
    size = max(10, width * 1.0)
    ax.axis([
        -size / 2 + center_x, size / 2 + center_x,
        -size / 2 + center_y, size / 2 + center_y
    ])
    ax.set_aspect('equal')
    image = fig_canvas_image(fig)
    plt.close(fig)

    return image

