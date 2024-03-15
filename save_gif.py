import gif
import plotly.graph_objects as go
import imageio
import pathlib
import matplotlib

import seaborn as sns

def save_df_as_image(df, path):
    # Set background to white
    norm = matplotlib.colors.Normalize(-1,1)
    colors = [[norm(-1.0), "white"],
            [norm( 1.0), "white"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    # Make plot
    plot = sns.heatmap(df, annot=True, cmap=cmap, cbar=False)
    fig = plot.get_figure()
    fig.savefig(path)


def animation_to_gif(fig, filename, frame_duration=100, width=1200, height=800):
    @gif.frame
    def plot(f, i):
        f_ = go.Figure(data=f["frames"][i]["data"], layout=f["layout"])
        f_["layout"]["updatemenus"] = []
        f_.update_layout(title=f["frames"][i]["layout"]["title"], width=width, height=height)
        return f_

    frames = [plot(fig, i) for i in range(len(fig["frames"]))]
    gif.save(frames, filename, duration=frame_duration)


def frames_to_gif(frames, filename):
    for i, frame in enumerate(frames):
        fig = go.Figure(data=frame["data"], layout=frame["layout"])
        fig.write_image(f"{filename.split('.')[0]}_{i}.png")
    images = []
    for i in range(len(frames)):
        name = f"{filename.split('.')[0]}_{i}.png"
        images.append(imageio.v2.imread(name))
        pathlib.Path(name).unlink()
    imageio.mimsave(filename, images, duration=2)
