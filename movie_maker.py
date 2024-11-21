import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import moviepy.editor as mpy

import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.defects import theta_cal, calculate_defects, S_cal, plot_defects
from nematic_env import ActiveNematicEnv

file_path = '/home/hou63/pj2/Nematic_RL/log_ucsd/PPO_25/data_dump.h5'
print('model loaded from:', file_path)
f = h5py.File(file_path, 'r')
print(f.keys())
D = f['D']
actions = f['actions']

print(D.shape, actions.shape)
actions = np.array(actions)

new_action = actions[::10]
print(new_action.shape)

D = D[1500:2000, :, :, :]
new_action = new_action[1500:2000]
print('Data step from 1500 to 2000')

print(D.shape, new_action.shape)


def generate_video_with_improved_display(D, actions, theta_cal, calculate_defects, plot_defects, _action2light, output_path="output.mp4"):
    n_step, _, grid_size, _ = D.shape
    assert len(actions) == n_step, "Mismatch between steps in D and actions."

    frames = []  # To store the generated frames
    cmap = "gray"  # Use gray colormap for the action heatmap
    norm = mcolors.Normalize(vmin=0, vmax=1)  # Normalize for heatmap

    for t in range(n_step):
        print(f"Processing frame {t + 1}/{n_step}...")

        # Step 1: Compute theta
        theta = theta_cal(D[t, 0, :, :], D[t, 1, :, :])

        # Step 2: Compute defects
        defects = calculate_defects(theta)

        # Step 3: Compute action heatmap
        action_heatmap = _action2light(self_intensity=1.0, device=None, action=actions[t], grid_size=grid_size)

        # Step 4: Create plot
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.set_axis_off()  # Remove axis for the main plot

        # Plot defects
        plot_defects(defects, theta, ax)

        # Overlay action heatmap in grayscale
        ax.imshow(action_heatmap, cmap=cmap, alpha=0.5, extent=[0, grid_size, 0, grid_size], origin="lower")

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.set_label("Action Intensity", fontsize=10)

        # Add title below plot to display action[5]
        action_text = f"Step: {t + 1}/{n_step}, Action[5]: {actions[t][5]:.2f}"
        ax.set_title(action_text, fontsize=12, pad=10)

        # Save current frame
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

        plt.close(fig)

    # Step 5: Generate video
    print("Generating video...")
    clip = mpy.ImageSequenceClip(frames, fps=10)  # 24 FPS
    clip.write_videofile(output_path, codec="libx264")
    print(f"Video saved to {output_path}")

# Example usage:
# generate_video_with_improved_display(D, actions, theta_cal, calculate_defects, plot_defects, _action2light, "defects_video.mp4")

tic = time.time()
# Example usage:
generate_video_with_improved_display(D, new_action, theta_cal, calculate_defects, plot_defects,
               ActiveNematicEnv._action2light, "/home/hou63/pj2/Nematic_RL/movie/test/defects_video_lights_on1.mp4")

toc = time.time()
print(f"Time taken: {toc - tic:.2f} seconds.")            