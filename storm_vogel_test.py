import gymnasium as gym
from stormvogel.extensions.gym_grid import *
from stormvogel import *
import IPython.display as ipd


env = gym.make("Taxi-v3", render_mode="rgb_array")  # Set `is_slippery=True` for stochastic behavior
sv_model = gymnasium_grid_to_stormvogel(env)
# This model is so big that it is better not to display it.
sv_model.summary()

target = get_target_state(env)
res = model_checking(sv_model, f'Rmax=? [S]')
gs = to_gymnasium_scheduler(sv_model, res.scheduler, GRID_ACTION_LABEL_MAP)
filename = gymnasium_render_model_gif(env, gs, filename="taxi")
extensions.embed_gif(filename)