from babyai.levels.test_levels import Level_TestGoToBlocked
from babyai.levels.bonus_levels import Level_UnlockLocalDist
from babyai.levels.iclr19_levels import Level_BossLevelNoUnlock
from gym_minigrid import rendering
from matplotlib import pyplot as plt
import pdb
import numpy as np

env = Level_UnlockLocalDist()
env.reset()
for i in range(10):
    env.step(1)
    obs = env.step(2)[0]
    # print(obs)
    print(str(env))
# rendering.
# obs["image"][0][0][0] = 200
# plt.imshow(obs["image"])
# plt.show()
# pdb.set_trace()
x = env.render("rgb-array", tile_size=32)
plt.imshow(x)
print(np.average(x))
plt.savefig("test.png")
