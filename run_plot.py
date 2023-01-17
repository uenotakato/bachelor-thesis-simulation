import common.utils as utils
import qlearning as q
import sarsa as s


utils.plot_total_reward_compare(s.acum_reward, q.acum_reward)
