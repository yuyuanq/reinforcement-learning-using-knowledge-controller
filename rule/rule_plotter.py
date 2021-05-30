import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from rule_cartpole import CartPoleRule
from rule_lunarlandercontinuous import LunarLanderContinuousRule
from rule_flappybird import FlappyBirdRule

sns.set_style("dark")
sns.despine(left=True)
plt.rcParams["font.family"] = "Times New Roman"

if __name__ == '__main__':
    # for cp
    plot_func_list = [[CartPoleRule.s0_ne, CartPoleRule.s0_po],
                      [CartPoleRule.s1_ne, CartPoleRule.s1_po],
                      [CartPoleRule.s2_ne, CartPoleRule.s2_po, CartPoleRule.s2_sm],
                      [CartPoleRule.s3_ne, CartPoleRule.s3_po, CartPoleRule.s3_sm]]
    range_list = [[-2, 2], [-2, 2], [-2, 2], [-2, 2]]
    legend_list = [['NE', 'PO'], ['NE', 'PO'], ['NE', 'PO', 'SM'], ['NE', 'PO', 'SM']]
    x_list = ['CartPosition', 'CartVelocity', 'PoleAngle', 'PoleVelocity']

    # for ll
    # plot_func_list = [[LunarLanderContinuousRule.s0_ne, LunarLanderContinuousRule.s0_po],
    #                   [LunarLanderContinuousRule.s1_sm],
    #                   [LunarLanderContinuousRule.s2_ne, LunarLanderContinuousRule.s2_po],
    #                   [LunarLanderContinuousRule.s3_la],
    #                   [LunarLanderContinuousRule.s4_ne, LunarLanderContinuousRule.s4_po, LunarLanderContinuousRule.s4_sm],
    #                   [LunarLanderContinuousRule.s5_ne, LunarLanderContinuousRule.s5_po]]
    # range_list = [[-0.1, 0.1], [-0.2, 1], [-1, 1], [-1, 0], [-1, 1], [-2, 2]]
    # legend_list = [['NE', 'PO'], ['SM'], ['NE', 'PO'], ['LA'], ['NE', 'PO', 'SM'], ['NE', 'PO']]
    # x_list = ['PositionX', 'PositionY', 'VelocityX', 'VelocityY', 'Angle', 'AngularVelocity']

    # for fb
    # plot_func_list = [[FlappyBirdRule.s0_sm, FlappyBirdRule.s0_la],
    #                   [FlappyBirdRule.s1_ne, FlappyBirdRule.s1_po],
    #                   [FlappyBirdRule.s3_ne],
    #                   [FlappyBirdRule.s4_po]]
    # range_list = [[100, 300], [-12, 12], [-60, 30], [-30, 60]]
    # legend_list = [['SM', 'LA'], ['NE', 'PO'], ['NE'], ['PO']]
    # x_list = ['Position', 'Velocity', 'Position - PositionT', 'Position - PositionB']

    for i in range(len(plot_func_list)):
        plot_x = np.linspace(range_list[i][0], range_list[i][1], 1000)

        fig, ax = plt.subplots(figsize=(4, 3), tight_layout=True)
        ax.grid(linestyle='-')

        for func in plot_func_list[i]:
            plt.plot(plot_x, func(plot_x), linewidth=3)

        ax.set_xlabel(x_list[i], fontsize=16)
        ax.set_ylabel('Membership', fontsize=16)

        plt.legend(legend_list[i], loc='lower right')

        plt.grid(axis='x')
        # plt.show()
        plt.savefig('cp_s{}.pdf'.format(i))
