import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch

sns.set_theme(style="darkgrid")
matplotlib.use('Agg')


def plot_mf(model, ob_low, ob_high, writer, update_count):
    for _action in model.actor.rule_dict.keys():
        for k, _rule in enumerate(model.actor.rule_dict[_action]):
            for j, _membership_network in enumerate(_rule.membership_network_list):
                fig = plt.figure(figsize=(4, 3))

                state_id = _rule.state_id[j]
                x = torch.linspace(ob_low[state_id], ob_high[state_id], 100).cuda()
                y = torch.zeros_like(x).cuda()

                for i, _ in enumerate(x):
                    y[i] = _membership_network(_.reshape(-1, 1)).detach()

                sns.lineplot(x=x.cpu().numpy(), y=y.cpu().numpy())
                plt.box(True)
                plt.ylim([0, 1])
                writer.add_figure('action_{}_rule_{}/state_{}'.format(_action, k, state_id), fig, update_count)
