from function_plot import *


def plot_k(agent_dicts_dict):
    ##### If all function the same, try its owwn plot
    functions = [agent_dict['general']['FUNCTION'] for agent_dict in agent_dicts_dict.values()]
    if len(set(functions))==1:
        try:
            plot_agent = eval(functions[0]+'_plot(agent_dicts_dict)')
        except:
            plot_agent = base_plot(agent_dicts_dict)
    else:
        plot_agent = base_plot(agent_dicts_dict)

