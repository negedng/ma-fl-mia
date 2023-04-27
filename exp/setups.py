#########################################
#
#          Experiment setups
#
#########################################

def get_experiment(exp_name='default'):
    if exp_name == 'default':
        return exp_default()
    if exp_name == 'all_homo':
        return exp_all_homo()
    raise ValueError(f'not recognized exp name: {exp_name}')

def exp_default():
    conf_changes = [{}]
    return conf_changes


def exp_all_homo():
    conf_changes = []
    l = list(range(22,65,2))
    l = list(reversed(l))
    for s in l:
        c = {"scale_mode" : s/64}
        conf_changes.append(c)
    return conf_changes
    
