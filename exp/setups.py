#########################################
#
#          Experiment setups
#
#########################################
import json
    
def get_experiment(exp_name='default', params=""):
    if exp_name == 'default':
        return exp_default()
    if exp_name == "unit_size":
        return exp_unit_size3()
    if exp_name == "scale_us":
        return exp_scale_us3()
    if exp_name == "minus_one":
        return exp_unit_size_minus_one2()
    if exp_name == 'all_homo':
        return exp_all_homo()
    if exp_name == 'config':
        return exp_config(params)
    raise ValueError(f'not recognized exp name: {exp_name}')

def exp_default():
    conf_changes = [{}]
    return conf_changes


def exp_config(params):
    conf_changes = [dict(json.loads(params))]
    return conf_changes


def exp_unit_size():
    conf_changes = []
    l = list(range(4,64,10))
    for s in l:
        c = {"unit_size" : s}
        c['ma_mode'] = 'no'
        c['split_mode'] = 'homogen'
        c['scale_mode'] = 1.0
        c['rounds'] = 70
        conf_changes.append(c)
    return conf_changes


def exp_unit_size2():
    conf_changes = []
    l = list(range(2,24,2))
    for s in l:
        c = {"unit_size" : s}
        c['ma_mode'] = 'no'
        c['split_mode'] = 'homogen'
        c['scale_mode'] = 1.0
        c['rounds'] = 70
        conf_changes.append(c)
    return conf_changes

def exp_unit_size3():
    conf_changes = []
    l = list(range(2,40,4))
    for s in l:
        c = {"unit_size" : s}
        c['ma_mode'] = 'no'
        c['split_mode'] = 'homogen'
        c['scale_mode'] = 1.0
        c['rounds'] = 70
        conf_changes.append(c)
    return conf_changes

def exp_unit_size_minus_one():
    conf_changes = []
    l = list(range(4,24,4))
    l = l + [24, 34, 44, 54, 64]
    for s in l:
        c = {"unit_size" : s}
        c['ma_mode'] = 'rm-cid'
        c['split_mode'] = 'homogen'
        c['scale_mode'] = -1
        c['rounds'] = 70
        conf_changes.append(c)
    return conf_changes


def exp_unit_size_minus_one2():
    conf_changes = []
    l = list(range(8,24,4))
    l = l + [24, 34, 44, 54, 64]
    for s in l:
        c = {"unit_size" : s}
        c['ma_mode'] = 'rm-cid'
        c['split_mode'] = 'homogen'
        c['scale_mode'] = -1
        c['rounds'] = 70
        c['cut_type'] = "diagonal"
        conf_changes.append(c)
    return conf_changes


def exp_scale_us2():
    conf_changes = []
    l = list(range(4,24,4))
    for s in l:
        c = {"unit_size" : s}
        c['ma_mode'] = 'rm-cid'
        c['split_mode'] = 'homogen'
        c['scale_mode'] = 0.75
        c['rounds'] = 70
        conf_changes.append(c)
    return conf_changes


def exp_scale_us():
    conf_changes = []
    l = list(range(32,65,16))
    for s in l:
        c = {"unit_size" : s}
        c['ma_mode'] = 'rm-cid'
        c['split_mode'] = 'homogen'
        c['scale_mode'] = 0.75
        c['rounds'] = 70
        conf_changes.append(c)
    return conf_changes  


def exp_scale_us3():
    conf_changes = []
    l = list(range(10,27,2))
    for s in l:
        c = {"unit_size" : s}
        c['ma_mode'] = 'rm-cid'
        c['split_mode'] = 'homogen'
        c['scale_mode'] = 0.9
        c['rounds'] = 70
        conf_changes.append(c)
    return conf_changes  


def exp_all_homo():
    conf_changes = []
    l = list(range(22,65,2))
    l = list(reversed(l))
    for s in l:
        c = {"scale_mode" : s/64}
        conf_changes.append(c)
    return conf_changes
    
