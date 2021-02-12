from models import tasks_alexnet
from models import tasks_resnet

nets_map = {
    'tasks_Alexnet': tasks_alexnet.tasks_Alexnet,
    'tasks_Resnet': tasks_resnet.tasks_Resnet
}

def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_network_fn