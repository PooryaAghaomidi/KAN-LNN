from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def t_callback():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f'checkpoints/cvae_{timestamp}'

    return SummaryWriter(model_name), model_name + '.pt'
