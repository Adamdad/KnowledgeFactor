import os.path as osp

from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook

from collections import OrderedDict

import numpy as np


@HOOKS.register_module()
class TensorboardVisLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 vis_tags=None,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        super(TensorboardVisLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir
        self.vis_tags = vis_tags

    @master_only
    def before_run(self, runner):
        super(TensorboardVisLoggerHook, self).before_run(runner)
        if (TORCH_VERSION == 'parrots'
                or digit_version(TORCH_VERSION) < digit_version('1.1')):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)
    
    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True, tags_to_skip=('time', 'data_time', 'relation'))
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
        if self.vis_tags is not None:
            for tag in self.vis_tags:
                if tag in runner.log_buffer.output.keys():
                    val = runner.log_buffer.output[tag]
                    self.writer.add_image(tag, val, self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()


class LogBuffer_ignore:

    def __init__(self, igore_key=['relation']):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ignore_keys = igore_key
        self.ready = False

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()
        self.ready = False

    def update(self, vars, count=1):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n=0):
        """Average latest n values or all values."""
        assert n >= 0
        for key in self.val_history:
            if key in self.ignore_keys:
                self.output[key] = self.val_history[key][-1]
            else:
                values = np.array(self.val_history[key][-n:])
                nums = np.array(self.n_history[key][-n:])
                avg = np.sum(values * nums) / np.sum(nums)
                self.output[key] = avg
        self.ready = True