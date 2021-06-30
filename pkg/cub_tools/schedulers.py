from torch.optim.lr_scheduler import ReduceLROnPlateau


class ReduceLROnPlateauScheduler:
    """Wrapper of torch.optim.lr_scheduler.ReduceLROnPlateau with __call__ method like other schedulers in 
        contrib\handlers\param_scheduler.py"""

    def __init__(
            self,
            optimizer,
            metric_name,
            mode='min', factor=0.1, patience=10,
            threshold=1e-4, threshold_mode='rel', cooldown=0,
            min_lr=0, eps=1e-8, verbose=False,
    ):
        self.metric_name = metric_name
        self.scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                           threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown,
                                           min_lr=min_lr, eps=eps, verbose=verbose)

    def __call__(self, engine, name=None):
        self.scheduler.step(engine.state.metrics[self.metric_name])

    def state_dict(self):
        return self.scheduler.state_dict()