# Practical 8

class CustomLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model  #Dimensionality of the model
        self.warmup_steps = warmup_steps
        self.step_count = 0     #Tracks the current step

    def step(self):     #Update the learning rate at each step
        self.step_count += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):   #Current learning rate
        scale = self.d_model ** -0.5
        step_factor = self.step_count ** -0.5
        warmup_factor = self.step_count * self.warmup_steps ** -1.5
        return scale * min(step_factor, warmup_factor)
