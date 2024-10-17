
class TrainConfig:
    def __init__(self, context_size, batch_size, epochs, learning_rate, 
                 saving_freq, save_path, validation_freq, count_validation_steps, generation_freq):
        self.context_size = context_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.saving_freq = saving_freq
        self.save_path = save_path
        self.validation_freq = validation_freq
        self.count_validation_steps = count_validation_steps
        self.generation_freq = generation_freq