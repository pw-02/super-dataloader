import random
from batch import Batch
from dataset import Dataset, DatasetPartition
from utils import create_unique_id
from itertools import cycle

class EndOfPartitionException(Exception):
    pass

class SequentialSampler:
    def __init__(self, dataset_len: int):
        self.dataset_len = dataset_len
        self.indices = list(range(dataset_len))
        self.current_index = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index < self.dataset_len:
            index = self.current_index
            self.current_index += 1
            return index
        else:
            self.current_index = 0  # Reset index for the next epoch
            raise StopIteration

    def __len__(self):
        return self.dataset_len


class RandomSampler:
    def __init__(self, dataset_len: int):
        self.dataset_len = dataset_len
        self.indices = list(range(dataset_len))
        self.shuffle_indices()

    def shuffle_indices(self):
        random.shuffle(self.indices)
        self.current_index = 0

    def __iter__(self):
        self.shuffle_indices()  # Shuffle indices at the start of each epoch
        return self
    
    def __next__(self):
        if self.current_index < self.dataset_len:
            index = self.indices[self.current_index]
            self.current_index += 1
            return index
        else:
            raise StopIteration

    def __len__(self):
        return self.dataset_len


class BatchSampler:
    def __init__(self, partitions, batch_size, shuffle=True, drop_last=False):
        
        self.partitions = partitions  # List of partitions to cycle through
        self.partitions_cycle = cycle(partitions)  # Create a cycle iterator over partitions
        self.active_partition:DatasetPartition = next(self.partitions_cycle)  # Start with the first partition
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.epoch_idx = 1
        self.current_idx = 0
        self.shuffle = shuffle
        self.sampler = self._create_sampler(len(self.active_partition))

    def __iter__(self):
        return self
    
    
    def _create_sampler(self, size):
        """Create a new sampler based on the shuffle setting."""
        if self.shuffle:
            return RandomSampler(size)
        else:
            return SequentialSampler(size)

    def __next__(self):
        batch = []
        while len(batch) < self.batch_size:
            try:
                batch.append(next(self.sampler))
            except StopIteration:
                if not self.drop_last and len(batch) > 0:
                    self.current_idx += 1
                    batch_id = f"{self.epoch_idx}_{self.active_partition.partition_id}_{self.current_idx}_{create_unique_id(batch, 16)}"
                    self.active_partition = next(self.partitions_cycle)  # Move to the next partition
                    self.sampler = self._create_sampler(len(self.active_partition))  # Create a new sampler for the new partition
                    if self.active_partition.partition_id == 1:
                        self.epoch_idx += 1
                    self.current_idx = 0
                    return Batch(batch, batch_id, self.epoch_idx, self.active_partition.partition_id)

                else:
                    # When the current partition is exhausted, move to the next partition
                    self.active_partition = next(self.partitions_cycle)  # Move to the next partition
                    self.sampler = self._create_sampler(len(self.active_partition))  # Create a new sampler for the new partition
                    if self.active_partition.partition_id == 1:
                        self.epoch_idx += 1
                    self.current_idx = 0
                    continue
              
        self.current_idx += 1
        batch_id = f"{self.epoch_idx}_{self.active_partition.partition_id}_{self.current_idx}_{create_unique_id(batch, 16)}"
        return Batch(batch, batch_id, self.epoch_idx, self.active_partition.partition_id)


if __name__ == "__main__":
    batch_size = 128
    size = 50000
    initial_seed = 42
    num_epochs = 10  # Specify the number of epochs you want to run

    dataset = Dataset(data_dir='s3://sdl-cifar10/test/', batch_size=128, drop_last=False, num_partitions=10)
    batch_sampler_random = BatchSampler(partitions=dataset.partitions.values(), batch_size=batch_size, shuffle=False, drop_last=False)


    for idx in range(100000):
        batch = next(batch_sampler_random)
        print(f'Batch {idx + 1} - Batch ID: {batch.batch_id}, Batch Size {len(batch.indicies)}')
      







    # for epoch in range(num_epochs):
    #     print(f"Starting epoch {epoch + 1}")

    #     # Initialize a BatchSampler with a random sampler for each epoch
    #     batch_sampler_random = BatchSampler(size, batch_size, initial_seed + epoch, shuffle=True, drop_last=False)
    #     try:
    #         # Iterate through each batch in the current epoch
    #         for counter, batch in enumerate(batch_sampler_random, start=1):
    #             print(f'Epoch {epoch + 1}, Batch {counter} - Batch ID: {batch.batch_id}, Batch Size {len(batch.indicies)}')
    #             # Process the batch here (e.g., training, validation, etc.)
    #             # Example: print(batch.data) if you want to see batch data
    #     except EndOfPartitionException:
    #         print(f"End of epoch {epoch + 1} reached\n")

    #     # Increment epoch seed if necessary
    #     # (You could also choose to use the same seed every epoch or modify the seed as needed)

    # print("All epochs completed.")
