import logging
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, RandomSampler
from nebula.core.datasets.changeablesubset import ChangeableSubset
import pickle as pk


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_set,
        train_set_indices,
        test_set,
        test_set_indices,
        local_test_set_indices,
        partition_id=0,
        partitions_number=1,
        batch_size=32,
        num_workers=0,
        val_percent=0.1,
        label_flipping=False,
        data_poisoning=False,
        poisoned_persent=0,
        poisoned_ratio=0,
        targeted=False,
        target_label=0,
        target_changed_label=0,
        noise_type="salt",
        trust = False,
        scenario_name="",
        idx = 0,
    ):
        super().__init__()
        self.train_set = train_set
        self.train_set_indices = train_set_indices
        self.test_set = test_set
        self.test_set_indices = test_set_indices
        self.local_test_set_indices = local_test_set_indices
        self.partition_id = partition_id
        self.partitions_number = partitions_number
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percent = val_percent
        self.label_flipping = label_flipping
        self.data_poisoning = data_poisoning
        self.poisoned_percent = poisoned_persent
        self.poisoned_ratio = poisoned_ratio
        self.targeted = targeted
        self.target_label = target_label
        self.target_changed_label = target_changed_label
        self.noise_type = noise_type
        self.trust = trust
        self.scenario_name = scenario_name
        self.idx = idx

        logging.debug(f"Train set indices: {train_set_indices}")
        logging.debug(f"Test set indices: {test_set_indices}")
        logging.debug(f"Local test set indices: {local_test_set_indices}")

        # Training / validation set
        # rows_by_sub = floor(len(train_set) / self.partitions_number)
        tr_subset = ChangeableSubset(
            train_set,
            train_set_indices,
            label_flipping=self.label_flipping,
            data_poisoning=self.data_poisoning,
            poisoned_persent=self.poisoned_percent,
            poisoned_ratio=self.poisoned_ratio,
            targeted=self.targeted,
            target_label=self.target_label,
            target_changed_label=self.target_changed_label,
            noise_type=self.noise_type,
        )

        train_size = round(len(tr_subset) * (1 - self.val_percent))
        val_size = len(tr_subset) - train_size

        data_train, data_val = random_split(
            tr_subset,
            [
                train_size,
                val_size,
            ],
        )

        # Test set
        # rows_by_sub = floor(len(test_set) / self.partitions_number)
        global_te_subset = ChangeableSubset(test_set, test_set_indices)

        # Local test set
        local_te_subset = ChangeableSubset(test_set, local_test_set_indices)

        if len(test_set) < self.partitions_number:
            raise "Too much partitions"

        # DataLoaders
        self.train_loader = DataLoader(
            data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=False,
        )
        self.val_loader = DataLoader(
            data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=False,
        )
        self.test_loader = DataLoader(
            local_te_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=False,
        )
        self.global_test_loader = DataLoader(
            global_te_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=False,
        )
        random_sampler = RandomSampler(data_source=data_val, replacement=False, num_samples=max(int(len(data_val) / 3), 300))
        self.bootstrap_loader = DataLoader(data_train, batch_size=self.batch_size, shuffle=False, sampler=random_sampler)
        logging.info("Train: {} Val:{} Test:{} Global Test:{}".format(len(data_train), len(data_val), len(local_te_subset), len(global_te_subset)))
        
        if trust:
            # Save data to local files to calculate the trustworthyness
            train_loader_filename = f"app/logs/{scenario_name}/trustworthiness/participant_{self.idx}_train_loader.pk"
            test_loader_filename = f"app/logs/{scenario_name}/trustworthiness/participant_{self.idx}_test_loader.pk"

            with open(train_loader_filename, 'wb') as f:
                pk.dump(self.train_loader, f)
                f.close()
            with open(test_loader_filename, 'wb') as f:
                pk.dump(self.test_loader, f)
                f.close()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return [self.test_loader, self.global_test_loader]

    def bootstrap_dataloader(self):
        return self.bootstrap_loader
