#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from GCSA import GCSA
from chest_xray_loss_experiment_common import (
    ResNet50WithInsertedModule,
    run_chestxray_medical_losses_experiments,
)


def build_model(num_classes: int):
    module = GCSA(in_channels=512, rate=4)
    return ResNet50WithInsertedModule(
        num_classes=num_classes,
        inserted_module=module,
        insert_after="layer2",
    )


if __name__ == "__main__":
    run_chestxray_medical_losses_experiments(
        script_stem="ResNet_layer2+GCSA",
        model_builder=build_model,
        optimizer_group_divisors=[
            ("conv1", 10),
            ("bn1", 10),
            ("layer1", 8),
            ("layer2", 6),
            ("inserted_module", 4),
            ("layer3", 3),
            ("layer4", 2),
            ("fc", 1),
        ],
        module_name="GCSA",
        insert_after="layer2",
    )
