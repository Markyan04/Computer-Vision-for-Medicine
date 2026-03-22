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

from MDFA_new import MDFA
from chest_xray_loss_experiment_common import (
    ResNet50WithInsertedModule,
    run_chestxray_medical_losses_experiments,
)


def build_model(num_classes: int):
    module = MDFA(dim_in=1024, dim_out=1024)
    return ResNet50WithInsertedModule(
        num_classes=num_classes,
        inserted_module=module,
        insert_after="layer3",
    )


if __name__ == "__main__":
    run_chestxray_medical_losses_experiments(
        script_stem="ResNet_layer3+MDFA",
        model_builder=build_model,
        optimizer_group_divisors=[
            ("conv1", 10),
            ("bn1", 10),
            ("layer1", 8),
            ("layer2", 6),
            ("layer3", 4),
            ("inserted_module", 3),
            ("layer4", 2),
            ("fc", 1),
        ],
        module_name="MDFA",
        insert_after="layer3",
    )
