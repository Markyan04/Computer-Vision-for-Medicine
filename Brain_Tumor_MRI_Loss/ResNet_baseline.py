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

from brain_tumor_mri_loss_experiment_common import ResNet50Baseline, run_brain_tumor_mri_medical_losses_experiments


def build_model(num_classes: int):
    return ResNet50Baseline(num_classes=num_classes)


if __name__ == '__main__':
    run_brain_tumor_mri_medical_losses_experiments(
        script_stem='ResNet_baseline',
        model_builder=build_model,
        optimizer_group_divisors=[
            ('conv1', 10),
            ('bn1', 10),
            ('layer1', 10),
            ('layer2', 10),
            ('layer3', 10),
            ('layer4', 10),
            ('fc', 1),
        ],
        module_name='Baseline',
        insert_after='none',
    )
