import argparse
import os

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='results dir')

# Data
parser.add_argument("--train_base_dir", type=str, default="d:/data/QUBIQ2021/training_data_v3/training_data_v3")
parser.add_argument("--vali_base_dir", type=str, default="d:/data/QUBIQ2021/validation_data_qubiq2021/validation_data_qubiq2021")
parser.add_argument("--test_base_dir", type=str, default="d:/data/QUBIQ2021/validation_data_qubiq2021/validation_data_qubiq2021")
parser.add_argument("--dataset", type=str, default='pancreatic-lesion', help='can be brain-growth, brain-tumor, kidney, pancreas, pancreatic-lesion, prostate')
parser.add_argument("--task", type=str, default='task01')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--seed', type=int, default=1234)

# Model
parser.add_argument("--initial_filter_size", type=int, default=48)
parser.add_argument("--patch_size", type=int, default=512)
parser.add_argument("--classes", type=int, default=2)

# Train
# parser.add_argument("--experiment_name", type=str, default="")
parser.add_argument("--restart", default=False, action='store_true')
parser.add_argument("--pretrained_model_path", type=str, default='./results/pancreatic-lesion_task01_2021-08-10_07-47-44/model/latest.pth')
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--lr_scheduler", type=str, default='cos')
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))

def save_args(obj, defaults, kwargs):
    for k,v in defaults.iteritems():
        if k in kwargs: v = kwargs[k]
        setattr(obj, k, v)

def get_config():
    config = parser.parse_args()
    return config
