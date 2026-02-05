#!/usr/bin/env python3
import argparse
import json
import torch
import math
import torch.nn as nn
from theta_model import build_theta_model, count_parameters
# ---------------------------------------------------------------
# Import your model code. Adjust path/module name if needed.
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Helper: count parameters per module
# ---------------------------------------------------------------
def summarize_parameters(model: nn.Module):
    summary = {
        "noise": 0,
        "input_proj": 0,
        "blocks": 0,
        "output_proj": 0,
        "other": 0,
    }

    detailed = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()

        if name.startswith("noise_embed") or name.startswith("noise_proj"):
            summary["noise"] += n
            detailed.append(("noise/" + name, n))

        elif name.startswith("input_proj"):
            summary["input_proj"] += n
            detailed.append(("input_proj/" + name, n))

        elif name.startswith("blocks"):
            summary["blocks"] += n
            detailed.append(("blocks/" + name, n))

        elif name.startswith("output_proj"):
            summary["output_proj"] += n
            detailed.append(("output_proj/" + name, n))

        else:
            summary["other"] += n
            detailed.append((name, n))

    return summary, detailed


def size_abbrv(config_path):
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Wrap in runtime dict expected by build_theta_model()
    runtime = {"config": config}

    # Build model
    model = build_theta_model(runtime)
    summary, detailed = summarize_parameters(model)
    total = sum(summary.values())
    if total >= 1000000000: 
        abbv = total // 1000000000
        abbv_len = len(str(abbv))
        abbv_1st = int(str(abbv)[0])
        if abbv_len > 1:
            abbv_2nd = int(str(abbv)[1])
            if abbv_2nd >= 5:
                message = str(abbv_1st+1)+"0"*max(0,(abbv_len-1))+"B"
            else:
                message = str(abbv_1st)+"0"*max(0,(abbv_len-1))+"B"
        else:
            message = str(abbv_1st)+"B"
        return message
    elif total >= 1000000:
        abbv = total // 1000000
        abbv_len = len(str(abbv))
        abbv_1st = int(str(abbv)[0])
        if abbv_len > 1:
            abbv_2nd = int(str(abbv)[1])
            if abbv_2nd >= 5:
                message = str(abbv_1st+1)+"0"*max(0,(abbv_len-1))+"M"
            else:
                message = str(abbv_1st)+"0"*max(0,(abbv_len-1))+"M"
        else:
            message = str(abbv_1st)+"M"
        return message
    elif total >= 1000:
        abbv = total // 1000
        abbv_len = len(str(abbv))
        abbv_1st = int(str(abbv)[0])
        if abbv_len > 1:
            abbv_2nd = int(str(abbv)[1])
            if abbv_2nd >= 5:
                message = str(abbv_1st+1)+"0"*max(0,(abbv_len-1))+"K"
            else:
                message = str(abbv_1st)+"0"*max(0,(abbv_len-1))+"K"
        else:
            message = str(abbv_1st)+"K"
        return message
    return "Error"

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Model parameter counter")
    parser.add_argument("config_path", type=str,
                        help="Path to config.json")
    args = parser.parse_args()

    # Load config
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Wrap in runtime dict expected by build_theta_model()
    runtime = {"config": config}

    # Build model
    model = build_theta_model(runtime)
    summary, detailed = summarize_parameters(model)

    print("\n==================== SUMMARY ====================")
    total = sum(summary.values())
    for k, v in summary.items():
        print(f"{k:20s}: {v:12,}  ({v/total:6.2%})")

    print("Total parameters:", f"{total:,}")
    print("=================================================\n")


    # print("Detailed breakdown:")
    # for name, n in detailed:
    #     print(f"{name:50s} {n:12,}")

    # print("\nDone.")


# ---------------------------------------------------------------
# Entry
# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
