# Anonymous

Anonymous Repo for an ICLR Submission paper.


## Installation

Our framework is based on [Verl](https://github.com/volcengine/verl) and [TinyV](https://github.com/volcengine/verl). To install our environment, you can use:



## Installation
Please refer to the requirement.txt file.

## Data Process



## Training

**1. RL with TinyV:** Please start your training from [run_grpo_tinyv.sh](./run_grpo_tinyv.sh). Since TinyV is integrated with Verl, you can simply follow Verl setups to define your hyperparameters. Specifically, there are two new arguments related to TinyV in the bash script:

```
VERIFIER_MODEL=${5:-" "}
VERIFIER_SETUP=${6:-"addon"}
```
Configuration Options:

`VERIFIER_MODEL`: Specifies the TinyV model to use for verification. Default is `/TinyV-1.5B`, but you can replace it with other TinyV model variants or your own fine-tuned verifier.

`VERIFIER_SETUP`: Defines how TinyV integrates with the training process. Options include:
- `addon` (default): TinyV works alongside existing rule-based verifiers. TinyV is triggered only when the rule-based verifier determines the answer is incorrect.
- `tinyv_only`: Uses TinyV exclusively for verification, without using rule-based verifiers entirely.

Important Note: If you intend to use TinyV, please add the suffix `_tinyv` to your data source name. Otherwise, it will fall back to the default verifier, which is Prime Math.


