# Megatron Clariden 

This is the codebase to run megatron efficiently at scale using Clariden. The codebase and arguments are more complex than nanotron but this is currently the price we pay for efficiency at scale. 

## Setup

1. Clone megatron to your `iopsstor`.
  ```
  cd /iopsstor/scratch/cscs/$USER/
  git clone https://github.com/swiss-ai/Megatron-LM.git
  ```

2. You need to set the weights & biases environment variable with your key. Ideally you add it in `~/.bashrc`.
  ```
  export WANDB_API_KEY=57b215a80f....
  ```

3. To enable the slack bot during training, you need to set the following environment variables, in a similar way:
```
  export SLACK_BOT_TOKEN=xoxb-60756381....
  export SLACK_CHANNEL_ID=C08....
  ```
  Now re-login or run `source ~/.bashrc` to make sure varibales are set. 

## Submit a run

1. Start an 8B run by submitting the `submit-llama3-8B.sh` sbatch script. 
  ```
  sbatch submit-llama3-8B.sh
  ```
  You can find all available arguments here: [https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/training/arguments.py](https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/training/arguments.py)
  (We will add 1.5B, 3B, and 70B baseline runs using the llama3 architecture very soon.)

2. The main logs will be visible ~2min after launching the script in the `TheMeg-Clariden` Project on your personal weights and biases page.

3. The local logs will be at `/iopsstor/scratch/cscs/schlag/Megatron-LM/logs` with the slurm outputs and error files at `/iopsstor/scratch/cscs/schlag/Megatron-LM/logs/slurm/training`.

## How to contribute

You can submit issues and create branches on `https://github.com/swiss-ai/Megatron-LM`. The main branch is protected so you won't be able to directly commit to it.

1. Pull the latest version
  ```
  git pull
  ```

2. Create a branch with a descriptive name
  ```
  git checkout -b my-contribution-branch
  ```

3. Make your changes locally and commit them. Make sure your latest commit works as expected. Do not commit unnecessary files. 

4. sync with the latest main
  ```
  git pull origin main
  ```

5. Push your commit to your branch
  ```
  git push origin my-contribution-branch
  ```

6. Go to `https://github.com/swiss-ai/Megatron-LM` and create a new issue that references your branch and explains what you did. Add `ischlag` or `TJ-Solergibert` for now. 

7. Use this command to return to main branch.
  ```
  git checkout main
  ```

## How To's

We will add detailed steps for specific workflows fairly soon.
