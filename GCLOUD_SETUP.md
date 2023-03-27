# Setting Up Gcloud with SSH

- Vertex > Workbench > New Notebook to create new notebook instance. Select Pytorch Env with GPU. Select to install GPU drivers for you.
- Install Gcloud SDK on your local machine. Follow the instructions here: https://cloud.google.com/sdk/docs/install
- Install zsh: `sudo apt install zsh` and oh-my-zsh: `sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"`
- Login to GCloud with CLI: `gcloud auth login`
- To connect to VM via ssh, run `gcloud compute ssh --zone=<ZONE> --project=<PROJECT> <INSTANCE_NAME>`
- Install Pyenv and Poetry to the VM. Follow the instructions [here](https://github.com/dqmis/code_academy_ai_course).
  - Alternative Pyenv installation for Linux: https://medium.com/thoughful-shower/how-to-install-python-using-pyenv-on-ubuntu-20-04-5db6295b804f
  - Install alternate python version: `pyenv install 3.10.2` and set it as global: `pyenv global 3.10.2`
- Generate your VMs SSH public key: `ssh-keygen -t ed25519 -C "your_email@example.com"` and add it to GitHub account. Settings > SSH and GPG keys > New SSH key.
- Clone your GitHub repo to the VM using SSH.
- Install all of the dependencies running: `poetry install`
- Connect VSCode to SSH > <CTRL> + <SHIFT> + <P> > Remote SSH. Add user and ip address
- Install Python, Jupyter extensions in VS Code