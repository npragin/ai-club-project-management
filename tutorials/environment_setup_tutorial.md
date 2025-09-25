# Environment Setup Tutorial

Author: Noah Pragin

This tutorial will guide you through setting up your development environment and preparing it for project work! In this tutorial, we will:

- Get you on the engineering servers (optional)
- Set up GitHub and Git
- Download and set up VS Code, the application you'll use to write code (optional)
- Ready Python and Python's package manager Pip
- Introduce Python virtual environments

## Where Will You Work?
You have two options for where you will write your code and train your models. The first is your computer, and the second is the engineering servers that OSU makes available to its students. OSU's engineering servers offer a Linux environment with numerous pre-installed tools, where CS and ECE students complete the majority of their coursework. We recommend using the ENGR servers, but you are free to make your own decision. To help you, here are the pros and cons of each environment:

| | Pros | Cons |
| - | - | - |
| **Engineering Servers** | - No need to install Python or Pip, it's ready for you<br>- If you end up needing to use the HPC it will be *slightly* easier to move your files over<br>- Keep your computer clean, you will already be doing a lot of work on the engineering servers as a CS/ECE student<br>- Consistent environment across groups (Linux), no "works on my machine" issues<br>- Don't take up space on your own computer | - You will need internet access to work on your project<br>- Can be slower on occasion due to network latency|
| **Your Computer** | - No internet access required to work on your project<br>- More responsive development experience (no latency because no internet connection)<br>- Full control over your environment | - Need to install Python and Pip yourself<br>- Potential issues for collaboration if partners use a different OS than you<br>- Uses up your storage space (datasets are big)<br>- Harder for us to help you |

#### **If you choose to work locally (on your own computer), skip to the [GitHub and Git Setup section](#git), otherwise continue to the Getting on the ENGR Servers section**

### Getting on the ENGR Servers
1. Navigate to [teach.engr.oregonstate.edu](https://teach.engr.oregonstate.edu/teach.php?type=want_auth)
2. Log in to TEACH using the same credentials you would use to access your email
3. If it fails, you do not have a TEACH account and need to create one by clicking "Create a new account (Enable your Engineering resources)"
4. Open a terminal and run `ssh <ONID>@access.engr.oregonstate.edu`, **replacing &lt;ONID&gt; with your ONID username** (for example, mine is praginn)
5. Enter your ONID password if prompted
6. Select your DUO device when prompted, and complete the MFA
7. You should now be able to execute commands on the ENGR servers!

### ENGR Access Verification

To ensure you are on the ENGR servers, run the command `hostname` and verify that you see something like `flipX.engr.oregonstate.edu`, where X is a number 1-4.

### OPTIONAL: Want to Never Use Duo or Your Password Again?

If you want to avoid having to enter your password and complete Duo MFA every time you log in, you can set up SSH keys for the ENGR servers. This is optional, but recommended if you plan to work on the ENGR servers often. Follow this [video tutorial](https://youtu.be/6ZbZXPIsZtI?si=LtiNKq6084Bxp-6P&t=68) (1:08 to 7:15) to set it up.

## <a id="git"></a>GitHub and Git Setup

Now that you know where you'll be working, we will set you up with Git and GitHub. Git is a version control system that helps you track changes to your code, and GitHub is a platform that hosts git repositories (AKA projects) online. Together, they enable you to save your work, collaborate with others, and maintain a record of your project changes to share with others.

#### **If you already have git and GitHub set up wherever you decided to work (verifiable via the [verification step for this section](#git-verification)), skip to the [Visual Studio Code Setup section](#vsc)**

#### **If you already have a GitHub account, skip to the [Setting up SSH Keys for GitHub section](#git-ssh)**

### Creating a GitHub Account
If you don't have a GitHub account, click this [link](https://github.com/signup) to create one

**NOTE**: Use your personal email for this! Your GitHub account should serve as a portfolio of your work, meaning it should remain accessible after you graduate and your school email is deactivated.

### <a id="git-ssh"></a> Setting up SSH Keys for GitHub
SSH keys provide a secure way to authenticate with GitHub without needing to enter your password every time. We'll generate these keys on the ENGR servers (or your local machine if working locally).

1. In your terminal, run `ssh-keygen -t ed25519 -C "your_email@email.com"`, **replacing `your_email@email.com` with the email address associated with your GitHub account**
2. When prompted for where to store the keys, press Enter to use the default location
3. When prompted for a passphrase, you can leave it empty (press Enter) for convenience, but adding a passphrase is more secure
4. Your key has now been created. Next, we must print the public key by running `cat ~/.ssh/id_ed25519.pub`
5. Copy the SSH key by selecting and copying the entire output (usually `Ctrl+Shift+C` or `Cmd+Shift+C`)
6. Add the key to GitHub:
    1. Go to GitHub and click your profile icon (top-right corner)
    2. Go to Settings → SSH and GPG keys → New SSH key
    3. Give your key a descriptive title (like "OSU ENGR Servers" or "My Laptop")
    4. Paste your copied key into the "Key" field
    5. Click "Add SSH Key"

### <a id="git-verification"></a>GitHub SSH Keys Verification

To ensure you set up your SSH keys correctly, run `ssh -T git@github.com` in your terminal. You should see a message like:

```txt
Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.
```

### Creating Your First Repository

Now that you've set up your SSH keys, let's create a test repository to practice the complete git workflow. This will help you understand how to create projects, make changes, and sync them between your local environment and GitHub.

#### **If you've made a repository before and hosted it on GitHub, skip to the [Visual Studio Code Setup section](#vsc)**

1. **Create a repository on GitHub**
   1. On GitHub, click the "+" icon in the top-right corner, then select "New repository"
   2. Give your repository a name like `test-repo` or `my-first-project`
   3. Check the box "Add a README file" (this adds a README.md file to your project, used to describe the repository so others know what it's about)
      - Normally, I would advise against initializing a repository with a README, but for this tutorial, it simplifies the process
   4. Click "Create repository"

2. **Clone the repository to your working environment**
   1. On your new repository's main page, click the green "Code" button
   2. Make sure "SSH" is selected (not HTTPS)
   3. Copy the SSH URL (it should look like `git@github.com:yourusername/test-repo.git`)
   4. In your terminal, navigate to where you want to store your projects
   5. Clone the repository: `git clone <paste-your-SSH-URL-here>`
   6. Navigate into the new directory: `cd test-repo` (or whatever you named it)

4. **Make some changes and push them back to GitHub**
   1. Make a change to the README file: `echo "\nThis is my first GitHub repository!" >> README.md`
   2. Check what files have changed: `git status`
   3. Stage your changes: `git add README.md` (or `git add .` to add all changes)
   4. Commit your changes: `git commit -m "Updated README with a personal message"`
   5. Push to GitHub: `git push`

### Git and GitHub Workflow Verification

If you can see your updated README.md file in your GitHub repository with your changes, congratulations! You've completed the full git/GitHub workflow. You now know how to:
- Create repositories on GitHub
- Clone them to your local environment
- Make changes and push them back
- Sync your work between local and remote copies of your repository

This workflow is precisely what you'll use for your AI Club project throughout the term.

## <a id="vsc"></a> Visual Studio Code Setup

You are free to use any text editor for your project, however, we recommend beginners to use Visual Studio Code (VS Code). VS Code is a free, lightweight code editor that integrates seamlessly with git, remote servers, and community extensions.

#### **If you're not using VS Code, you can skip to the [Python and Pip Setup section](#python-setup)**

### VS Code Installation
1. Go to [code.visualstudio.com/download](https://code.visualstudio.com/download) and download the version of VS Code for your operating system
2. After installing VS Code, launch it

### Recommended Extensions

**NOTE**: If you are working on the ENGR servers, the Remote - SSH extension is required.

VS Code's power comes from its extensions. You can open the Extensions panel by clicking the extensions icon in the left sidebar (four squares icon) or pressing `Ctrl+Shift+X` (`Cmd+Shift+X` on Mac). Whenever installing Extensions, be sure to verify the author. Here are the ones we recommend for your project:

1. **Python Extension Pack**
   - In the Extensions panel, search for "Python" and install the extension published by Microsoft
   - This automatically includes Pylance (a powerful language server for Python), a Python debugger, and some other nifty stuff
   - This extension pack is useful to you because most of the code you write for your project will be in Python!
2. **Jupyter Extension Pack**
   - In the Extensions panel, search for "Jupyter" and install the extension published by Microsoft
   - This provides Jupyter notebook support, allowing you to create, edit, and more seamlessly interact with Jupyter notebooks directly in VS Code
   - This extension pack is useful to you because our tutorials for the libraries you'll use to train your models will be in Jupyter notebooks!
3. **Remote - SSH** (if working on ENGR servers):
   - In the Extensions panel, search for "Remote - SSH"
   - Install the extension by Microsoft
   - This allows you to connect directly to ENGR servers from within VS Code and edit files as if they were on your computer

### Setting Up Remote Development

#### **If you're working locally, skip to the [VS Code Verification section](#vsc-verification)**

1. **Add ENGR Servers to Extension**
   - Press `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) to open the command palette
   - Type "connect to host" and you should see "Remote-SSH: Connect to Host..." - select it
   - Select "Add New SSH Host..."
   - Enter `ssh <ONID>@access.engr.oregonstate.edu` (replace `<ONID>` with your actual ONID)
   - Select where to save the SSH configuration (the default is fine)

2. **Connect to the ENGR Servers**
   - Now that you've added the host, open the command palette again (`Ctrl+Shift+P` or `Cmd+Shift+P`)
   - Type "connect to host" and select "Remote-SSH: Connect to Host..."
   - Select the host you just added
   - A new VS Code window will open, connecting you to the ENGR servers

3. **Explore**
   - Once connected, open the file explorer by clicking the file icon in the left sidebar or pressing `Ctrl+Shift+E` (`Cmd+Shift+E` on Mac)
   - You can now edit files on the ENGR servers using VS Code's full interface

### <a id="vsc-verification"></a>VS Code Verification

1. Create a new file with a `.py` extension (for example, `test.py`) and verify that you see Python syntax highlighting (keywords in different colors, such as for, if, def, etc.)
2. If you are working on the ENGR servers, verify that you see `SSH: <ONID>@access.engr.oregonstate.edu` in the bottom-left corner of VS Code, indicating you are connected to the ENGR servers.

## <a id="python"></a>Python and Pip download
Should use ENGR to just module load, but link to the Python download documentation
## venv setup and usage
Set up a venv, source it, Pip install something with venv sourced, deactivate, and show that the dependency is only available after sourcing the venv

TODO(npragin): Make sure there are verification steps throughout for students to ensure they've done things correctly
