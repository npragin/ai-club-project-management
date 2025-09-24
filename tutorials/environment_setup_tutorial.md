# Environment Setup Tutorial

Author: Noah Pragin

This is a tutorial for getting your development environment set up and ready to work on a project! In this tutorial we will:

- Get you on the engineering servers (optional)
- Setup GitHub and git
- Download and setup VS Code, the application you'll use to write code (optional)
- Ready Python and Python's package manager Pip
- Introduce Python virtual environments

## Where Will You Work?
You have two options on where you will write your code and train your models. The first is your computer, and the second are the engineering servers that OSU makes available to its students. OSU's engineering servers provide a Linux environment with many tools pre-installed, and are where CS and ECE students complete a majority of their coursework. We recommend using the ENGR servers, but you are free to make your own decision. To help you, here are the pros and cons of each environment:

| | Pros | Cons |
| - | - | - |
| **Engineering Servers** | - No need to install Python or Pip, it's ready for you<br>- If you end up needing to use the HPC it will be *slightly* easier to move your files over<br>- Keep your computer clean, you will already be doing a lot of work on the engineering servers as a CS/ECE student<br>- Consistent environment across groups (Linux), no "works on my machine" issues<br>- Don't take up space on your own computer | - You will need internet access to work on your project<br>- Can be slower on occasion due to network latency|
| **Your Computer** | - No internet access required to work on your project<br>- More responsive development experience (no latency because no internet connection)<br>- Full control over your environment | - Need to install Python and Pip yourself<br>- Potential issues for collaboration if partners use a different OS than you<br>- Uses up your storage space (datasets are big)<br>- Harder for us to help you |

#### **If you choose to work locally (on your own computer), skip to the [GitHub and git setup section](#git-setup), otherwise continue to the Getting on the ENGR Servers section**

### Getting on the ENGR Servers
1. Navigate to [teach.engr.oregonstate.edu](https://teach.engr.oregonstate.edu/teach.php?type=want_auth)
2. Log in to TEACH using the same credentials you would to access your email
3. If it fails, you do not have a TEACH account and need to create one by clicking "Create a new account (Enable your Engineering resources)"
4. Open a terminal and run `ssh <ONID>@access.engr.oregonstate.edu`, **replacing &lt;ONID&gt; with your ONID username** (for example, mine is praginn)
5. Enter your ONID password if prompted
6. Select your DUO device when prompted, complete the MFA
7. You should now be able to execute commands on the ENGR servers!

#### Verification

To ensure you are on the ENGR servers, run the command `hostname` and verify that you see something like `flipX.engr.oregonstate.edu`, where X is a number 1-4.

## <a id="git-setup"></a>GitHub and git setup
SSH keys and repository set up (create a directory and repository)
## Visual Studio Code Download
How to download VSC, what extensions to download (python, jupyter, remote ssh, pylance if it doesn't come with python)
## Python and Pip download
Should use ENGR to just module load, but link to python download documentation
## venv setup and usage
Setup a venv, source it, pip install something with venv sourced, deactivate, show that dependency is only available when venv is sourced

TODO(npragin): Make sure there are verification steps throughout for students to ensure they've done things correctly
