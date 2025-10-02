# AI Club Project Workshop

Welcome to the Oregon State AI Club's Project Workshop repository! This repository contains everything you need to build your first machine learning project over the fall term.

## What's Inside

### 📚 Tutorials
Step-by-step guides to help you learn the industry-standard tools for machine learning. Each tutorial includes practical exercises and real-world applications to prepare you for your ML project.

- **Environment Setup** - Get your development environment ready for your project!
- **NumPy** - Learn fast numerical computing with arrays
- **Pandas** - Master data loading, manipulation, and analysis
- **Matplotlib** - Create visualizations to understand your data and results
- **Scikit-learn** - Train, evaluate, and save machine learning models
<!-- - **PyTorch** - Build and train neural networks -->

### 🛠️ Example Project
Follow along with a complete machine learning project that demonstrates the full workflow:
- **Week 1**: Setting up the environment and tools (no code yet!)
- **Week 2**: Loading and exploring the [student dropout dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Week 3**: Performing exploratory data analysis and building a baseline model
- **Weeks 4-10**: (Coming soon as we progress through the term!)

## Getting Started

1. **Complete the Environment Setup Tutorial** (`tutorials/environment_setup_tutorial.md`)
   - Set up your development environment (ENGR servers or local machine)
   - Configure Git and GitHub for version control
   - Install Python, pip, and create virtual environments

2. **Work Through the Tutorials**
   - Start with NumPy
   - Move to Pandas and (optionally) Matplotlib
   - Finish with scikit-learn!
   - Complete the exercises in each tutorial to reinforce your learning
   <!-- - Go through the PyTorch tutorial if you're ready to build your own neural network -->

3. **Start Your Project!**
   - Check out our slides on our [website](https://osu-ai.club/project-workshop) that outline the project workflow
   - Begin working on your own ML project using the skills you've learned
   - If you need a reference, check the `example_project/`

## Contributing

- Claim your authorship at the top of your tutorial
- Link to official documentation when introducing new functions or interfaces
- Include practical exercises with solutions
- Contextualize content with ML applications and benefits
- Strip notebook outputs before committing (see workflow below)

### Notebook Outputs

To keep the tutorials clean, we strip outputs from Jupyter notebooks before committing. **Before committing notebooks, run**:
```bash
nbstripout --extra-keys="metadata.kernelspec metadata.language_info" **/*.ipynb
```

## Questions or Issues?

- Reach out to the AI Club officers on [Discord](https://discord.com/invite/2rncuBvaUC)
- Bring questions to our weekly workshop meetings
- Open an issue in this repository
