
import os

# Repository name
repo_name = "ForexMastermind"  # A suggested cool name for your repository

# Create folder structure
folders = [
    'Experts/MQ4',
    'Experts/MQ5',
    'Indicators/MQ4',
    'Indicators/MQ5',
    'Scripts/MQ4',
    'Scripts/MQ5',
    'Include/MQ4',
    'Include/MQ5',
    'Libraries/MQ4',
    'Libraries/MQ5',
    'Data',
    'Backtest_Results/EA1',
    'Backtest_Results/EA2'
]

# Create the repository folder
os.makedirs(repo_name, exist_ok=True)

# Change directory to the created repository
os.chdir(repo_name)

# Create the inner folder structure
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create .gitignore and README.md files
with open('.gitignore', 'w') as f:
    f.write('.ex4\n.ex5\n.log\n')

with open('README.md', 'w') as f:
    f.write('# ForexMastermind Repository\n')
    f.write('A repository for managing MQL4 and MQL5 scripts, indicators, EAs, and related components.')

print(f"Folder structure for '{repo_name}' created!")
