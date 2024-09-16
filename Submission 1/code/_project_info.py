import subprocess
import os

def generate_project_info():
    # Caminho para o ambiente Conda, ajuste conforme necessário
    conda_path = "C:\\Users\\rafae\\miniconda3\\envs\\pandas_env"
    
    # Captura a lista de pacotes instalados
    result = subprocess.run([conda_path, 'list'], stdout=subprocess.PIPE, text=True)
    packages = result.stdout

    # Informações do repositório (ajuste o caminho para o seu caso)
    git_url = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], stdout=subprocess.PIPE, text=True)
    repository_url = git_url.stdout.strip()
    
    # Salva as informações em um arquivo
    with open("project_info.txt", "w") as file:
        file.write("Software used: Python, specific libraries:\n")
        file.write(packages)
        file.write("\nCode repository: " + repository_url)
        file.write("\nParameters: Default values, tuning details (please specify)")

# Substitua '/path/to/conda' pelo caminho correto do seu Conda no sistema
generate_project_info()
