from typing import NoReturn
from GLCM import GLCMFeatureExtraction
import time
import os
import argparse

def main()->NoReturn:
    parser = argparse.ArgumentParser(description='Esse código executa todo o pipeline de extração de treino e teste das características usando GLCM.')
    parser.add_argument('-d', '--dataset', required=True, help='Diretório para o Dataset')
    
    args = parser.parse_args()
    
    #Checagens do diretório passado
    if not os.path.exists(args.dataset):
        print(f"Erro: O diretóio '{args.dataset}' não existe")
        exit(1)
    elif not os.path.isdir(args.dataset):
        print(f"Erro: O diretório '{args.dataset}' não é um diretório")
        exit(1)
    elif len(os.listdir(args.dataset)) == 0:
        print(f"Erro: O diretório '{args.dataset}' está vazio")
        exit(1)
    
        
    extrator = GLCMFeatureExtraction(args.dataset)

    extrator.train()
    
    extrator.test()

if __name__ == "__main__":
    main()