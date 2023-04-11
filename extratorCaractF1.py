import numpy as np
import cv2
import os
from sklearn.preprocessing import normalize

# Caminho para a pasta principal
main_dir = './Formula One Cars/'

# Lista para armazenar os resultados
results = []
count = 0

# Loop sobre todas as pastas na pasta principal
for subdir in os.listdir(main_dir):
    
    # Caminho para a subpasta atual
    sub_dir = os.path.join(main_dir, subdir)
    print(subdir)

    # Loop sobre todas as imagens na subpasta atual
    for img_name in os.listdir(sub_dir):
        # Caminho para a imagem atual
        img_path = os.path.join(sub_dir, img_name)
        
        # Carrega a imagem
        img = cv2.imread(img_path)
        
        # Converte a imagem para o espaço de cores HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calcula o histograma de cores com 90 bins no eixo H e 128 bins no eixo S
        # hist = cv2.calcHist([hsv_img], [0, 1], None, [180, 256], [0, 180, 0, 256]) # 46080
        hist = cv2.calcHist([hsv_img], [0, 1], None, [90, 128], [0, 180, 0, 256]) # 11520
        # hist = cv2.calcHist([hsv_img], [0, 1], None, [64, 102], [0, 180, 0, 256]) # 8192
        # hist = cv2.calcHist([hsv_img], [0, 1], None, [64, 64], [0, 180, 0, 256]) # 4096

        results_normalized = normalize([hist.flatten()])
        # Adiciona os resultados a lista
        results.append([results_normalized.flatten(), count])
    count += 1  

results = np.array(results, dtype=object)

np.save("features.npy", results)