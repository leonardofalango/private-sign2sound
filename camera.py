import cv2

# Inicialize a câmera
cap = cv2.VideoCapture(0)  # Use 0 para a primeira câmera conectada, 1 para a segunda, e assim por diante

# Verifique se a câmera está aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

# Loop para capturar imagens
while True:
    # Captura um frame da câmera
    ret, frame = cap.read()

    # Verifica se o frame foi capturado corretamente
    if not ret:
        print("Erro ao capturar o frame")
        break

    # Mostra o frame capturado
    cv2.imshow('Camera', frame)

    # Aguarda pela tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
