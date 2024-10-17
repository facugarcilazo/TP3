import numpy as np

class RedHopfield:
    def __init__(self, tamaño):
        # Inicia la matriz de pesos
        self.tamaño = tamaño
        self.pesos = np.zeros((tamaño, tamaño))

    def entrenar(self, patrones):
        # Entrenamiento de la red usando la regla de Hebb
        for patron in patrones:
            self.pesos += np.outer(patron, patron)
        # Se prueba de que no haya auto-conexiones (pesos en la diagonal = 0)
        np.fill_diagonal(self.pesos, 0)

    def recordar(self, patron, iteraciones=5):
        # Método para recuperar el patrón guardado
        salida = patron.copy()
        for _ in range(iteraciones):
            for i in range(len(patron)):
                activacion = np.dot(self.pesos[i], salida)
                salida[i] = 1 si activacion >= 0 else -1
        return salida

def crear_patron_circulo(tamaño):
    # Generar un patrón binario de un círculo (aro "C") de tamaño 10 pixeles x 10 pixeles
    patron = np.zeros((tamaño, tamaño))
    radio = tamaño // 4
    centro = tamaño // 2
    for x in range(tamaño):
        for y in range(tamaño):
            if (x - centro) ** 2 + (y - centro) ** 2 <= radio ** 2:
                patron[x, y] = 1
            else:
                patron[x, y] = -1
    return patron.flatten()

def crear_imagen_con_ruido(patron, nivel_ruido=0.3):
    # Generar una imagen ruidosa basada en un patrón
    patron_ruidoso = patron.copy()
    cantidad_pixeles_ruido = int(nivel_ruido * len(patron))
    indices_volteo = np.random.choice(len(patron), cantidad_pixeles_ruido, replace=False)
    patron_ruidoso[indices_volteo] = -patron_ruidoso[indices_volteo]
    return patron_ruidoso

def encontrar_centro_circulo(imagen, tamaño):
    # Identificar el centro del círculo (coordenadas x, y)
    imagen_redimensionada = imagen.reshape(tamaño, tamaño)
    coordenadas = np.argwhere(imagen_redimensionada == 1)
    if len(coordenadas) == 0:
        return None
    centro_x = int(np.mean(coordenadas[:, 0]))
    centro_y = int(np.mean(coordenadas[:, 1]))
    return centro_x, centro_y

# Tamaño de la imagen (10x10 píxeles)
tamaño = 10

# Crear un patrón de círculo (aro "C")
patron_circulo = crear_patron_circulo(tamaño)

# Inicializar la red de Hopfield
red_hopfield = RedHopfield(tamaño * tamaño)

# Entrenar la red con el patrón de círculo
red_hopfield.entrenar([patron_circulo])

# Generar 3 imágenes ruidosas basadas en el patrón de círculo
imagenes_ruidosas = [generar_imagen_con_ruido(patron_circulo, nivel_ruido=0.3) for _ in range(3)]

# Probar la red con las imágenes ruidosas
for i, imagen_ruidosa in enumerate(imagenes_ruidosas):
    print(f"\nProcesando imagen {i+1}:")
    
    # Intentar recuperar el patrón limpio
    imagen_limpiada = red_hopfield.recordar(imagen_ruidosa)
    
    # Identificar si el patrón es un círculo y determinar sus coordenadas
    centro = encontrar_centro_circulo(imagen_limpiada, tamaño)
    
    if centro:
        print(f"Aro 'C' encontrado en la imagen, alineado. {i+1}. Coordenadas del centro: X = {centro[0]}, Y = {centro[1]}")
    else:
        print(f"No se encontró el aro 'C' en la imagen, no está alineado {i+1}.")
