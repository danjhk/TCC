''' Implementa funções de processamento de imagens
@package image_processing

@author Daniel Kim & Igor Nakamura
@date Date: 2020-04-17

@verbatim
@endverbatim 
'''


# -----------------------------
# Standard library dependencies
# -----------------------------
from typing import List, Dict

# -------------------
# Third-party imports
# -------------------
import cv2
from google.colab.patches import cv2_imshow

def count_video_frames(video_cap):
    """ Conta a quantidade de frames de um video

    Args:
        video_cap: objeto VideoCapture

    @return num_frames: inteiro contendo a quantidade de frames

    Examples:
        >>> count_video_frames(video_cap)
        '84'
    """
    cap = video_cap
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    else:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return num_frames

def show_video_frames(video_cap):
    """ Mostra o conteúdo de um conjunto de frames ao usuário

    Args:
        video_cap: objeto VideoCapture
    
    @return None
    
    Examples:
        >>> show_video_frames(video_cap)
    """
    
    cap = video_cap
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2_imshow(frame)
        # Pressione 'Q' no teclado para sair
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        else: 
            break
    # Libera o objeto VideoCapture
    cap.release()
    # Fecha todos os frames
    cv2.destroyAllWindows()

def stack_video_frames(video_cap, frames_per_stack):
    """ Agrupa os frames de um vídeo em listas que contém uma 
    quantidade de "frames_per_stack" de frames

    Args:
        video_cap: sequência de frames a ser agrupada
        frames_per_stack: quantidade de frames por lista
    
    @return stacked_frames_list: lista de listas de imagens, 
        correspondendo aos frames de um vídeo
    
    Examples:
        >>> stack_video_frames(video_cap, 9)
    """
    
    frames_list = []
    while(video_cap.isOpened()):
    # Captura frame por frame
      ret, frame = video_cap.read()
      if ret == True:
        frames_list.append(frame)    
      else: 
        break
        
    num_frames = len(frames_list)
    num_stacks = num_frames - frames_per_stack + 1
    first_frame_of_nth_stack = 0
    last_frame_of_nth_stack = frames_per_stack  
    stacked_frames_list = []

    for n in range(num_stacks): # n representa o enésimo stack
      stacked_frames_list.append(
          frames_list[first_frame_of_nth_stack:last_frame_of_nth_stack])
      first_frame_of_nth_stack += 1
      last_frame_of_nth_stack += 1
    return stacked_frames_list

def preprocess(stacked_frames_list, scale):
    """ Essa função realiza o pré-processamento para um conjunto de frames. 
    Especificamente, reduz a escala dos frames e subtrai o seu background.
    <br>
    Args:
        stacked_frames_list: uma lista de imagens, correspondendo
            aos frames de um vídeo
        scale: inteiro que indica a escala a ser aplicada no redimensionamento
            dos frames. Ex: 2, para escalar a imagem para a metade do tamanho original

    @return stack_list: lista de imagens, com os frames na escala
        especificada e com foreground extraído
    """

    stack_list = [] # lista para armazenar os frames processados

    for stacked_frames in stacked_frames_list: # itera para cada stack de frames
      backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = False,
                                                varThreshold = 5) # Gaussian mixture model para
                                                                        # extração do background
      frame_list = [] # lista para armazenar os frames pré-processados

      for frame in stacked_frames: # itera para cada frame dentro do stack

        height , width, channels =  frame.shape
        dim = (int(height/scale), int(width/scale))
        resize = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) # downsampling

        fgMask = backSub.apply(resize) # aplica o foregorund extractor
        fgMask3 = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB) # transforma máscara p 3 canais RGB

        new_frame = cv2.bitwise_and(resize, fgMask3) # aplica operador lógico AND bit a bit (pixel a pixel)

        cv2_imshow(fgMask3) # máscara
        cv2_imshow(new_frame) # frame reduzido e com máscara aplicada
        frame_list.append(new_frame) # adiciona o frame no stack

      stack_list.append(frame_list) # adiciona o stack na lista
    
    return stack_list

def grayscale(stacked_frames_list):
    """ Esta função recebe uma lista com frames empilhados e retorna listas 
    contendo os frames agrupados em stacks, em tons de cinza

    Args:
        stacked_frames_list: uma lista de imagens, correspondendo
            aos frames de um vídeo
        
    @return stack_grayscale: lista de imagens, com os frames em escalas de cinza
    """    
    stacked_grayscale = [[cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) \
                            for frame in stacked_frames] \
                            for stacked_frames in stacked_frames_list]      

def compute_gradient(stacked_frames_list):
    """ Esta função recebe uma lista com frames empilhados e retorna 
    listas contendo os frames agrupados em stacks, com seus 
    gradientes na direção X e Y
    
    Args:
        stacked_frames_list: uma lista de imagens, correspondendo aos frames de um vídeo
    
    Multiple return: 
        stacked_gradient_x: uma lista de listas, contendo o gradiente
            na direção X dos frames contidos em stacked_frames_list
        stacked_gradient_y: uma lista de listas, contendo o gradiente
            na direção Y dos contidos em stacked_frames_list
    <br>
    """
    stacked_gradient_x = [[cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5) \
                           for frame in stacked_frames] \
                           for stacked_frames in stacked_frames_list]  
    stacked_gradient_y = [[cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5) \
                           for frame in stacked_frames] \
                           for stacked_frames in stacked_frames_list]
    return stacked_gradient_x, stacked_gradient_y

def optical_flow(stacked_frames_list):
    """ Esta função recebe uma lista com frames empilhados e retorna listas contendo 
    os frames agrupados em stacks, com seus fluxos ópticos na direção X e Y

    Args:
        stacked_frames_list: uma lista de imagens, correspondendo
            aos frames de um vídeo
        
    Multiple return:
        stacked_optical_x: uma lista de listas, contendo a posição X
            do fluxo óptico para cada frame
        stacked_optical_y: uma lista de listas, contendo a posição Y
            do fluxo óptico para cada frame
    """    
    corner_detect_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 0,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    stacked_optical_x, stacked_optical_y = []


    prev_frame = cv2.cvtColor(stacked_frames_list[0][0], cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_frame, mask = None, **corner_detect_params)

    x = []
    y = []

    # primeiro stack
    for i in range(len(stacked_frames_list[0]-1)):
      frame = cv2.cvtColor(stacked_frames_list[0][i+1])

      p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, p0, None, **lk_params)
      
      good_new = p1[st==1] # Select good points

      x_new, y_new = p1.ravel()

      x.append(x_new)
      y.append(y_new)
      
      prev_frame = frame.copy()
      p0 = good_new.reshape(-1,1,2)

    
    stacked_optical_x.append(x)
    stacked_optical_y.append(y)

    # stacks restantes
    for i in range(1, len(stacked_frames_list)):
      x = []
      y = []
      for stacked_frames in stacked_frames_list[i]:
        frame = cv2.cvtColor(stacked_frames_list[0][i+1])

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, p0, None, **lk_params)
      
        good_new = p1[st==1] # Select good points

        x_new, y_new = p1.ravel()

        x.append(x_new)
        y.append(y_new)

        prev_frame = frame.copy()
        p0 = good_new.reshape(-1,1,2)

      stacked_optical_x.append(x)
      stacked_optical_y.append(y)

