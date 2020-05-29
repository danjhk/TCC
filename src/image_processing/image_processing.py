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
import numpy as np
from skimage.measure import label, regionprops

def count_frames(video_cap):
    """ Conta a quantidade de frames de um video

    Args:
        video_cap: objeto VideoCapture

    @return num_frames: inteiro contendo a quantidade de frames

    Examples:
        count_frames(video_cap)
    \hidecallergraph
    """
    cap = video_cap
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    else:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return num_frames

def show_frames(video_cap):
    """ Mostra o conteúdo de um conjunto de frames ao usuário

    Args:
        video_cap: objeto VideoCapture
    
    @return None
    
    Examples:
        >>> show_frames(video_cap)
    \hidecallergraph
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


def stack_frames(frames_list, frames_per_stack):
    """ Agrupa os frames de um vídeo em listas que contém uma 
    quantidade de "frames_per_stack" de frames

    Args:
        frames_list: lista contendo uma sequência de frames
        frames_per_stack: quantidade de frames por lista
    
    @return stacked_frames_list: lista de listas de imagens, 
        correspondendo aos frames de um vídeo
    
    Examples:
        >>> stack_frames(frames_list, 9)
    \hidecallergraph
    """        
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

def scaling(frames_list, scale):
    """ Essa função realiza reduz a escala dos frames por um fator determinado.

    Args:
        frames_list: lista contendo uma sequência de frames
        scale: inteiro que indica a escala a ser aplicada no redimensionamento
            dos frames. Ex: 2, para escalar a imagem para a metade do tamanho original

    @return updated_list: uma lista de imagens, com os frames na escala especificada
            
    Examples:
        >>> scaling(frames_list, 2)
    \hidecallergraph
    """
    height , width, channels =  frames_list[0].shape
    dim = (int(height/scale), int(width/scale))
    updated_list = []
    
    for frame in frames_list: # itera para cada frame
      
        res = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) # downsampling

        updated_list.append(res)# adiciona o frame no stack
    return updated_list

def redim_weizmann(weiz_frames_list):
    """ Essa função redimensiona os frames da base Weizmann, para
        que possua uma proporção altura/largura = 1.33, igual aos frames da
        base KTH.

    Args:
        weiz_frames_list: lista de frames de um vídeo da base Weizmann
        
    @return weiz_frames_list: uma lista de imagens, com os frames 
        da base Weizmann com formato (135, 180, 3)
            
    Examples:
        >>> redim_weizmann(weiz_frames_list)
    \hidecallergraph
    """
    i = 0
    height = int(weiz_frames_list[0].shape[1])
    width = int(weiz_frames_list[0].shape[0])
    for frame in weiz_frames_list:
        weiz_frames_list[i] = frame[4:width-5, 0:height] # exclui as 4 primeiras colunas da esquerda 
                                                         # e 5 últimas da direita
        i += 1
    return weiz_frames_list

def video2list(video_cap):
    """ Essa função converte um objeto de vídeo para uma lista de frames

    Args:
        video_cap: sequência de frames de um vídeo

    @return frames_list: lista contendo os frames do vídeo

    Examples:
        >>> video2list(video_cap)
    \hidecallergraph
    """
    
    frames_list = []
    while(video_cap.isOpened()):
        # Captura frame por frame
        ret, frame = video_cap.read()
        if ret:
            frames_list.append(frame)
        else:
            break
    video_cap.release()
    return frames_list
 
def foreground_extraction(frames_list, lr, thr, hist_len):
    """ Essa função realiza a extração do plano frontal de um vídeo, e retorna
        uma lista de imagens correspondendo ao frames do mesmo.

    Args:
        frames_list: lista contendo uma sequência de frames
        lr: um número decimal que representa a taxa de aprendizado
          do algoritmo de subtração
        thr: um número inteiro que representa o limiar para definir a distância
          máxima ao qual um pixel ainda é considerado como pertencente ao fundo
        hist_len: um número inteiro que representa o histórico de frames considerados
          para o background model

    @return updated_list: uma lista de imagens, contendo os frames com o plano frontal extraído

    Examples:
        >>> foreground_extraction(video_cap, lr = 0.85, thr = 24, hist_len = 15)
    \hidecallergraph
    """

    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = False,
                                                varThreshold = thr, history = hist_len) # Gaussian mixture model para
                                                                        # extração do background
    height = int(frames_list[0].shape[0])
    width = int(frames_list[0].shape[1])
    horizontal_disp = int(width/4)
    n = 1
    updated_list = []
    last_good_column = int(width/2)
    for frame in frames_list:
        fgMask = backSub.apply(frame, learningRate = lr) # aplica o foregorund extractor
    #     fgMask3 = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB) # transforma máscara p 3 canais RGB

    #     new_frame = cv2.bitwise_and(frame, fgMask3) # aplica operador lógico AND bit a bit (pixel a pixel)
        lbl = label(fgMask)
        props = regionprops(lbl)

        max = 0
        for i in range(len(props)):
            if (props[i].area > max) and (props[i].area > 40):
                max = props[i].area
                column = int(props[i].centroid[1])

        mask = np.zeros(frame.shape, dtype=np.uint8)
        if (column-horizontal_disp <0 or column+horizontal_disp> width):
            column = last_good_column
        else:
            last_good_column = column
        
        mask[0:height,column-horizontal_disp:column+horizontal_disp] = 255
        new_frame = cv2.bitwise_and(frame, mask) # aplica operador lógico AND bit a bit (pixel a pixel)
        #cv2_imshow(fgMask3) # máscara
        #cv2_imshow(new_frame) # frame reduzido e com máscara aplicada
        if n >= 15: # a partir do 15° frame
            updated_list.append(new_frame) # adiciona o frame no stack
        n += 1
    return updated_list

def grayscale(frames_list):
    """ Esta função recebe uma lista de frames e
        retorna uma lista contendo os frames em tons de cinza
    Args:
        frames_list: lista contendo uma sequência de frames
        
    @return frames_grayscale: uma lista de imagens, com os frames em escalas de cinza

    Examples: 
        >>> grayscale(frames_list)
    \hidecallergraph
    """    
    stacked_grayscale = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) \
                            for frame in frames_list] 
        
    return stacked_grayscale

def compute_gradient(frames_list):
    """ Esta função recebe uma lista com frames e retorna 
    uma lista contendo os frames com seus gradientes na direção X e Y
    
    Args:
        frames_list: lista contendo uma sequência de frames
    
    Multiple return: 
        gradient_x_list: uma lista de imagens, contendo o gradiente
            na direção X dos frames contidos em frames_list
        gradient_y_list: uma lista de imagens, contendo o gradiente
            na direção Y dos frames contidos em frames_list

    """
    gradient_x = [cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5) for frame in frames_list]
    gradient_y = [cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5) for frame in frames_list]
    return gradient_x, gradient_y

def optical_flow(frames_list):
    """ Esta função recebe uma lista com frames empilhados e retorna lista contendo 
    os frames com seus fluxos ópticos na direção X e Y

    Args:
        frames_list: lista contendo uma sequência de frames
        
    Multiple return:
        opt_x_frames: uma lista de imagens, contendo o fluxo óptico
            na direção X, para cada frame
        opt_y_frames: uma lista de imagens, contendo o fluxo óptico
            na direção Y, para cada frame
    """    
    corner_detect_params = dict( maxCorners = 100,
                        qualityLevel = 0.2,
                        minDistance = 2,
                        blockSize = 7 )
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 0,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.015))
                                                                # (type, max_count, epsilon)
    # listas para armazenar os frames com fluxo óptico
    opt_x_frames = []
    opt_y_frames = []
    
    # inicializando parâmetros
    prev_frame = frames_list[0]
    opt_flow_x = np.zeros_like(prev_frame)
    opt_flow_y = np.zeros_like(prev_frame)
    p0 = cv2.goodFeaturesToTrack(prev_frame, mask = None, **corner_detect_params)
    
    # variáveis auxiliares
    opt_x = [[] for x in range(len(p0))]
    opt_y = [[] for x in range(len(p0))]
    x_initial = [0 for x in range(len(p0))]
    y_initial = [0 for x in range(len(p0))]

    for i in range(1, len(frames_list)):
        cp_frame = frames_list[i].copy()
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, cp_frame, p0, None, **lk_params)

        tracks = []
        n = 0
      
        for (x_i, y_i), (x0, y0) in zip(p1[st==1].reshape(-1, 2), p0[st==1].reshape(-1,2)):
            if i == 1: # 2o frame/1a iteração
                x_initial[n] = x_i
                y_initial[n] = y_i
                if n == 0: # primeiro ponto, cria um frame vazio
                    opt_flow_initial = np.zeros_like(prev_frame)
                cv2.circle(opt_flow_initial, (x0, y0), 2, (255,255,255), -1) # marca o ponto inicial

            opt_x[n].append((x_i, y_initial[n]))
            opt_y[n].append((x_initial[n], y_i))
            tracks.append((x_i,y_i))

            # cv2.circle(cp_frame, (x_i, y_i), 2, (255, 255, 255), -1) #visualizar o resultado do tracking
            cv2.circle(opt_flow_x, (x_i, y_initial[n]), 2, (255, 255, 255), -1)
            cv2.circle(opt_flow_y, (x_initial[n], y_i), 2, (255, 255, 255), -1)
            
            n+=1
            
        if i == 1: # 2o frame/1a iteração - insere frames com pontos iniciais
            opt_x_frames.append(opt_flow_initial)
            opt_y_frames.append(opt_flow_initial)

        # desenha a linha do fluxo óptico
        cv2.polylines(opt_flow_x, [np.int32(x) for x in opt_x], True, (255, 255, 255), 2)
        cv2.polylines(opt_flow_y, [np.int32(y) for y in opt_y], True, (255, 255, 255), 2)
        
        # armazena os frames com o fluxo óptico
        opt_x_frames.append(opt_flow_x)
        opt_y_frames.append(opt_flow_y)
        #opt_x_frames.append(cp_frame)
        
        # reinicializa a matriz de fluxo óptico para o próximo frame
        opt_flow_x = np.zeros_like(prev_frame)
        opt_flow_y = np.zeros_like(prev_frame)
        
        prev_frame = frames_list[i].copy()
        p0 = np.float32([tr for tr in tracks]).reshape(-1,1,2)
        
    
    return opt_x_frames, opt_y_frames

def stack_channels(gray_channel, gradient_x_channel, gradient_y_channel, opt_x_channel, opt_y_channel):
    stacked_channels = []
    stacked_channels.append(gray_channel)
    stacked_channels.append(gradient_x_channel)
    stacked_channels.append(gradient_y_channel)
    stacked_channels.append(opt_x_channel)
    stacked_channels.append(opt_y_channel)
    return np.array(stacked_channels)

def preprocess(filepath, dataset, stacks_per_list):

    cap = cv2.VideoCapture(filepath)

    frames_list = video2list(cap)
    if dataset == 'Weizmann':
        frames_list = redim_weizmann(frames_list)
        scaled_frames = scaling(frames_list, 2.25)
    elif dataset == 'KTH':
        scaled_frames = scaling(frames_list, 2)
    frames_fg = foreground_extraction(scaled_frames, lr = 0.85, thr = 24, hist_len = 15)

    gray_frames = grayscale(frames_fg)

    gradient_x, gradient_y = compute_gradient(gray_frames)

    optical_flow_x, optical_flow_y = optical_flow(gray_frames)

    stacked_gray = stack_frames(gray_frames, stacks_per_list)

    stacked_gradient_x = stack_frames(gradient_x, stacks_per_list)
    stacked_gradient_y = stack_frames(gradient_y, stacks_per_list)

    stacked_optical_flow_x = stack_frames(optical_flow_x, stacks_per_list)
    stacked_optical_flow_y = stack_frames(optical_flow_y, stacks_per_list)

    preprocessed_frames_tuple = (stacked_gray, stacked_gradient_x, stacked_gradient_y, stacked_optical_flow_x, stacked_optical_flow_y)
    return preprocessed_frames_tuple
