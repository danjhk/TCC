''' Implementa funções de uso geral
@package utils

@author Daniel Kim & Igor Nakamura
@date Date: 2020-04-17

@verbatim
@endverbatim 
'''

# -----------------------------
# Standard library dependencies
# -----------------------------
import re
from typing import List, Dict

# -------------------
# Third-party imports
# -------------------
import cv2
import glob

def get_weizmann_filepaths():
    """ Obtém um dicionário de dicionários referentes ao dataset Weizmann 
    como descrito em get_filepaths

    @return Um dicionário de dicionários
    """
    path_weiz = r'Weizmann/'
    person_weiz = 'daria'
    action_weiz = 'bend'
    names_list_weiz = create_names_list(path_weiz, action_weiz)
    actions_list_weiz = create_actions_list(path_weiz, person_weiz)    
    return(get_filepaths(path_weiz, names_list_weiz, actions_list_weiz))

def get_kth_filepaths():
    """ Obtém um dicionário de dicionários referentes ao dataset KTH 
    como descrito em get_filepaths

    @return Um dicionário de dicionários
    """
    path_kth = r'KTH/'
    person_kth = 'person01'
    action_kth = 'boxing_d1_uncomp'
    names_list_kth = create_names_list(path_kth, action_kth)
    actions_list_kth = create_actions_list(path_kth, person_kth)            
    return(get_filepaths(path_kth, names_list_kth, actions_list_kth))

def create_names_list(path, action):
    """ Obtém a lista dos nomes das pessoas que executam as ações 
    de um dataset específico 

    Args:
        path: caminho para para a pasta desejada. Ex: 'Weizmann/' ou 'KTH/'
        action: nome da ação. Ex: 'bend' ou 'handclapping_d4_uncomp'

    @return names_list: lista dos nomes das pessoas de um dataset. 
        Como todas as pessoas executam o mesmo conjunto de ações, essa 
        variável contêm todos os nomes das pessoas do dataset.
    
    Examples:
        >>> names_list = ['daria', 'shahar', 'lena', 'lyova', etc]
        >>> names_list = ['person01', 'person02', 'person03', etc]
    """
    names_list = []
    filenames = glob.glob(path+'*'+action+'*')
    pattern = re.compile("\/(.*?)\_")
    names_list = [re.search(pattern, filename).group(1) for filename in filenames]  
    return names_list

def create_actions_list(path, person):
    """ Obtém a lista de ações referentes a um dataset específico 

    Args:
        path: caminho para para a pasta desejada. Ex: 'Weizmann/' ou 'KTH/'
        person: pessoa que executa a ação. Ex: 'daria' ou 'person01'

    @return actions_list: lista de ações referentes a uma pessoa do dataset.
        Como uma única pessoa executa todas as ações do dataset, essa variável 
        contêm todas as ações do dataset.

    Examples:
        >>> actions_list = ['bend', 'wave1', 'wave2', 'walk', 'skip', etc]
        >>> actions_list = ['handclapping_d4_uncomp', 'boxing_d3_uncomp', etc]
    """
    actions_list = []
    filenames = glob.glob(path+person+'*')
    pattern = re.compile("\_(.*?)\.")
    actions_list = [re.search(pattern, filename).group(1) for filename in filenames]
    return actions_list

def get_filepaths(path, names_list, actions_list):
    """ Obtém os caminhos para os arquivos de video de um dataset específico

    Args: 
        path: caminho para para a pasta desejada. Ex: 'Weizmann/' ou 'KTH/'
        names_list: lista com os nomes das pessoas que executam a ação. Ex: 
            'daria' para Weizmann e 'person01'para KTH
        actions_list: lista com as ações. Ex: 'bend', 'walk', etc para Weizmann
            e 'walking_d1_uncomp', 'person23_boxing_d2_uncomp' para KTH

    @return filepaths: dicionário de dicionários cujos 'keys' do dicionário 
        mais externo são os nomes das pessoas. Os 'keys' dos dicionários mais
        internos são as ações associadas a uma pessoa específica. Por fim,
        os 'values' do dicionário são o caminho completo para o arquivo de vídeo.

    Examples: 
        >>> print(filepaths['daria']['bend']) 
        'Weizmann/daria_bend.avi'
        >>> print(filepaths['person01']['boxing_d1_uncomp']) 
        'KTH/person01_boxing_d1_uncomp'
    """

    person_actions_regex_dicts = {}  
    filepaths = {}
    for name in names_list:
        person_actions_regex_dicts[name] = \
            create_actions_regex_dict(name, actions_list)
        filepaths[name] = get_person_filepaths(path, 
                                               person_actions_regex_dicts[name])
    return filepaths

def create_actions_regex_dict(name, actions_list):
    """ Cria um dicionário de regex para buscar os nomes dos arquivos desejados

    Args:
        name: nome de uma pessoa. Ex: 'daria' ou 'person01'
        actions_list: lista de ações. Já descrito na função get_filepaths

    @return actions_regex_dict: dicionário cujos 'keys' são as ações contidas em 
            actions_list e cujos 'values' são regex que serão usados para
            procurar o nome do arquivo de vídeo

    Examples:
        >>> print(actions_regex_dict['bend']) 
        r'daria_bend*'
        >>> print(actions_regex_dict['running_d3_uncomp']) 
        r'person25_running_d3_uncomp*'
    """
    actions_regex_dict = {}
    for action in actions_list:
        actions_regex_dict[action] = r'%s_%s*' % (name,action)
    return actions_regex_dict

def get_person_filepaths(path, actions_regex_dict):
    """ Obtém os caminhos dos arquivos a partir do regex_dict recebido.

    Args:
        path: caminho para para a pasta desejada. Ex: 'Weizmann/' ou 'KTH/'
        actions_regex_dict: idém ao descrito em 'create_actions_regex_dict'

    @return filepaths: dicionário cujos 'keys' são as ações e cujos 'values'
            são os caminhos dos arquivos de vídeo contendo as ações 
            de uma pessoa específica

    Examples:
        >>> filepaths['bend'] = 'Weizmann/daria_bend.avi'
        >>> filepaths['boxing_d1_uncomp'] = 'KTH/person01_boxing_d1_uncomp'
    """
    filepaths = {}
    for key in actions_regex_dict:
        filepaths[key] = glob.glob(path+actions_regex_dict[key])
        if filepaths[key]:
            filepaths[key] = filepaths[key][0]
    return filepaths