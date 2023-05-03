import ujson
from typing import List


def get_dialog(method: str, idx: int) -> List[str]:
    '''
    get a dialog, a list of string
    '''
    with open(f'./data/{method}/{idx}.json', 'r', encoding='utf-8') as f:
        dialog = ujson.load(f)

    return dialog


def split_into_sessions(dialog: List[str]):
    '''
    split the dialog into many sessions,\n
    each session ends with `支持者 (supporter)`.\n
    As a result, we can get many `(context, response)` pairs,\n
    wherein the context ends with `求助者 (help-seeker)`
    '''
    sessions = []
    for index, element in enumerate(dialog):
        if element[:3] == '支持者':
            sessions.append(dialog[:index + 1])
    return sessions