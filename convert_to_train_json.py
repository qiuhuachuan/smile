import os
import ujson
from typing import List
import copy

from utils.util import get_dialog, split_into_sessions


def get_ctx_response(sessions: List[List[str]]):
    small_train = []
    for session in sessions:
        session_copy = copy.deepcopy(session)

        while len(''.join(session_copy)) > 2048:
            # when the current total length exceeds 2048,
            # we need to pop the first element
            session_copy.pop(0)

        history = session_copy[:-1]
        supporter_msg = session_copy[-1]

        context = ''.join(history) + supporter_msg[:4]
        response = supporter_msg[4:]
        example = {'instruction': '', 'input': context, 'output': response}
        small_train.append(example)
    return small_train


if __name__ == '__main__':
    method = 'smile'
    full_train = []
    file_range = 56032
    for idx in range(file_range):
        dialog = get_dialog(method=method, idx=idx)

        sessions = split_into_sessions(dialog=dialog)
        small_train = get_ctx_response(sessions=sessions)
        full_train += small_train

    print('full length of training examples', len(full_train))  # 355733
    os.makedirs(f'./out/{method}', exist_ok=True)
    with open(f'./out/{method}/train.json', 'w', encoding='utf-8') as f:
        ujson.dump(full_train, f, ensure_ascii=False, indent=2)

    print('DONE')
