import uuid
from dataclasses import dataclass

import requests
from chainlit import Message, on_message, user_session, on_chat_start


@dataclass
class ChatInfo:
    owner: str
    msg: str
    unique_id: str


@on_chat_start
def start():
    unique_id = str(uuid.uuid1())
    user_session.set('key', unique_id)


@on_message
def main(msg: str):
    unique_id = user_session.get('key')

    owner = 'seeker'
    SeekerChatInfo: ChatInfo = {
        'owner': owner,
        'msg': msg,
        'unique_id': unique_id
    }
    try:
        res = requests.post(url='http://mechat.westlake.ink:6001/v1/chat',
                            json=SeekerChatInfo)

        res = res.json()
        response = res['item']['msg']
        Message(content=response).send()
    except Exception as e:
        print(f'ERROR: {e}')
        Message(content='Server error, and try again later.').send()
