import requests as rq
import logging
import json

PORT = 5000
headers = {
'Content-Type': 'application/json',
}

def positive_text():
    json_data = {
        'text': "Try talking with ChatGPT, our new AI system which is optimized for dialogue. Your feedback will help us improve it. https://t.co/sHDm57g3Kr",
    }
    resp = rq.post(headers=headers, url='http://localhost:{}/predict'.format(PORT), json=json_data)
    resul = json.loads(resp.text.replace('\n', ''))
    return str(resul['prediction'])

def negative_text():
    json_data = {
        'text': "This is what ChatGPT AI by @OpenAI has to say about Tinubu's presidential aspiration despite his frail health and history with corruption. https://t.co/aLt4SUYzvP",
    }
    resp = rq.post(headers=headers, url='http://localhost:{}/predict'.format(PORT), json=json_data)
    resul = json.loads(resp.text.replace('\n', ''))
    return str(resul['prediction'])

def neutral_text():
    json_data = {
        'text': "ChatGPT: Optimizing Language Models for Dialogue https://t.co/GLEbMoKN6w #AI #MachineLearning #DataScience #ArtificialIntelligence\n\nTrending AI/ML Article Identified &amp; Digested via Granola; a Machine-Driven RSS Bot by Ramsey Elbasheer https://t.co/RprmAXUp34",
    }
    resp = rq.post(headers=headers, url='http://localhost:{}/predict'.format(PORT), json=json_data)
    resul = json.loads(resp.text.replace('\n', ''))
    return str(resul['prediction'])

if __name__ == '__main__':
    logging.info("Positive Test:")
    assert 'Positive Text' == positive_text()

    logging.info("Negative Test")
    assert 'Negative Text' == negative_text()

    logging.info("Neutral Test")
    assert 'Neutral Text' == neutral_text()