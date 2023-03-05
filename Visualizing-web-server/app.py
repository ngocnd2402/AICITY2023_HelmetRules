from fastapi import FastAPI, Request, Form, Depends, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from glob import glob
import json 
import uvicorn

WEBM_BBOX_VIDEO_PATH = 'static/data/bbox-video/webm'

app = FastAPI()

templates = Jinja2Templates(directory='templates')
app.mount("/static", StaticFiles(directory="static"), name="static")

videos = sorted(os.listdir('static/data/bbox-video/webm'))
n_video = len(videos)
batch_size = 20
n_batch = n_video // batch_size + (n_video % batch_size != 0)

print(f'Total videos: {n_video}')
print(f'Total batches: {n_batch}')

@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    context = {
        'request': request,
        'videos': videos[0:batch_size],
        'batch_id': 0,
        'last_batch_id': n_batch - 1
    }

    return templates.TemplateResponse('index.html', context)

@app.get('/{batch_id}', response_class=HTMLResponse)
def get_video(*, request: Request, batch_id: int):
    context = {
        'request': request,
        'videos': videos[batch_id * batch_size:(batch_id + 1) * batch_size],
        'batch_id': batch_id,
        'last_batch_id': n_batch - 1
    }

    return templates.TemplateResponse('index.html', context)


if __name__ == '__main__':
    uvicorn.run('app:app', port=6969, host='localhost', reload=True)