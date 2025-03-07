from fastapi import FastAPI
from ..frontend.src import models
from .databse import engine
from. routers import post , user,auth , vote
from .config import settings
from app import schema  



#print(settings.database_password)


models.Base.metadata.create_all(bind = engine)

app = FastAPI()



app.include_router(post.router)
app.include_router(user.router)
app.include_router(auth.router)
app.include_router(vote.router)



@app.get("/")
async def root():
    return{"message":"Hello welcome to mt API....."}

 