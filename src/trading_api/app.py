from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from trading_api.routers import routers
from fastapi.templating import Jinja2Templates
from pathlib import Path

app = FastAPI()


for router in routers:
    app.router.include_router(router=router)

static_dir = Path(__file__).parent / "static"
template_dir = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=template_dir)


@app.get("/report/{symbol}")
def server_app(request: Request, symbol: str):
    return templates.TemplateResponse(
        request=request, name="symbol.html", context={"symbol": symbol}
    )


@app.get("/")
def serve_home_page(request: Request):
    return templates.TemplateResponse(request=request, name="home.html", context={})
