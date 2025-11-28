from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .container import services
from .schemas import CreditRequest, PredictionResponse
from .settings import STATIC_DIR, TEMPLATES_DIR

logger = logging.getLogger(__name__)

# Global readiness flag
app_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_ready
    
    logger.info("Starting application initialization...")
    logger.info("Loading models and explainability resources...")
    
    try:
        # Verify all services are loaded
        logger.info("✓ Preprocessor loaded")
        logger.info("✓ Predictor loaded")
        logger.info("✓ Explainer loaded")
        
        # Verify LLM is loaded
        if services.explainer.llm_pipeline is not None:
            logger.info("✓ LLM model loaded and ready")
        else:
            logger.warning("⚠ LLM model not available (will skip LLM explanations)")
        
        # Verify global insights are loaded
        if services.explainer.global_insights:
            insights_count = sum(
                len(v) if isinstance(v, dict) else 0 
                for v in services.explainer.global_insights.values()
            )
            logger.info(f"✓ Global explainability insights loaded ({insights_count} entries)")
        
        app_ready = True
        logger.info("✓ Application ready to accept requests")
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize application: {e}", exc_info=True)
        raise
    
    yield
    
    logger.info("Shutting down application...")


app = FastAPI(
    title="Fast Credit Risk Explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if not app_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "loading", "message": "Application is still initializing. Please wait..."}
        )
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    if app_ready:
        llm_status = "ready" if services.explainer.llm_pipeline is not None else "unavailable"
        return {
            "status": "ready",
            "llm": llm_status,
            "models_loaded": True,
        }
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "loading",
            "message": "Application is still initializing models and resources."
        }
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(payload: CreditRequest, include_llm: bool = False):
    if not app_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application is still initializing. Please wait and try again."
        )
    
    try:
        (
            probability,
            risk,
            lime_features,
            shap_features,
            risk_increasing,
            risk_decreasing,
            global_context,
            model_agreement,
            llm_text,
        ) = services.predict(payload.dict(), include_llm=include_llm)
    except Exception as exc:  # pragma: no cover - defensive guardrail
        raise HTTPException(status_code=500, detail=str(exc))

    return PredictionResponse(
        probability=probability,
        risk_level=risk,
        lime_features=lime_features,
        shap_features=shap_features,
        risk_drivers_increasing=risk_increasing,
        risk_drivers_decreasing=risk_decreasing,
        global_context=global_context,
        model_agreement=model_agreement,
        llm_summary=llm_text,
    )

