from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, conint, confloat


class CreditRequest(BaseModel):
    person_age: conint(ge=18, le=100)
    person_income: conint(ge=1000)
    person_home_ownership: str = Field(pattern="^(RENT|OWN|MORTGAGE|OTHER)$")
    person_emp_length: confloat(ge=0, le=60)
    loan_intent: str
    loan_grade: str
    loan_amnt: conint(ge=500, le=1000000)
    loan_int_rate: confloat(ge=0, le=100)
    loan_percent_income: confloat(ge=0, le=1)
    cb_person_default_on_file: str = Field(pattern="^(Y|N)$")
    cb_person_cred_hist_length: conint(ge=0, le=50)


class LimeFeature(BaseModel):
    feature: str
    impact: float


class ShapFeature(BaseModel):
    feature: str
    shap_value: float


class RiskDriver(BaseModel):
    feature: str
    impact: float
    direction: str


class GlobalContext(BaseModel):
    feature: str
    global_shap_rank: Optional[int] = None
    global_lime_rank: Optional[int] = None
    sensitivity: Optional[float] = None


class PredictionResponse(BaseModel):
    probability: float
    risk_level: str
    lime_features: List[LimeFeature]
    shap_features: Optional[List[ShapFeature]] = None
    risk_drivers_increasing: Optional[List[RiskDriver]] = None
    risk_drivers_decreasing: Optional[List[RiskDriver]] = None
    global_context: Optional[List[GlobalContext]] = None
    model_agreement: Optional[dict] = None
    llm_summary: Optional[str] = None

