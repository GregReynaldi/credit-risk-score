import pytest
from pydantic import ValidationError
from app.schemas import CreditRequest


def test_credit_request_valid():
    data = {
        "person_age": 30,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "VENTURE",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.2,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3,
    }
    req = CreditRequest(**data)
    assert req.person_age == 30


def test_credit_request_age_boundary():
    with pytest.raises(ValidationError):
        CreditRequest(
            person_age=17,
            person_income=50000,
            person_home_ownership="RENT",
            person_emp_length=5.0,
            loan_intent="VENTURE",
            loan_grade="B",
            loan_amnt=10000,
            loan_int_rate=10.5,
            loan_percent_income=0.2,
            cb_person_default_on_file="N",
            cb_person_cred_hist_length=3,
        )


def test_credit_request_invalid_home_ownership():
    with pytest.raises(ValidationError):
        CreditRequest(
            person_age=30,
            person_income=50000,
            person_home_ownership="INVALID",
            person_emp_length=5.0,
            loan_intent="VENTURE",
            loan_grade="B",
            loan_amnt=10000,
            loan_int_rate=10.5,
            loan_percent_income=0.2,
            cb_person_default_on_file="N",
            cb_person_cred_hist_length=3,
        )

