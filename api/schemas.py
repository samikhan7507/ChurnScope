"""
schemas.py — Strict Pydantic v2 request and response contracts for the ChurnScope API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
MultipleLinesType = Literal["Yes", "No", "No phone service"]
InternetServiceType = Literal["DSL", "Fiber optic", "No"]
InternetAddonType = Literal["Yes", "No", "No internet service"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodType = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
RiskTierType = Literal["Low", "Medium", "High", "Critical"]

INTERNET_ADDON_FIELDS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)


class CustomerFeatures(BaseModel):
    """Validated telecom customer profile matching the 19-feature IBM Telco schema."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "customerID": "CUST-12345",
                "gender": "Female",
                "SeniorCitizen": "0",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.5,
                "TotalCharges": 286.5,
            }
        },
    )

    customerID: str | None = Field(default=None, max_length=64)
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: Literal["0", "1"] = Field(..., description="'1' if senior citizen, else '0'")
    Partner: YesNo = Field(..., description="Whether customer has a partner")
    Dependents: YesNo = Field(..., description="Whether customer has dependents")
    tenure: int = Field(..., ge=0, le=120, description="Months with the provider (0–120)")
    PhoneService: YesNo = Field(..., description="Whether customer subscribes to phone service")
    MultipleLines: MultipleLinesType = Field(..., description="Multiple phone lines status")
    InternetService: InternetServiceType = Field(..., description="Internet connection tier")
    OnlineSecurity: InternetAddonType = Field(..., description="Online security add-on")
    OnlineBackup: InternetAddonType = Field(..., description="Online backup add-on")
    DeviceProtection: InternetAddonType = Field(..., description="Device protection add-on")
    TechSupport: InternetAddonType = Field(..., description="Technical support add-on")
    StreamingTV: InternetAddonType = Field(..., description="Streaming TV add-on")
    StreamingMovies: InternetAddonType = Field(..., description="Streaming movies add-on")
    Contract: ContractType = Field(..., description="Subscription contract term")
    PaperlessBilling: YesNo = Field(..., description="Whether billing is paperless")
    PaymentMethod: PaymentMethodType = Field(..., description="Customer payment method")
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0, description="Current monthly bill (USD)")
    TotalCharges: float = Field(..., ge=0.0, le=100000.0, description="Cumulative charges (USD)")

    @model_validator(mode="after")
    def validate_service_consistency(self) -> CustomerFeatures:
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError("MultipleLines must be 'No phone service' when PhoneService is 'No'.")
        if self.PhoneService == "Yes" and self.MultipleLines == "No phone service":
            raise ValueError(
                "MultipleLines cannot be 'No phone service' when PhoneService is 'Yes'."
            )
        if self.InternetService == "No":
            invalid = [
                f for f in INTERNET_ADDON_FIELDS if getattr(self, f) != "No internet service"
            ]
            if invalid:
                raise ValueError(
                    f"Fields {invalid} must be 'No internet service' when InternetService is 'No'."
                )
        else:
            invalid = [
                f for f in INTERNET_ADDON_FIELDS if getattr(self, f) == "No internet service"
            ]
            if invalid:
                raise ValueError(
                    f"Fields {invalid} cannot be 'No internet service' when InternetService is "
                    f"{self.InternetService!r}."
                )
        return self


class PredictionResponse(BaseModel):
    """Single-customer churn prediction output."""

    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: Literal[0, 1]
    churn_label: Literal["Churn", "No Churn"]
    risk_tier: RiskTierType
    confidence: float = Field(..., ge=0.0, le=1.0)
    threshold_used: float = Field(..., ge=0.0, le=1.0)
    customer_id: str | None = None


class ShapFeatureContribution(BaseModel):
    """Individual feature contribution computed via SHAP TreeExplainer."""

    feature: str
    display_name: str
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class ExplainPredictionResponse(PredictionResponse):
    """Prediction paired with local TreeExplainer SHAP values and retention actions."""

    base_value: float
    shap_values: list[ShapFeatureContribution]
    recommendations: list[str]


class BatchPredictRequest(BaseModel):
    """Batch scoring request payload."""

    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Aggregated and per-customer batch prediction results."""

    total: int
    churn_count: int
    churn_rate: float
    predictions: list[PredictionResponse]
    summary: dict[str, Any]


class HealthResponse(BaseModel):
    """API and ML model operational health report."""

    status: Literal["healthy", "degraded"]
    model_version: str
    algorithm: str | None = None
    is_calibrated: bool = False
    cv_roc_auc: float | None = None
    val_roc_auc: float | None = None
    n_features: int | None = None
    trained_at: str | None = None
    threshold: float
    is_model_loaded: bool


class ErrorResponse(BaseModel):
    """Standardized error envelope (never leaks internal stack traces)."""

    error: str
    request_id: str | None = None
    detail: str | None = None
