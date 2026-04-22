from typing import Dict, List

from pydantic import BaseModel


class TeamFeatures(BaseModel):
    legendarykills_sum: float
    timecc_sum: float
    timecc_mean: float
    timecc_rate: float
    firsttower: float
    firstinhib: float
    firstbaron: float
    firstdragon: float
    is_blue_side: float
    cc_per_min: float
    totunitshealed_sum: float
    totcctimedealt_sum: float
    visionscore_sum_diff_vs_opp: float
    item5_sum: float
    item6_sum: float
    trinket_sum: float
    assists_sum: float
    largestkillingspree_sum: float
    killingsprees_sum: float
    longesttimespentliving_sum: float
    doublekills_sum: float
    triplekills_sum: float
    quadrakills_sum: float
    pentakills_sum: float
    largestcrit_sum: float
    totheal_sum: float
    dmgselfmit_sum: float
    dmgtoturrets_sum: float
    turretkills_sum: float
    inhibkills_sum: float
    pinksbought_sum: float
    wardsbought_sum: float
    wardsplaced_sum: float
    wardskilled_sum: float
    trinket_mean: float
    kda_ratio: float
    gold_spent_ratio: float
    jungle_control_share: float
    assists_sum_diff_vs_opp: float
    dragonkills_diff_vs_opp: float
    baronkills_diff_vs_opp: float
    inhibkills_diff_vs_opp: float

class PredictRequest(BaseModel):
    features: TeamFeatures

class BatchPredictRequest(BaseModel):
    requests: List[PredictRequest]

class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    win_probability: float
    model_version: str

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]

class HealthResponse(BaseModel):
    status: str

class ModelInfoResponse(BaseModel):
    model_name: str
    run_name: str
    experiment_name: str
    features: List[str]


class FeatureImportanceResponse(BaseModel):
    importances: Dict[str, float]
