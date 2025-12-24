from typing import List

from cti_analysis.models.scores import SimilarityScore, RerankResult


def run_reranking(cfg, scores: List[SimilarityScore]) -> List[RerankResult]:
    """
    IR-friendly reranking stub; replace with actual reranker logic.
    """
    return [RerankResult(query_id=s.query_id, ranked_targets=[s], meta={}) for s in scores]

