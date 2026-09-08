# CTI/triple_extraction/notebook_eval.py
# triple_extraction/notebook_eval.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Triple = Tuple[str, str, str]

def _triple_text(t: Triple) -> str:
    s, p, o = t
    return f"{str(s).lower().strip()} {str(p).lower().strip()} {str(o).lower().strip()}"

def _vectorize_pairs(extracted: List[Triple], ground: List[Triple]):
    corpus = [ _triple_text(t) for t in extracted ] + [ _triple_text(t) for t in ground ]
    if not corpus:
        return np.zeros((len(extracted), 0)), np.zeros((len(ground), 0))
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(corpus)
    return X[:len(extracted)], X[len(extracted):]

def cti_hal_ground_truth(annotations: List[Dict], threat_actor: str) -> List[Triple]:
    triples: List[Triple] = []
    for ann in annotations:
        tech = ann.get("technique")
        meta = ann.get("metadata", {})
        tname = str(meta.get("technique_name", tech)).lower() if (meta.get("technique_name", tech)) else None
        tool  = meta.get("tool_name")
        tool_names = []
        if isinstance(tool, list):
            tool_names = [str(t).lower() for t in tool if t]
        elif tool:
            tool_names = [str(tool).lower()]
        if tool_names:
            for t in tool_names:
                if tname:
                    triples.append((t, "executes", tname))
        else:
            if threat_actor and tname:
                triples.append((str(threat_actor).lower(), "uses", tname))
    return triples

def evaluate_sample_vector_similarity(extracted: List[Triple], ground: List[Triple], threshold: float, debug: bool=False) -> Dict[str,Any]:
    if not extracted or not ground:
        return {"precision":0.0,"recall":0.0,"f1":0.0,"tp":0,"fp":len(extracted or []),"fn":len(ground or []),"hit":0}
    X_ext, X_gt = _vectorize_pairs(extracted, ground)
    if X_ext.shape[1] == 0 or X_gt.shape[1] == 0:
        return {"precision":0.0,"recall":0.0,"f1":0.0,"tp":0,"fp":len(extracted),"fn":len(ground),"hit":0}
    sims = cosine_similarity(X_ext, X_gt)
    tp, matched = 0, set()
    for i in range(sims.shape[0]):
        j = int(np.argmax(sims[i]))
        sc = sims[i, j]
        if sc >= threshold and j not in matched:
            tp += 1; matched.add(j)
    fp = len(extracted) - tp
    fn = len(ground) - tp
    P = tp/len(extracted) if extracted else 0.0
    R = tp/len(ground) if ground else 0.0
    F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
    if debug: print(f"[debug] tp={tp} fp={fp} fn={fn}")
    return {"precision":P,"recall":R,"f1":F1,"tp":tp,"fp":fp,"fn":fn,"hit":1 if tp>0 else 0}
