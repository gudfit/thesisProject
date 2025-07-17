# scripts/ood_hard_utils.py
import re,hashlib,random,spacy
_nlp=None
def _nlp_get():
    global _nlp
    if _nlp is None:
        try:_nlp=spacy.load("en_core_web_sm")
        except Exception:_nlp=False
    return _nlp
_stop={"the","a","an","and","or","to","of","in","on","at","for","with","by","is","are","was","were"}
def _clauses(t):
    return re.split(r"[,;:.?!]",t)
def _hashint(s):
    return int(hashlib.md5(s.encode()).hexdigest(),16)
def mask_and_truncate(texts,k=8):
    nlp=_nlp_get();out=[]
    for t in texts:
        if not t or not t.strip():
            out.append(t);continue
        mt=t
        if nlp:
            d=nlp(t);repls=[]
            for e in d.ents:repls.append((e.start_char,e.end_char,"<%s>"%e.label_))
            if repls:
                parts=[];last=0
                for s,e,r in repls:
                    parts.append(mt[last:s]);parts.append(r);last=e
                parts.append(mt[last:]);mt="".join(parts)
        toks=re.findall(r"\S+",mt)
        out.append(" ".join(toks[:k]))
    return out
def mask_shuffle_trunc(texts,k=6,seed_offset=0):
    nlp=_nlp_get();out=[]
    for t in texts:
        if not t or not t.strip():
            out.append(t);continue
        mt=t
        if nlp:
            d=nlp(t);repls=[]
            for e in d.ents:repls.append((e.start_char,e.end_char,"<%s>"%e.label_))
            if repls:
                parts=[];last=0
                for s,e,r in repls:
                    parts.append(mt[last:s]);parts.append(r);last=e
                parts.append(mt[last:]);mt="".join(parts)
        cl=_clauses(mt);h=_hashint(mt)+seed_offset;random.seed(h);random.shuffle(cl);mt=" ".join(cl).strip()
        toks=[w for w in re.findall(r"\S+",mt) if (w.lower() not in _stop or w.startswith("<"))]
        if len([w for w in toks if w.startswith("<")])<2:toks.insert(0,"<ORG>")
        out.append(" ".join(toks[:k]))
    return out

