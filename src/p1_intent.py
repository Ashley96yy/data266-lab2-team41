"""Small, reference-free intent helpers for source scope and answer completeness."""
import re


def requested_document_scope(question):
    current=question.split(' | ',1)[0]
    # A document mentioned as a citation in quoted/untrusted instructions is
    # not an instruction to restrict retrieval to that document.
    patterns=[r'\b(?:according\s+to|from|in|using|use\s+only)\s+(?:the\s+)?(?:document|doc)\s*([1-3])\b',
              r'\b(?:what\s+does|what\s+did|does|did)\s+(?:document|doc)\s*([1-3])\s+(?:itself\s+)?(?:report|state|say|show)\b']
    return list(dict.fromkeys('DOC'+m[1] for pattern in patterns for m in re.finditer(pattern,current,re.I)))


def requested_periods(question):
    pattern=re.compile(r'\b((?:19|20)\d{2})\s*[-\u2013\u2014]\s*((?:19|20)\d{2})\b')
    consumed=[];periods=[]
    for m in pattern.finditer(question):
        nearby=question[max(0,m.start()-30):min(len(question),m.end()+20)].casefold()
        if 'average' in nearby or re.search(r'\bavg\b',nearby):
            periods.append((m.start(),f'{m[1]}-{m[2]} average'));consumed.append(m.span())
    for m in re.finditer(r'\b(?:19|20)\d{2}\b',question):
        if not any(a<=m.start()<b for a,b in consumed):periods.append((m.start(),m[0]))
    return list(dict.fromkeys(value for _,value in sorted(periods)))


def requires_explanation_or_multiple_tasks(question):
    return bool(re.search(r'\b(?:then\s+use|explain|why|separately|also\s+state|explicitly\s+revis(?:e|es|ed))\b',question,re.I))


def is_session_provenance_inquiry(question):
    return bool(re.search(r'\b(?:is|was|are|were|did|does|do)\s+(?:that|this|those|these)\b|\b(?:source|origin)\s+of\s+(?:that|this|the\s+(?:assumed|session|user))\b|\bwhere\s+did\s+(?:that|this)\b',question,re.I))


def is_counterfactual_comparison(question):
    return bool(re.search(r'\b(?:assum\w*|hypothetical|counterfactual)\b',question,re.I)
                and re.search(r'\b(?:compar\w*|higher|lower|above|below|difference|worse|better)\b',question,re.I))
