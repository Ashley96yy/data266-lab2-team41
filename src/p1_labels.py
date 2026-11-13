"""Parse explicit singular/plural figure and table references in source order."""
import re

IDENTIFIER = r'([A-C12])\s*[-\u2013\u2014]\s*(\d+)(\s*\(corrected\))?'
HEAD = re.compile(r'\b(Figures?|Tables?)\s+' + IDENTIFIER + r'(?![\w-])', re.I)
TAIL = re.compile(r'\s*(?:,\s*(?:and\s+)?|and\s+|with\s+|versus\s+|vs\.?\s+|or\s+)(?:(Figures?|Tables?)\s+)?' + IDENTIFIER + r'(?![\w-])', re.I)


def find_explicit_visual_labels(query):
    if not isinstance(query,str):
        raise TypeError('query must be a string')
    found=[]
    occupied_until=-1
    for match in HEAD.finditer(query):
        if match.start()<occupied_until:
            continue
        kind='Figure' if match[1].casefold().startswith('figure') else 'Table'
        def label(groups,kind):
            _,prefix,number,corrected=groups
            return f'{kind} {prefix.upper()}-{number}'+(' (corrected)' if corrected else '')
        found.append(label(match.groups(),kind))
        position=match.end()
        while True:
            more=TAIL.match(query,position)
            if more is None:
                break
            if more[1]:kind='Figure' if more[1].casefold().startswith('figure') else 'Table'
            found.append(label(more.groups(),kind))
            position=more.end()
        occupied_until=position
    return list(dict.fromkeys(found))
