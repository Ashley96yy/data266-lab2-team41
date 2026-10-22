"""Generic structured-evidence extraction and resolution for Part 1.

The module parses tables and revision notices from the read-only PDFs into
normalized facts. It resolves direct lookups, changes, differences, forecast
errors, source conflicts, and precise unsupported lookups without using
question IDs or storing answers from the practice set.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import re

import pdfplumber


NOT_FOUND_ANSWER = (
    "The retrieved evidence does not support an answer to this question."
)

TRUST_RANK = {"low": 0, "medium": 1, "high": 2}

ENTITY_PATTERNS = (
    (
        "emerging and developing economies",
        re.compile(r"\bemerging(?:\s+and\s+developing)?\s+econom(?:y|ies)\b", re.I),
    ),
    (
        "advanced economies",
        re.compile(r"\badvanced[-\s]+econom(?:y|ies)\b", re.I),
    ),
    ("United States", re.compile(r"\b(?:United States|U\.?S\.?)\b", re.I)),
    ("Euro area", re.compile(r"\bEuro(?:pean)?\s+area\b", re.I)),
    ("China", re.compile(r"\bChina(?:'s)?\b", re.I)),
    ("India", re.compile(r"\bIndia(?:'s)?\b", re.I)),
    ("Brazil", re.compile(r"\bBrazil(?:'s)?\b", re.I)),
    ("World", re.compile(r"\bworld(?:\s+economy|\s+output)?\b", re.I)),
)

METRIC_PATTERNS = (
    (
        "labour force participation rate",
        re.compile(r"\blabou?r[-\s]+force\s+participation(?:\s+rate)?\b", re.I),
    ),
    (
        "budget balance",
        re.compile(r"\b(?:budget\s+balance|federal\s+deficit|budget\s+deficit)\b", re.I),
    ),
    ("revenues", re.compile(r"\b(?:federal\s+)?revenues?\b", re.I)),
    ("spending", re.compile(r"\b(?:federal\s+)?spending\b", re.I)),
    (
        "output growth",
        re.compile(
            r"\b(?:output[-\s]+(?:growth|change|contract(?:ion|ed)?)|"
            r"GDP[-\s]+growth|economic[-\s]+growth)\b",
            re.I,
        ),
    ),
    (
        "unemployment rate",
        re.compile(r"\b(?:registered\s+urban\s+)?unemployment(?:\s+rate)?\b", re.I),
    ),
    ("inflation rate", re.compile(r"\b(?:headline\s+CPI|core\s+CPI|inflation)(?:\s+rate)?\b", re.I)),
)

TABLE_LABEL_RE = re.compile(r"\bTable\s+(?:[12A-C])-\d+\b", re.I)
VISUAL_LABEL_RE = re.compile(
    r"\b(?:Figure|Table)\s+(?:[12A-C])-\d+(?:\s*\(corrected\))?\b",
    re.I,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _clean(value):
    if value is None:
        return ""
    return " ".join(
        str(value)
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .split()
    )


def _trust_for_page(document_metadata, page_number):
    trust_level = document_metadata["trust_level"]
    for override in document_metadata.get("section_trust_overrides", []):
        if page_number in override.get("pages", []):
            trust_level = override["trust_level"]
    return trust_level


def _mentions(text, patterns):
    matches = []
    for canonical, pattern in patterns:
        for match in pattern.finditer(text):
            matches.append((match.start(), canonical))
    matches.sort()
    return list(dict.fromkeys(canonical for _, canonical in matches))


def _entity(text):
    matches = _mentions(text, ENTITY_PATTERNS)
    return matches[0] if matches else None


def _metric(text):
    matches = _mentions(text, METRIC_PATTERNS)
    return matches[0] if matches else None


def _table_titles(page_text):
    titles = []
    for line in page_text.splitlines():
        line = _clean(line)
        if TABLE_LABEL_RE.match(line):
            titles.append(line)
    return titles


def _period_spec(value):
    text = _clean(value)
    if not text:
        return None
    years = YEAR_RE.findall(text)
    if len(years) >= 2 and ("average" in text.casefold() or "-" in text):
        normalized = f"{years[0]}-{years[1]}"
        if "average" in text.casefold():
            normalized += " average"
        return {"period": normalized, "kind": "reported"}
    if not years:
        return None
    folded = text.casefold()
    if "forecast" in folded or "*" in text:
        kind = "forecast"
    elif "outturn" in folded:
        kind = "outturn"
    else:
        kind = "reported"
    return {"period": years[0], "kind": kind}


def _numbers(value):
    return [float(match.group(0)) for match in NUMBER_RE.finditer(_clean(value))]


def _table_id(title):
    match = TABLE_LABEL_RE.search(title or "")
    return match.group(0).title().replace("A-", "A-").replace("B-", "B-") if match else None


def _fact(record, entity, metric, period, value, value_kind, source_text):
    return {
        "doc_id": record["doc_id"],
        "page": record["page"],
        "citation": f"{record['doc_id']}:p{record['page']}",
        "table_id": record.get("table_id"),
        "trust_level": record["trust_level"],
        "entity": entity,
        "metric": metric,
        "period": period,
        "value": float(value),
        "value_kind": value_kind,
        "source_text": _clean(source_text),
    }


def _facts_from_wide_table(record, rows, header_index):
    header = rows[header_index]
    period_columns = [
        (index, _period_spec(value))
        for index, value in enumerate(header)
        if _period_spec(value) is not None
    ]
    if len(period_columns) < 2:
        return []
    first_period_column = period_columns[0][0]
    title_entity = _entity(record["title"])
    title_metric = _metric(record["title"])
    current_entity = title_entity
    facts = []

    for row in rows[header_index + 1 :]:
        row = list(row) + [""] * max(0, len(header) - len(row))
        prefix = [_clean(value) for value in row[:first_period_column]]
        if not any(prefix):
            continue
        joined_prefix = " ".join(prefix)
        if joined_prefix.casefold().startswith(("source", "* forecast")):
            continue

        if first_period_column >= 2:
            if prefix[0]:
                current_entity = _entity(prefix[0]) or prefix[0]
            entity = current_entity
            metric = _metric(prefix[1]) if len(prefix) > 1 else None
        else:
            label = prefix[0]
            label_metric = _metric(label)
            label_entity = _entity(label)
            if label_metric:
                entity = title_entity
                metric = label_metric
            elif label_entity:
                entity = label_entity
                metric = title_metric
            else:
                entity = title_entity
                metric = title_metric

        if not entity or not metric:
            continue
        for column_index, period in period_columns:
            if column_index >= len(row):
                continue
            values = _numbers(row[column_index])
            if not values:
                continue
            if len(values) >= 2 and period["kind"] == "reported":
                facts.append(
                    _fact(
                        record,
                        entity,
                        metric,
                        period["period"],
                        values[0],
                        "outturn",
                        " ".join(row),
                    )
                )
                facts.append(
                    _fact(
                        record,
                        entity,
                        metric,
                        period["period"],
                        values[1],
                        "forecast",
                        " ".join(row),
                    )
                )
            else:
                facts.append(
                    _fact(
                        record,
                        entity,
                        metric,
                        period["period"],
                        values[0],
                        period["kind"],
                        " ".join(row),
                    )
                )
    return facts


def _facts_from_row_table(record, rows):
    header = rows[0]
    entity = _entity(record["title"]) or _entity(_clean(header[0]))
    if not entity:
        return []
    metric_columns = [
        (index, _metric(value))
        for index, value in enumerate(header)
        if index > 0 and _metric(value) is not None
    ]
    if not metric_columns:
        return []
    facts = []
    for row in rows[1:]:
        period = _period_spec(row[0] if row else "")
        if period is None:
            continue
        for column_index, metric in metric_columns:
            if column_index >= len(row):
                continue
            values = _numbers(row[column_index])
            if not values:
                continue
            facts.append(
                _fact(
                    record,
                    entity,
                    metric,
                    period["period"],
                    values[0],
                    period["kind"],
                    " ".join(row),
                )
            )
    return facts


def _facts_from_table(record):
    rows = [
        [_clean(value) for value in row]
        for row in record["rows"]
        if any(_clean(value) for value in row)
    ]
    while rows and rows[0] and TABLE_LABEL_RE.match(rows[0][0]):
        rows.pop(0)
    if not rows:
        return []
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if sum(_period_spec(value) is not None for value in row) >= 2
        ),
        None,
    )
    if header_index is not None:
        return _facts_from_wide_table(record, rows, header_index)
    return _facts_from_row_table(record, rows)


def _revision_facts(doc_id, page, trust_level, page_text):
    normalized = _clean(page_text)
    pattern = re.compile(
        r"(?P<entity>U\.?S\.?|United States|Euro area|China)\s+"
        r"(?P<metric>output growth|unemployment rate|inflation rate)\s+"
        r"for\s+(?P<year>(?:19|20)\d{2}).{0,80}?revised\s+from\s+"
        r"(?P<old>-?\d+(?:\.\d+)?)%?.{0,100}?\bto\s+"
        r"(?P<new>-?\d+(?:\.\d+)?)%",
        re.I,
    )
    facts = []
    for match in pattern.finditer(normalized):
        record = {
            "doc_id": doc_id,
            "page": page,
            "table_id": None,
            "trust_level": trust_level,
        }
        facts.append(
            _fact(
                record,
                _entity(match.group("entity")),
                _metric(match.group("metric")),
                match.group("year"),
                float(match.group("new")),
                "revision",
                match.group(0),
            )
        )
    return facts


def _narrative_facts(doc_id, page, trust_level, page_text):
    facts = []
    sentences = [
        _clean(fragment)
        for fragment in re.split(r"(?<=[.!?])\s+|\n+", page_text)
        if _clean(fragment)
    ]
    for sentence in sentences:
        entities = _mentions(sentence, ENTITY_PATTERNS)
        metrics = _mentions(sentence, METRIC_PATTERNS)
        years = YEAR_RE.findall(sentence)
        percentages = re.findall(r"(-?\d+(?:\.\d+)?)%", sentence)
        if not (
            len(entities) == 1
            and len(metrics) == 1
            and len(years) >= 1
            and len(percentages) == 1
        ):
            continue
        record = {
            "doc_id": doc_id,
            "page": page,
            "table_id": None,
            "trust_level": trust_level,
        }
        facts.append(
            _fact(
                record,
                entities[0],
                metrics[0],
                years[0],
                float(percentages[0]),
                "narrative",
                sentence,
            )
        )
    return facts


def extract_structured_facts(input_dir, documents):
    """Extract normalized, evidence-safe facts from all PDF tables and notices."""
    table_records = []
    page_records = []
    facts = []
    for doc_id, metadata in documents.items():
        pdf_path = input_dir / metadata["file"]
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1
                trust_level = _trust_for_page(metadata, page_number)
                page_text = page.extract_text() or ""
                page_records.append(
                    {
                        "doc_id": doc_id,
                        "page": page_number,
                        "trust_level": trust_level,
                        "text": page_text,
                    }
                )
                if trust_level == "low":
                    continue
                titles = _table_titles(page_text)
                extracted_tables = page.extract_tables()
                # A prose introduction can repeat the table label before the
                # actual caption. The last N label lines correspond to the N
                # extracted tables and preserve the semantic caption.
                caption_titles = titles[-len(extracted_tables) :] if extracted_tables else []
                for table_index, rows in enumerate(extracted_tables):
                    row_title = _clean(rows[0][0]) if rows and rows[0] else ""
                    title = (
                        row_title
                        if TABLE_LABEL_RE.match(row_title)
                        else caption_titles[table_index]
                        if table_index < len(caption_titles)
                        else f"Table on page {page_number}"
                    )
                    record = {
                        "doc_id": doc_id,
                        "page": page_number,
                        "citation": f"{doc_id}:p{page_number}",
                        "trust_level": trust_level,
                        "table_id": _table_id(title),
                        "title": title,
                        "rows": rows,
                    }
                    table_records.append(record)
                    table_facts = _facts_from_table(record)
                    vintage_years = YEAR_RE.findall(metadata.get("data_vintage", ""))
                    if vintage_years:
                        vintage_year = int(vintage_years[-1])
                        for fact in table_facts:
                            if (
                                fact["period"].isdigit()
                                and int(fact["period"]) >= vintage_year
                                and fact["value_kind"] == "reported"
                            ):
                                fact["value_kind"] = "forecast"
                    facts.extend(table_facts)
                facts.extend(
                    _revision_facts(doc_id, page_number, trust_level, page_text)
                )
                facts.extend(
                    _narrative_facts(doc_id, page_number, trust_level, page_text)
                )
    unique = {}
    for fact in facts:
        key = (
            fact["doc_id"],
            fact["page"],
            fact["table_id"],
            fact["entity"],
            fact["metric"],
            fact["period"],
            fact["value"],
            fact["value_kind"],
        )
        unique[key] = fact
    return {
        "tables": table_records,
        "facts": list(unique.values()),
        "pages": page_records,
    }


def parse_query_frame(question, conversation_context=None):
    """Parse entity, metric, period, scope, and operation without question IDs."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    entities = _mentions(question, ENTITY_PATTERNS)
    metrics = _mentions(question, METRIC_PATTERNS)
    years = list(dict.fromkeys(YEAR_RE.findall(question)))

    recent_questions = []
    if isinstance(conversation_context, dict):
        context_turns = conversation_context.get(
            "recent_turns",
            conversation_context.get("prior_turns", []),
        )
        recent_questions = [
            turn.get("question", "")
            for turn in context_turns
            if isinstance(turn, dict)
        ]
    for earlier in reversed(recent_questions):
        if not entities:
            entities = _mentions(earlier, ENTITY_PATTERNS)
        if not metrics:
            metrics = _mentions(earlier, METRIC_PATTERNS)
        if not years:
            years = list(dict.fromkeys(YEAR_RE.findall(earlier)))
        if entities and metrics and years:
            break

    folded = _clean(question).casefold()
    if "forecast error" in folded or "outturn minus" in folded:
        operation = "forecast_error"
    elif any(term in folded for term in ("agree", "disagree", "conflict")):
        operation = "source_agreement"
    elif len(years) >= 2 and any(
        term in folded for term in ("between", "from", "change", "rise", "rose", "fall", "fell")
    ):
        operation = "change"
    elif any(
        term in folded
        for term in (
            "by how many",
            "by how much",
            "difference",
            "outperform",
            "below",
            "above",
            "which was larger",
            "which is larger",
        )
    ):
        operation = (
            "difference"
            if len(entities) >= 2
            or len(metrics) >= 2
            or any(
                term in folded
                for term in ("below", "above", "outperform", "versus", "which")
            )
            else "direct"
        )
    else:
        operation = "direct"

    if "outturn" in folded or "actual" in folded:
        requested_kind = "outturn"
    elif "forecast" in folded and operation != "forecast_error":
        requested_kind = "forecast"
    else:
        requested_kind = None

    requested_docs = [
        f"DOC{match.group(1)}"
        for match in re.finditer(r"\b(?:document|doc)\s*([1-3])\b", question, re.I)
    ]
    labels = [match.group(0) for match in VISUAL_LABEL_RE.finditer(question)]
    return {
        "entities": entities,
        "metrics": metrics,
        "years": years,
        "operation": operation,
        "requested_kind": requested_kind,
        "requested_doc_ids": list(dict.fromkeys(requested_docs)),
        "requested_labels": list(dict.fromkeys(labels)),
    }


def _current_group_facts(facts, requested_kind=None):
    session_facts = [
        fact for fact in facts if fact["value_kind"] == "session_override"
    ]
    if session_facts and requested_kind is None:
        return session_facts
    if requested_kind:
        matching = [fact for fact in facts if fact["value_kind"] == requested_kind]
        return matching
    revisions = [fact for fact in facts if fact["value_kind"] == "revision"]
    if revisions:
        return revisions
    outturns = [fact for fact in facts if fact["value_kind"] == "outturn"]
    if outturns:
        return outturns + [
            fact
            for fact in facts
            if fact["value_kind"] == "narrative"
        ]
    return [fact for fact in facts if fact["value_kind"] != "forecast"] or facts


def _best_fact(facts):
    return sorted(
        facts,
        key=lambda fact: (
            -TRUST_RANK.get(fact["trust_level"], -1),
            0 if fact["value_kind"] == "revision" else 1,
            fact["doc_id"],
            fact["page"],
        ),
    )[0]


def _group_by_key(facts):
    grouped = defaultdict(list)
    for fact in facts:
        grouped[(fact["entity"], fact["metric"], fact["period"])].append(fact)
    return grouped


def _select_facts(frame, structured):
    selected = []
    for fact in structured["facts"]:
        if frame["requested_doc_ids"] and fact["doc_id"] not in frame["requested_doc_ids"]:
            continue
        if frame["entities"] and fact["entity"] not in frame["entities"]:
            continue
        if frame["metrics"] and fact["metric"] not in frame["metrics"]:
            continue
        if frame["years"] and fact["period"] not in frame["years"]:
            continue
        requested_tables = [
            label for label in frame["requested_labels"] if label.casefold().startswith("table")
        ]
        if requested_tables and fact["table_id"] not in requested_tables:
            continue
        selected.append(fact)
    return selected


def _fact_conflict(facts):
    current = _current_group_facts(facts)
    values = {round(fact["value"], 8) for fact in current}
    return current if len(values) > 1 else []


def _citations(facts):
    return list(
        dict.fromkeys(
            fact["citation"] for fact in facts if fact.get("citation")
        )
    )


def _figures(facts):
    return list(
        dict.fromkeys(fact["table_id"] for fact in facts if fact.get("table_id"))
    )


def _answer_status(facts):
    return (
        "session_override"
        if any(fact["value_kind"] == "session_override" for fact in facts)
        else "answered"
    )


def _value_text(value):
    return f"{value:.1f}%"


def _conflict_answer(facts):
    current = _current_group_facts(facts)
    preferred = _best_fact(current)
    distinct = {}
    for fact in current:
        distinct.setdefault((fact["doc_id"], fact["value"]), fact)
    ordered = sorted(
        distinct.values(),
        key=lambda fact: (-TRUST_RANK.get(fact["trust_level"], -1), fact["doc_id"]),
    )
    values = ", while ".join(
        f"{fact['doc_id']} reports {_value_text(fact['value'])}"
        for fact in ordered
    )
    answer = (
        f"The sources conflict: {values}. Following the manifest, prefer the "
        f"higher-trust value of {_value_text(preferred['value'])} from "
        f"{preferred['doc_id']}."
    )
    return {
        "answer": answer,
        "status": "conflict",
        "citations": _citations(ordered),
        "figure_references": _figures(ordered),
    }, ordered


def _direct_answer(frame, selected):
    grouped = _group_by_key(selected)
    if len(grouped) != 1:
        return None
    facts = next(iter(grouped.values()))
    conflict = _fact_conflict(facts)
    if conflict:
        return _conflict_answer(facts)
    current = _current_group_facts(facts, frame["requested_kind"])
    if not current:
        return None
    fact = _best_fact(current)
    prefix = (
        "Under the session assumption, "
        if fact["value_kind"] == "session_override"
        else ""
    )
    answer = (
        f"{prefix}{fact['entity']} {fact['metric']} was {_value_text(fact['value'])} "
        f"in {fact['period']}."
    )
    return {
        "answer": answer,
        "status": _answer_status([fact]),
        "citations": _citations([fact]),
        "figure_references": _figures([fact]),
    }, [fact]


def _forecast_error_answer(frame, selected):
    grouped = _group_by_key(selected)
    candidates = []
    for key, facts in grouped.items():
        outturns = [fact for fact in facts if fact["value_kind"] == "outturn"]
        forecasts = [fact for fact in facts if fact["value_kind"] == "forecast"]
        if outturns and forecasts:
            outturn = _best_fact(outturns)
            same_source_forecasts = [
                fact
                for fact in forecasts
                if fact["doc_id"] == outturn["doc_id"]
                and fact.get("table_id") == outturn.get("table_id")
            ]
            forecast = _best_fact(same_source_forecasts or forecasts)
            candidates.append((key, outturn, forecast))
    if len(candidates) != 1:
        return None
    (entity, metric, period), outturn, forecast = candidates[0]
    difference = outturn["value"] - forecast["value"]
    answer = (
        f"The forecast error was {difference:.1f} percentage points: "
        f"{_value_text(outturn['value'])} outturn minus "
        f"{_value_text(forecast['value'])} forecast for {entity} {metric} "
        f"in {period}."
    )
    supporting = [outturn, forecast]
    return {
        "answer": answer,
        "status": "answered",
        "citations": _citations(supporting),
        "figure_references": _figures(supporting),
    }, supporting


def _single_current_fact(facts):
    conflict = _fact_conflict(facts)
    if conflict:
        return None
    current = _current_group_facts(facts)
    return _best_fact(current) if current else None


def _change_answer(frame, selected):
    if len(frame["entities"]) != 1 or len(frame["metrics"]) != 1 or len(frame["years"]) < 2:
        return None
    grouped = _group_by_key(selected)
    start_year, end_year = frame["years"][0], frame["years"][-1]
    keys = [
        (frame["entities"][0], frame["metrics"][0], start_year),
        (frame["entities"][0], frame["metrics"][0], end_year),
    ]
    if not all(key in grouped for key in keys):
        return None
    start = _single_current_fact(grouped[keys[0]])
    end = _single_current_fact(grouped[keys[1]])
    if start is None or end is None:
        return None
    difference = end["value"] - start["value"]
    direction = "rose" if difference > 0 else "fell" if difference < 0 else "was unchanged"
    magnitude = abs(difference)
    answer = (
        f"{start['entity']} {start['metric']} {direction} by {magnitude:.1f} "
        f"percentage points, from {_value_text(start['value'])} in {start_year} "
        f"to {_value_text(end['value'])} in {end_year}."
    )
    supporting = [start, end]
    prefix = (
        "Under the session assumption, "
        if _answer_status(supporting) == "session_override"
        else ""
    )
    answer = prefix + answer if prefix else answer
    return {
        "answer": answer,
        "status": _answer_status(supporting),
        "citations": _citations(supporting),
        "figure_references": _figures(supporting),
    }, supporting


def _difference_answer(frame, selected):
    grouped = _group_by_key(selected)
    if len(frame["years"]) != 1:
        return None
    period = frame["years"][0]
    pairs = []
    if len(frame["entities"]) == 2 and len(frame["metrics"]) == 1:
        pairs = [
            (frame["entities"][0], frame["metrics"][0], period),
            (frame["entities"][1], frame["metrics"][0], period),
        ]
    elif len(frame["entities"]) == 1 and len(frame["metrics"]) == 2:
        pairs = [
            (frame["entities"][0], frame["metrics"][0], period),
            (frame["entities"][0], frame["metrics"][1], period),
        ]
    if not pairs or not all(key in grouped for key in pairs):
        return None
    first = _single_current_fact(grouped[pairs[0]])
    second = _single_current_fact(grouped[pairs[1]])
    if first is None or second is None:
        return None
    difference = first["value"] - second["value"]
    larger, smaller = (first, second) if difference >= 0 else (second, first)
    answer = (
        f"{larger['entity']} {larger['metric']} was larger by "
        f"{abs(difference):.1f} percentage points in {period} "
        f"({_value_text(larger['value'])} versus {_value_text(smaller['value'])})."
    )
    supporting = [first, second]
    prefix = (
        "Under the session assumption, "
        if _answer_status(supporting) == "session_override"
        else ""
    )
    answer = prefix + answer if prefix else answer
    return {
        "answer": answer,
        "status": _answer_status(supporting),
        "citations": _citations(supporting),
        "figure_references": _figures(supporting),
    }, supporting


def _precise_absence(frame, structured):
    if not (
        len(frame["entities"]) == 1
        and len(frame["metrics"]) == 1
        and len(frame["years"]) == 1
        and frame["operation"] == "direct"
    ):
        return False
    entity, metric, year = (
        frame["entities"][0],
        frame["metrics"][0],
        frame["years"][0],
    )
    entity_pattern = next(pattern for name, pattern in ENTITY_PATTERNS if name == entity)
    metric_pattern = next(pattern for name, pattern in METRIC_PATTERNS if name == metric)
    for page in structured["pages"]:
        if page["trust_level"] == "low":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", page["text"]):
            if (
                entity_pattern.search(sentence)
                and metric_pattern.search(sentence)
                and year in sentence
            ):
                return False
    return True


def _session_facts(conversation_context):
    if not isinstance(conversation_context, dict):
        return []
    facts = []
    for item in conversation_context.get("session_overrides", []):
        if not isinstance(item, dict):
            continue
        entity = _entity(item.get("subject", ""))
        metric = _metric(item.get("metric", ""))
        period_match = YEAR_RE.search(item.get("period", ""))
        values = _numbers(item.get("value", ""))
        if not entity or not metric or period_match is None or not values:
            continue
        facts.append(
            {
                "doc_id": "SESSION",
                "page": 0,
                "citation": None,
                "table_id": None,
                "trust_level": "high",
                "entity": entity,
                "metric": metric,
                "period": period_match.group(0),
                "value": values[0],
                "value_kind": "session_override",
                "source_text": "Conversation-scoped user assumption.",
            }
        )
    return facts


def resolve_structured_query(question, structured, conversation_context=None):
    """Return a generic structured answer or None for VLM/text reasoning."""
    frame = parse_query_frame(question, conversation_context)
    selected = _select_facts(frame, structured)
    if not frame["metrics"]:
        inferred_metrics = sorted({fact["metric"] for fact in selected})
        if len(inferred_metrics) == 1:
            frame["metrics"] = inferred_metrics
            selected = _select_facts(frame, structured)
    if not frame["requested_doc_ids"]:
        selected.extend(
            fact
            for fact in _session_facts(conversation_context)
            if (not frame["entities"] or fact["entity"] in frame["entities"])
            and (not frame["metrics"] or fact["metric"] in frame["metrics"])
            and (not frame["years"] or fact["period"] in frame["years"])
        )
    if frame["operation"] == "forecast_error":
        resolved = _forecast_error_answer(frame, selected)
    elif frame["operation"] == "change":
        resolved = _change_answer(frame, selected)
    elif frame["operation"] == "difference":
        resolved = _difference_answer(frame, selected)
    elif frame["operation"] == "source_agreement":
        grouped = _group_by_key(selected)
        resolved = None
        if len(grouped) == 1:
            facts = next(iter(grouped.values()))
            resolved = _conflict_answer(facts) if _fact_conflict(facts) else _direct_answer(frame, selected)
    else:
        resolved = _direct_answer(frame, selected)

    if resolved is None and _precise_absence(frame, structured):
        return {
            "answer": {
                "answer": NOT_FOUND_ANSWER,
                "status": "not_found",
                "citations": [],
                "figure_references": [],
            },
            "supporting_facts": [],
            "query_frame": frame,
        }
    if resolved is None:
        return None
    answer, supporting = resolved
    return {
        "answer": answer,
        "supporting_facts": deepcopy(
            [fact for fact in supporting if fact["doc_id"] != "SESSION"]
        ),
        "query_frame": frame,
    }


def structured_text_results(resolution):
    """Convert supporting facts into evidence-packet-compatible text results."""
    results = []
    for rank, fact in enumerate(resolution.get("supporting_facts", []), start=1):
        results.append(
            {
                "chunk_id": f"structured_{rank}",
                "doc_id": fact["doc_id"],
                "page": fact["page"],
                "citation": fact["citation"],
                "trust_level": fact["trust_level"],
                "text": (
                    f"Structured evidence from {fact.get('table_id') or 'document text'}: "
                    f"entity={fact['entity']}; metric={fact['metric']}; "
                    f"period={fact['period']}; value={fact['value']}; "
                    f"value_kind={fact['value_kind']}. {fact['source_text']}"
                ),
                "dense_rank": None,
                "bm25_rank": None,
                "rrf_score": 1.0,
            }
        )
    return results
