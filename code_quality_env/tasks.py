from __future__ import annotations

from .models import (
    FileSnapshot,
    FindingType,
    GroundTruthFinding,
    Severity,
    TaskObjective,
    TaskSpec,
)


def build_tasks() -> list[TaskSpec]:
    return [
        TaskSpec(
            name="easy_api_logging_review",
            difficulty="easy",
            objective=TaskObjective(
                goal="Review a tiny API handler and flag obvious readability and missing logging/comment issues.",
                required_focus=[FindingType.READABILITY, FindingType.LOGGING, FindingType.COMMENTS],
                max_steps=12,
            ),
            files=[
                FileSnapshot(
                    file_path="handlers/user_profile.py",
                    content="""def get_profile(db, uid):
    if uid==None:
        return {\"err\":\"missing\"}
    x=db.fetch(uid)
    if x is None:
        return {\"err\":\"notfound\"}
    return {\"name\":x.name,\"signup\":x.s}\n""",
                )
            ],
            ground_truth=[
                GroundTruthFinding(
                    finding_id="easy-1",
                    file_path="handlers/user_profile.py",
                    line=2,
                    finding_type=FindingType.READABILITY,
                    severity=Severity.LOW,
                    must_include_keywords=["is none", "comparison"],
                ),
                GroundTruthFinding(
                    finding_id="easy-2",
                    file_path="handlers/user_profile.py",
                    line=4,
                    finding_type=FindingType.LOGGING,
                    severity=Severity.MEDIUM,
                    must_include_keywords=["log", "notfound"],
                ),
                GroundTruthFinding(
                    finding_id="easy-3",
                    file_path="handlers/user_profile.py",
                    line=1,
                    finding_type=FindingType.COMMENTS,
                    severity=Severity.LOW,
                    must_include_keywords=["docstring", "function"],
                ),
            ],
        ),
        TaskSpec(
            name="medium_batch_job_review",
            difficulty="medium",
            objective=TaskObjective(
                goal="Review a batch ETL job for maintainability and instrumentation issues before production rollout.",
                required_focus=[FindingType.READABILITY, FindingType.LOGGING, FindingType.COMMENTS],
                max_steps=18,
            ),
            files=[
                FileSnapshot(
                    file_path="jobs/daily_sync.py",
                    content="""def run(cfg, rows, sink, logger):
    ok = 0
    bad = 0
    for r in rows:
        if 'id' not in r or 'email' not in r:
            bad += 1
            continue
        v = r['email'].strip().lower()
        if '@' not in v:
            bad += 1
            continue
        sink.upsert(r['id'], {'email': v})
        ok += 1
    if bad > 0:
        pass
    return {'ok': ok, 'bad': bad}\n""",
                ),
                FileSnapshot(
                    file_path="jobs/helpers.py",
                    content="""def normalize_name(name):
    if not name:
        return ''
    return ' '.join([p.capitalize() for p in name.strip().split(' ') if p])\n""",
                ),
            ],
            ground_truth=[
                GroundTruthFinding(
                    finding_id="med-1",
                    file_path="jobs/daily_sync.py",
                    line=14,
                    finding_type=FindingType.LOGGING,
                    severity=Severity.HIGH,
                    must_include_keywords=["logger", "bad"],
                ),
                GroundTruthFinding(
                    finding_id="med-2",
                    file_path="jobs/daily_sync.py",
                    line=1,
                    finding_type=FindingType.COMMENTS,
                    severity=Severity.LOW,
                    must_include_keywords=["docstring", "batch"],
                ),
                GroundTruthFinding(
                    finding_id="med-3",
                    file_path="jobs/daily_sync.py",
                    line=4,
                    finding_type=FindingType.READABILITY,
                    severity=Severity.MEDIUM,
                    must_include_keywords=["validate", "helper"],
                ),
                GroundTruthFinding(
                    finding_id="med-4",
                    file_path="jobs/helpers.py",
                    line=4,
                    finding_type=FindingType.READABILITY,
                    severity=Severity.LOW,
                    must_include_keywords=["split", "whitespace"],
                ),
            ],
        ),
        TaskSpec(
            name="hard_service_refactor_review",
            difficulty="hard",
            objective=TaskObjective(
                goal="Perform an in-depth review of a service method and detect subtle readability, logging, and comment quality defects.",
                required_focus=[FindingType.READABILITY, FindingType.LOGGING, FindingType.COMMENTS],
                max_steps=24,
            ),
            files=[
                FileSnapshot(
                    file_path="services/reporting.py",
                    content="""def build_monthly_report(user, txns, writer, logger, now):
    # TODO remove this once accounting migration is done
    a = []
    for t in txns:
        if t['user_id'] != user.id:
            continue
        if t.get('status') == 'VOID':
            continue
        if 'amount' not in t:
            continue
        amt = float(t['amount'])
        if amt < 0:
            amt = 0
        a.append((t.get('category','misc'), amt, t.get('ts')))

    totals = {}
    for c, amt, ts in a:
        if ts is None:
            continue
        m = ts[:7]
        if m not in totals:
            totals[m] = {}
        if c not in totals[m]:
            totals[m][c] = 0
        totals[m][c] += amt

    rows = []
    for month in sorted(totals.keys()):
        cats = totals[month]
        gross = sum(cats.values())
        rows.append({'month': month, 'gross': gross, 'categories': cats})

    writer.write(rows)
    return {'count': len(rows), 'at': now}\n""",
                ),
                FileSnapshot(
                    file_path="services/formatting.py",
                    content="""def format_currency(v, currency='USD'):
    if currency == 'USD':
        return '$' + ('%.2f' % v)
    if currency == 'EUR':
        return 'EUR ' + ('%.2f' % v)
    return str(v)\n""",
                ),
            ],
            ground_truth=[
                GroundTruthFinding(
                    finding_id="hard-1",
                    file_path="services/reporting.py",
                    line=3,
                    finding_type=FindingType.READABILITY,
                    severity=Severity.MEDIUM,
                    must_include_keywords=["variable", "descriptive"],
                ),
                GroundTruthFinding(
                    finding_id="hard-2",
                    file_path="services/reporting.py",
                    line=2,
                    finding_type=FindingType.COMMENTS,
                    severity=Severity.MEDIUM,
                    must_include_keywords=["todo", "tracked issue"],
                ),
                GroundTruthFinding(
                    finding_id="hard-3",
                    file_path="services/reporting.py",
                    line=31,
                    finding_type=FindingType.LOGGING,
                    severity=Severity.HIGH,
                    must_include_keywords=["logger", "written"],
                ),
                GroundTruthFinding(
                    finding_id="hard-4",
                    file_path="services/reporting.py",
                    line=14,
                    finding_type=FindingType.READABILITY,
                    severity=Severity.MEDIUM,
                    must_include_keywords=["extract", "function"],
                ),
                GroundTruthFinding(
                    finding_id="hard-5",
                    file_path="services/formatting.py",
                    line=1,
                    finding_type=FindingType.COMMENTS,
                    severity=Severity.LOW,
                    must_include_keywords=["docstring", "currency"],
                ),
            ],
        ),
        TaskSpec(
            name="hard_incident_postmortem_review",
            difficulty="hard",
            objective=TaskObjective(
                goal="Review a production incident postmortem code path and catch observability, maintainability, and documentation defects across service, retry, and alerting modules.",
                required_focus=[FindingType.READABILITY, FindingType.LOGGING, FindingType.COMMENTS],
                max_steps=28,
            ),
            files=[
                FileSnapshot(
                    file_path="incident/recovery.py",
                    content="""def recover(events, store, notifier, logger, now):
    out = []
    for e in events:
        if e.get('kind') != 'failure':
            continue
        sid = e.get('service')
        if sid is None:
            continue
        prev = store.get(sid)
        if prev and prev.get('resolved'):
            continue
        score = 0
        if e.get('severity') == 'critical':
            score += 5
        if e.get('attempts', 0) > 3:
            score += 2
        if score > 3:
            notifier.page(sid, 'incident escalation')
        out.append({'service': sid, 'score': score, 'at': now})
    store.write_many(out)
    return {'written': len(out)}\n""",
                ),
                FileSnapshot(
                    file_path="incident/retry.py",
                    content="""def backoff(n):
    if n <= 0:
        return 0
    v = 1
    for _ in range(n):
        v = v * 2
    return min(v, 64)\n""",
                ),
                FileSnapshot(
                    file_path="incident/alerts.py",
                    content="""def format_alert(service, sev, attempts):
    return f\"svc={service}|sev={sev}|a={attempts}\"\n""",
                ),
            ],
            ground_truth=[
                GroundTruthFinding(
                    finding_id="post-1",
                    file_path="incident/recovery.py",
                    line=1,
                    finding_type=FindingType.COMMENTS,
                    severity=Severity.MEDIUM,
                    must_include_keywords=["docstring", "recovery"],
                ),
                GroundTruthFinding(
                    finding_id="post-2",
                    file_path="incident/recovery.py",
                    line=19,
                    finding_type=FindingType.LOGGING,
                    severity=Severity.HIGH,
                    must_include_keywords=["logger", "write"],
                ),
                GroundTruthFinding(
                    finding_id="post-3",
                    file_path="incident/recovery.py",
                    line=2,
                    finding_type=FindingType.READABILITY,
                    severity=Severity.MEDIUM,
                    must_include_keywords=["out", "descriptive"],
                ),
                GroundTruthFinding(
                    finding_id="post-4",
                    file_path="incident/recovery.py",
                    line=14,
                    finding_type=FindingType.READABILITY,
                    severity=Severity.MEDIUM,
                    must_include_keywords=["score", "named constant"],
                ),
                GroundTruthFinding(
                    finding_id="post-5",
                    file_path="incident/retry.py",
                    line=1,
                    finding_type=FindingType.COMMENTS,
                    severity=Severity.LOW,
                    must_include_keywords=["docstring", "backoff"],
                ),
                GroundTruthFinding(
                    finding_id="post-6",
                    file_path="incident/alerts.py",
                    line=1,
                    finding_type=FindingType.LOGGING,
                    severity=Severity.MEDIUM,
                    must_include_keywords=["structured", "fields"],
                ),
            ],
        ),
        TaskSpec(
            name="hard_distributed_checkout_review",
            difficulty="hard",
            objective=TaskObjective(
                goal="Review a distributed checkout workflow and catch subtle idempotency, observability, and readability defects across orchestrator, queue, and billing modules.",
                required_focus=[FindingType.READABILITY, FindingType.LOGGING, FindingType.COMMENTS],
                max_steps=32,
            ),
            files=[
                FileSnapshot(
                    file_path="checkout/orchestrator.py",
                    content="""def process_checkout(cmd, inventory, billing, outbox, logger, now):
    req_id = cmd.get('request_id')
    if req_id is None:
        return {'ok': False, 'err': 'missing_request_id'}
    items = cmd.get('items', [])
    reserved = []
    for it in items:
        sku = it.get('sku')
        qty = int(it.get('qty', 1))
        if qty <= 0:
            continue
        if not inventory.reserve(sku, qty):
            for s, q in reserved:
                inventory.release(s, q)
            return {'ok': False, 'err': 'no_stock', 'sku': sku}
        reserved.append((sku, qty))

    charge = billing.charge(cmd.get('user_id'), cmd.get('payment_token'), cmd.get('total'))
    if not charge.get('ok'):
        for s, q in reserved:
            inventory.release(s, q)
        return {'ok': False, 'err': 'payment_failed'}

    outbox.append({'type': 'OrderPlaced', 'request_id': req_id, 'ts': now, 'items': items})
    return {'ok': True, 'charge_id': charge.get('id')}
""",
                ),
                FileSnapshot(
                    file_path="checkout/outbox.py",
                    content="""def flush_outbox(outbox, publisher, logger):
    sent = 0
    for e in outbox:
        publisher.publish(e)
        sent += 1
    outbox.clear()
    return sent
""",
                ),
                FileSnapshot(
                    file_path="checkout/billing.py",
                    content="""def charge(user_id, token, total, gateway, logger):
    # TODO dedupe by request key later
    if total is None:
        return {'ok': False, 'err': 'missing_total'}
    resp = gateway.pay(token, float(total))
    if not resp.get('accepted'):
        return {'ok': False, 'err': 'declined'}
    return {'ok': True, 'id': resp.get('id')}
""",
                ),
            ],
            ground_truth=[
                GroundTruthFinding(
                    finding_id="dist-1",
                    file_path="checkout/orchestrator.py",
                    line=1,
                    finding_type=FindingType.COMMENTS,
                    severity=Severity.MEDIUM,
                    must_include_keywords=["docstring", "checkout"],
                ),
                GroundTruthFinding(
                    finding_id="dist-2",
                    file_path="checkout/orchestrator.py",
                    line=3,
                    finding_type=FindingType.LOGGING,
                    severity=Severity.HIGH,
                    must_include_keywords=["logger", "request_id"],
                ),
                GroundTruthFinding(
                    finding_id="dist-3",
                    file_path="checkout/orchestrator.py",
                    line=17,
                    finding_type=FindingType.LOGGING,
                    severity=Severity.HIGH,
                    must_include_keywords=["logger", "payment_failed"],
                ),
                GroundTruthFinding(
                    finding_id="dist-4",
                    file_path="checkout/outbox.py",
                    line=3,
                    finding_type=FindingType.LOGGING,
                    severity=Severity.HIGH,
                    must_include_keywords=["publish", "logger"],
                ),
                GroundTruthFinding(
                    finding_id="dist-5",
                    file_path="checkout/outbox.py",
                    line=1,
                    finding_type=FindingType.COMMENTS,
                    severity=Severity.LOW,
                    must_include_keywords=["docstring", "outbox"],
                ),
                GroundTruthFinding(
                    finding_id="dist-6",
                    file_path="checkout/billing.py",
                    line=2,
                    finding_type=FindingType.COMMENTS,
                    severity=Severity.MEDIUM,
                    must_include_keywords=["todo", "tracked issue"],
                ),
                GroundTruthFinding(
                    finding_id="dist-7",
                    file_path="checkout/billing.py",
                    line=3,
                    finding_type=FindingType.READABILITY,
                    severity=Severity.MEDIUM,
                    must_include_keywords=["validate", "helper"],
                ),
            ],
        ),
    ]
