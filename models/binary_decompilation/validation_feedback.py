"""Compact counterexamples and explicit execution coverage for replacement tests."""

from collections import Counter
import json


def difference(expected, actual):
    a = expected if isinstance(expected, bytes) else expected.encode('utf-8')
    b = actual if isinstance(actual, bytes) else actual.encode('utf-8')
    first = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)))
    start = max(0, first - 48)
    return {"expected_bytes": len(a), "actual_bytes": len(b), "first_difference": first,
            "expected_excerpt": a[start:first + 160].decode('utf-8', 'backslashreplace'),
            "actual_excerpt": b[start:first + 160].decode('utf-8', 'backslashreplace')}


def runtime_feedback(tests, events, name):
    hits = Counter(e.get('test_id', 'direct_probe') for e in events
                   if e.get('event') == 'replacement' and e.get('name') == name)
    failures = []
    for test in tests['tests']:
        if test.get('returncode') in (0, 77):
            continue
        item = {key: test.get(key) for key in ('test', 'actual_exit_status', 'expected_exit_status')}
        if 'expected' in test and 'actual' in test:
            item.update(expected_exit_status=test['expected'].get('returncode'),
                        actual_exit_status=test['actual'].get('returncode'))
            for stream in ('stdout', 'stderr'):
                expected, actual = test['expected'].get(stream, ''), test['actual'].get(stream, '')
                if expected != actual:
                    item[stream] = difference(expected, actual)
        for stream in ('stdout', 'stderr'):
            if stream in test.get('byte_differences', {}):
                item[stream] = test['byte_differences'][stream]
            elif 'expected_' + stream in test:
                expected, actual = test['expected_' + stream], test.get(stream, '')
                if expected != actual:
                    item[stream] = difference(expected, actual)
        failures.append(item)
    loader_errors = list(dict.fromkeys(e.get('description', '') for e in events if e.get('type') == 'error'))
    infrastructure = [test.get('stderr', '')[:1200] for test in tests['tests']
                      if 'Binary isolation infrastructure failed:' in test.get('stderr', '')]
    crashes = [e for e in events if e.get('event') == 'crash'][:2]
    kind = 'loader_error' if loader_errors else 'infrastructure' if infrastructure else 'runtime' if failures else 'missing_execution' if not hits else 'accepted'
    exercised = [test for test in tests['tests'] if hits.get(test.get('test'))]
    exercised.extend(test for test in tests['tests'] if test.get('target_executed') and test not in exercised)
    details = {"failure_kind": kind, "failing_cases": len(failures), "counterexamples": failures[:6],
               "loader_errors": [s[:1200] for s in loader_errors[:2]], "crashes": crashes,
               "infrastructure_errors": infrastructure[:2],
               "coverage": {"suite_cases": len(tests['tests']), "target_exercising_cases": len(hits),
                            "replacement_calls_by_test": dict(hits),
                            "target_passed_cases": sum(t.get('returncode') == 0 for t in exercised),
                            "target_failed_cases": sum(t.get('returncode') not in (0, 77) for t in exercised),
                            "note": "Calls are capped at the instrumentation sample limit per process"}}
    # Counterexamples retain the first divergence, not the end of enormous help output.
    return details, json.dumps(details)
