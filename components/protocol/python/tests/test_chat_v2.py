import copy
import json
from pathlib import Path

import pytest
from oryn_protocol.task_contracts import TASK_CONTRACTS, make_profile, validate_invocation, validate_profile, validate_requirements, validate_result

FIXTURE = json.loads((Path(__file__).parents[2] / 'fixtures/task-contracts/chat-v2.json').read_text())


@pytest.mark.parametrize('profile', FIXTURE['profiles'].values())
def test_thinking_is_accepted_even_for_fixed_unsupported_and_unknown_routes(profile):
    validate_profile(profile)
    assert 'thinking' not in profile['parameters']
    for thinking in (True, False):
        validate_invocation(profile, {'messages': [{'role': 'user', 'content': 'Synthetic question'}], 'thinking': thinking, 'maxTokens': 4096})
    with pytest.raises(ValueError):
        validate_invocation(profile, {'messages': [{'role': 'user', 'content': 'Synthetic question'}], 'thinking': 'false'})


@pytest.mark.parametrize('case', FIXTURE['cases'])
def test_metadata_results_use_shared_golden_cases(case):
    validate_result(FIXTURE['profiles']['unknown'], case['result'], case['input'])


def test_legacy_and_malformed_results_remain_closed():
    assert TASK_CONTRACTS['chat-v1']['inputSchema']['properties']['maxTokens']['maximum'] == 512
    assert 'thinking' not in TASK_CONTRACTS['chat-v1']['inputSchema']['properties']
    profile = make_profile('chat-v2', thinking={'support': 'fixed', 'default': True})
    validate_invocation(profile, {'messages': [{'role': 'user', 'content': 'Question'}], 'thinking': False})
    case = FIXTURE['cases'][0]
    wrong = copy.deepcopy(case['result'])
    wrong['value']['metadata']['thinking']['effective'] = True
    with pytest.raises(ValueError):
        validate_result(profile, wrong, case['input'])
    with pytest.raises(ValueError):
        validate_result(profile, case['result'], case['input'] | {'thinking': True})
    with pytest.raises(ValueError):
        validate_result(profile, case['result'], case['input'] | {'maxTokens': 1})
    with pytest.raises(ValueError):
        validate_requirements({'task': 'text-generation', 'contract': {'id': 'chat-v2', 'revision': 1}, 'parameterValues': {'thinking': False}})
