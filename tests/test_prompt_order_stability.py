"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Regression tests for issue #1931: the prompts that ``add_episode`` builds must not
# depend on string-hash order (salted per process) or on the order in which the
# database happens to return rows. All tests are database-free.

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EpisodeType, EpisodicNode, get_entity_node_from_record
from graphiti_core.tracer import create_tracer
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge
from graphiti_core.utils.maintenance.node_operations import extract_nodes

# Type names on both sides of 'Entity' in sort order. With a set-derived order each
# node has a 1/2 chance of the sorted order, so all twelve at once is ~1/4096.
ENTITY_TYPE_NAMES = [
    'Account',
    'Budget',
    'Contract',
    'Document',
    'Event',
    'Fund',
    'Location',
    'Organization',
    'Person',
    'Place',
    'Product',
    'Topic',
]


def _make_episode(uuid: str, content: str = 'content') -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid,
        name=uuid,
        group_id='group_1',
        source=EpisodeType.text,
        source_description='test',
        content=content,
        valid_at=utc_now(),
    )


def _make_edge(uuid: str, fact: str) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='RELATES_TO',
        group_id='group_1',
        fact=fact,
        episodes=[],
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
        valid_at=None,
        invalid_at=None,
    )


@pytest.mark.asyncio
async def test_extracted_node_labels_are_sorted():
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'extracted_entities': [
                {'name': f'{type_name} one', 'entity_type_id': i + 1}
                for i, type_name in enumerate(ENTITY_TYPE_NAMES)
            ]
        }
    )
    clients = GraphitiClients.model_construct(
        driver=MagicMock(),
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
        llm_client=llm_client,
    )
    entity_types = {
        name: type(name, (BaseModel,), {'__doc__': f'A {name.lower()}.'})
        for name in ENTITY_TYPE_NAMES
    }

    nodes, _ = await extract_nodes(
        clients, _make_episode('episode'), previous_episodes=[], entity_types=entity_types
    )

    assert len(nodes) == len(ENTITY_TYPE_NAMES)
    for node in nodes:
        assert node.labels == sorted(node.labels), node.labels


@pytest.mark.parametrize(
    'db_labels',
    [
        ['Place', 'Entity', 'Entity_group1'],
        ['Entity_group1', 'Entity', 'Place'],
        ['Entity', 'Place', 'Entity_group1'],
    ],
)
def test_labels_read_from_database_do_not_depend_on_database_order(db_labels):
    record = {
        'uuid': 'node_uuid',
        'name': 'Some place',
        'name_embedding': None,
        'group_id': 'group1',
        'labels': db_labels,
        'created_at': '2026-01-01T00:00:00+00:00',
        'summary': '',
        'attributes': {},
    }

    node = get_entity_node_from_record(record, GraphProvider.NEO4J)

    assert node.labels == ['Entity', 'Place']


@pytest.mark.asyncio
async def test_edge_dedup_prompt_does_not_depend_on_candidate_order():
    related = [_make_edge('r1', 'Alice works at Acme'), _make_edge('r2', 'Alice joined Acme')]
    existing = [_make_edge('e1', 'Alice lives in Paris'), _make_edge('e2', 'Alice owns a car')]
    extracted = _make_edge('new', 'Alice is employed by Acme')
    episode = _make_episode('episode')

    prompts = []
    resolved_uuids = []
    for related_order, existing_order in [
        (related, existing),
        (related[::-1], existing[::-1]),
    ]:
        llm_client = MagicMock()
        # The model answers by index; idx 0 must name the same candidate in both runs.
        llm_client.generate_response = AsyncMock(
            return_value={'duplicate_facts': [0], 'contradicted_facts': []}
        )

        resolved, _, _ = await resolve_extracted_edge(
            llm_client,
            extracted.model_copy(deep=True),
            [e.model_copy(deep=True) for e in related_order],
            [e.model_copy(deep=True) for e in existing_order],
            episode,
            edge_type_candidates=None,
        )

        prompts.append(llm_client.generate_response.await_args_list[0].args[0])
        resolved_uuids.append(resolved.uuid)

    assert prompts[0] == prompts[1]
    assert resolved_uuids[0] == resolved_uuids[1]


class _StopAfterPreviousEpisodes(Exception):
    pass


@pytest.mark.asyncio
async def test_previous_episodes_follow_the_callers_order(monkeypatch):
    from graphiti_core import graphiti as graphiti_module

    requested = ['ep_a', 'ep_b', 'ep_c']
    # The database returns the requested episodes in an order of its own.
    from_database = [_make_episode('ep_c'), _make_episode('ep_a'), _make_episode('ep_b')]
    monkeypatch.setattr(EpisodicNode, 'get_by_uuids', AsyncMock(return_value=from_database))

    seen: list[list[str]] = []

    async def capture_previous_episodes(clients, episode, previous_episodes, *args, **kwargs):
        seen.append([e.uuid for e in previous_episodes])
        raise _StopAfterPreviousEpisodes

    monkeypatch.setattr(graphiti_module, 'extract_nodes', capture_previous_episodes)

    graphiti = Graphiti.__new__(Graphiti)
    graphiti.tracer = create_tracer()
    driver = MagicMock()
    graphiti._resolve_request_scope = lambda group_id: ('group_1', driver, MagicMock())

    with pytest.raises(_StopAfterPreviousEpisodes):
        await graphiti.add_episode(
            name='episode',
            episode_body='content',
            source_description='test',
            reference_time=utc_now(),
            group_id='group_1',
            previous_episode_uuids=requested,
        )

    assert seen == [requested]
