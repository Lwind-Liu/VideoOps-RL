"""Permission-separated specialist agents and evidence-aware coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .multivideo_env import MultiVideoHighlightEnv


@dataclass(frozen=True)
class RoutePlan:
    use_text: bool
    use_image: bool
    top_k: int
    inspect_budget: int
    context_radius: int


class QueryRouter:
    VISUAL_CUES = {"appearance", "butterfly", "color", "flowers", "forest", "hide", "look", "scene", "see", "tree", "visual", "wear"}

    def route(self, query: str, video_has_dialogue: bool) -> RoutePlan:
        words = set(query.lower().replace("?", "").replace(".", "").split())
        visual = bool(words & self.VISUAL_CUES) or not video_has_dialogue
        # Image search remains active for dialogue tasks because the final claim must be grounded visually.
        return RoutePlan(use_text=video_has_dialogue, use_image=True, top_k=6 if visual else 5, inspect_budget=3 if visual else 2, context_radius=1)


class TimelineScout:
    allowed_tools = {"search_transcript", "expand_context"}

    def propose(self, env: MultiVideoHighlightEnv, query: str, top_k: int) -> dict[str, Any]:
        return env.search_transcript(query, top_k)

    def expand(self, env: MultiVideoHighlightEnv, shot_id: str, radius: int) -> dict[str, Any]:
        return env.expand_context(shot_id, radius)


class VisionAnalyst:
    allowed_tools = {"search_visual", "inspect_keyframe"}

    def propose(self, env: MultiVideoHighlightEnv, query: str, top_k: int) -> dict[str, Any]:
        return env.search_visual(query, top_k)

    def inspect(self, env: MultiVideoHighlightEnv, shot_id: str) -> dict[str, Any]:
        return env.inspect_keyframe(shot_id)


class EvidenceAuditor:
    allowed_tools = {"request_audit"}

    def audit(self, env: MultiVideoHighlightEnv, shot_ids: list[str]) -> dict[str, Any]:
        return env.request_audit(shot_ids)


class Coordinator:
    """Fuses independent proposals through the shared temporal evidence graph."""

    def select(self, env: MultiVideoHighlightEnv, inspect_budget: int) -> list[str]:
        ranked = env.graph.ranked(inspect_budget)
        if not ranked:
            return []
        selected = [ranked[0].shot_id]
        top = ranked[0].fused_score
        for node in ranked[1:]:
            if len(selected) >= inspect_budget or node.fused_score < 0.88 * top:
                break
            if min(abs(env.graph.order.index(node.shot_id) - env.graph.order.index(item)) for item in selected) <= 1:
                selected.append(node.shot_id)
        return sorted(selected, key=env.graph.order.index)


def fixed_chunk_baseline(env: MultiVideoHighlightEnv) -> dict[str, Any]:
    shot_id = env.units[len(env.units) // 2]["shot_id"]
    env.graph.update("image", {shot_id: 1.0})
    env.state.searched_modalities.append("image")
    env.inspect_keyframe(shot_id)
    env.request_audit([shot_id])
    return env.submit([shot_id])


def single_agent(env: MultiVideoHighlightEnv) -> dict[str, Any]:
    router, query = QueryRouter(), env.task["query"]
    plan = router.route(query, env.has_dialogue)
    if plan.use_text:
        env.search_transcript(query, plan.top_k)
    if plan.use_image:
        env.search_visual(query, plan.top_k)
    selected = Coordinator().select(env, 1)
    if not selected:
        return env.submit([])
    env.inspect_keyframe(selected[0])
    return env.submit(selected)


def multi_agent_expert(env: MultiVideoHighlightEnv) -> dict[str, Any]:
    query = env.task["query"]
    plan = QueryRouter().route(query, env.has_dialogue)
    timeline, vision, auditor, coordinator = TimelineScout(), VisionAnalyst(), EvidenceAuditor(), Coordinator()
    if plan.use_text:
        timeline.propose(env, query, plan.top_k)
    if plan.use_image:
        vision.propose(env, query, plan.top_k)
    # Feature benchmarks expose variable-length temporal proposals. The expert
    # verifies one anchor, expands the decoded proposal, then audits the whole
    # grounded interval; no target label is consulted by this policy.
    proposals = getattr(env, "proposals", [])
    if proposals:
        proposal = proposals[0]
        vision.inspect(env, proposal.anchor_shot_id)
        expanded = timeline.expand(env, proposal.anchor_shot_id, 0)
        selected = expanded.get("shot_ids", proposal.shot_ids)
        auditor.audit(env, selected)
        return env.submit(selected)
    selected = coordinator.select(env, plan.inspect_budget)
    if not selected:
        return env.submit([])
    for shot_id in selected:
        vision.inspect(env, shot_id)
    # Expand only when the top evidence is weak; this makes tool depth query-dependent.
    if env.graph.nodes[selected[0]].fused_score < 0.30 and env.state.tool_calls + 2 < env.max_steps:
        timeline.expand(env, selected[0], plan.context_radius)
    audit = auditor.audit(env, selected)
    if not audit["passed"] and len(selected) > 1:
        selected = selected[:1]
        auditor.audit(env, selected)
    return env.submit(selected)
