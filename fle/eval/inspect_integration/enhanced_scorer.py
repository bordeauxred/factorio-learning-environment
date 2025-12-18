"""Enhanced scorer with throughput proportion, production score, and step change tracking."""

import logging
from typing import List
from inspect_ai.scorer import scorer, Score, Target, Scorer, accuracy, mean, score
from inspect_ai.agent import AgentState
from inspect_ai.util import store_as

from fle.eval.inspect_integration.controlled_solver import TrajectoryData

logger = logging.getLogger(__name__)


@scorer(metrics=[mean()])
def measured_throughput_scorer() -> Scorer:
    """Track raw measured throughput"""

    async def score(state: AgentState, target: Target) -> Score:
        try:
            trajectory_data = store_as(TrajectoryData)
            throughput = (
                trajectory_data.final_measured_throughput
                if trajectory_data.throughput_available
                else 0.0
            )

            return Score(
                value=throughput,
                answer=f"{throughput:.2f}",
                explanation=f"Measured Throughput: {throughput:.2f}",
                metadata={"throughput": throughput},
            )
        except Exception as e:
            logger.error(f"Error in measured throughput scorer: {e}")
            return Score(
                value=0.0, answer="0.00", explanation=f"Scorer error: {str(e)}"
            )

    return score


@scorer(metrics=[mean()])
def throughput_proportion_scorer() -> Scorer:
    """Track proportion of desired throughput achieved"""

    async def score(state: AgentState, target: Target) -> Score:
        try:
            trajectory_data = store_as(TrajectoryData)

            # Use measured throughput if available, otherwise fall back to ground truth
            if trajectory_data.throughput_available and trajectory_data.final_measured_throughput > 0:
                achieved_throughput = trajectory_data.final_measured_throughput
                metric_type = "throughput"
            else:
                # Fallback to ground truth (for backward compat)
                achieved_throughput = trajectory_data.final_ground_truth_score or 0.0
                metric_type = "ground_truth"

            # Get expected quota from metadata
            metadata = (
                getattr(state, "metadata", {}) if hasattr(state, "metadata") else {}
            )
            expected_score = metadata.get("expected_production_score", 100.0)

            # Calculate proportion (capped at 1.0)
            proportion = (
                min(achieved_throughput / expected_score, 1.0)
                if expected_score > 0
                else 0.0
            )

            return Score(
                value=proportion,
                answer=f"{proportion:.3f}",
                explanation=f"{metric_type.title()}: {achieved_throughput:.2f}/{expected_score:.2f} = {proportion:.3f}",
                metadata={
                    "achieved_throughput": achieved_throughput,
                    "expected_score": expected_score,
                    "proportion": proportion,
                    "quota_achieved": achieved_throughput >= expected_score,
                    "metric_type": metric_type,
                },
            )

        except Exception as e:
            logger.error(f"Error in throughput proportion scorer: {e}")
            return Score(
                value=0.0, answer="0.000", explanation=f"Scorer error: {str(e)}"
            )

    return score


@scorer(metrics=[mean()])
def production_score_tracker() -> Scorer:
    """Track overall production score"""

    async def score(state: AgentState, target: Target) -> Score:
        try:
            trajectory_data = store_as(TrajectoryData)
            production_score = (
                trajectory_data.final_score or trajectory_data.production_score or 0.0
            )

            # Get additional metrics from trajectory
            total_steps = trajectory_data.total_steps or 0
            error = trajectory_data.error

            return Score(
                value=production_score,
                answer=f"{production_score:.2f}",
                explanation=f"Production score: {production_score:.2f} over {total_steps} steps"
                + (f" (Error: {error})" if error else ""),
                metadata={
                    "production_score": production_score,
                    "total_steps": total_steps,
                    "has_error": bool(error),
                    "steps_per_score": total_steps / production_score
                    if production_score > 0
                    else 0,
                    "score_per_step": production_score / total_steps
                    if total_steps > 0
                    else 0,
                },
            )

        except Exception as e:
            logger.error(f"Error in production score tracker: {e}")
            return Score(
                value=0.0, answer="0.00", explanation=f"Scorer error: {str(e)}"
            )

    return score


@scorer(metrics=[mean()])
def step_change_tracker() -> Scorer:
    """Track change in production score from last step"""

    async def score(state: AgentState, target: Target) -> Score:
        try:
            trajectory_data = store_as(TrajectoryData)
            scores = trajectory_data.scores or []

            if len(scores) < 2:
                # Not enough data for change calculation
                change = 0.0
                final_change = 0.0
            else:
                # Calculate change from last step
                change = scores[-1] - scores[-2] if len(scores) >= 2 else 0.0
                # Calculate total change from first to last
                final_change = scores[-1] - scores[0] if len(scores) >= 2 else 0.0

            # Calculate additional change metrics
            max_single_step_gain = max(
                (scores[i] - scores[i - 1] for i in range(1, len(scores))), default=0.0
            )
            min_single_step_change = min(
                (scores[i] - scores[i - 1] for i in range(1, len(scores))), default=0.0
            )
            avg_step_change = (
                final_change / (len(scores) - 1) if len(scores) > 1 else 0.0
            )

            # Use absolute change as score (to track magnitude of improvement)
            score_value = abs(change)

            return Score(
                value=score_value,
                answer=f"{change:.3f}",
                explanation=f"Step change: {change:.3f}, Total change: {final_change:.3f}, Avg: {avg_step_change:.3f}",
                metadata={
                    "last_step_change": change,
                    "total_change": final_change,
                    "average_step_change": avg_step_change,
                    "max_single_step_gain": max_single_step_gain,
                    "min_single_step_change": min_single_step_change,
                    "total_steps_with_scores": len(scores),
                    "scores_trajectory": scores[-10:]
                    if len(scores) > 10
                    else scores,  # Last 10 scores for analysis
                },
            )

        except Exception as e:
            logger.error(f"Error in step change tracker: {e}")
            return Score(
                value=0.0, answer="0.000", explanation=f"Scorer error: {str(e)}"
            )

    return score


@scorer(metrics=[accuracy(), mean()])
def comprehensive_factorio_scorer() -> Scorer:
    """Comprehensive scorer combining all metrics"""

    async def score(state: AgentState, target: Target) -> Score:
        try:
            trajectory_data = store_as(TrajectoryData)

            # Use measured throughput if available, otherwise ground truth
            if trajectory_data.throughput_available and trajectory_data.final_measured_throughput > 0:
                achieved_score = trajectory_data.final_measured_throughput
                metric_type = "throughput"
            else:
                achieved_score = (
                    trajectory_data.final_score or trajectory_data.production_score or 0.0
                )
                metric_type = "ground_truth"

            # Get ground truth separately for metadata
            ground_truth_score = trajectory_data.final_ground_truth_score or 0.0

            scores = trajectory_data.scores or []
            error = trajectory_data.error
            total_steps = trajectory_data.total_steps or 0

            # Get expected quota from metadata
            metadata = (
                getattr(state, "metadata", {}) if hasattr(state, "metadata") else {}
            )
            expected_score = metadata.get("expected_production_score", 100.0)

            # Calculate all metrics
            throughput_proportion = (
                min(achieved_score / expected_score, 1.0)
                if expected_score > 0
                else 0.0
            )
            quota_achieved = achieved_score >= expected_score and not error

            # Step change metrics
            last_step_change = scores[-1] - scores[-2] if len(scores) >= 2 else 0.0
            total_change = scores[-1] - scores[0] if len(scores) >= 2 else 0.0
            avg_step_change = (
                total_change / (len(scores) - 1) if len(scores) > 1 else 0.0
            )

            # Performance metrics
            score_per_step = achieved_score / total_steps if total_steps > 0 else 0.0
            max_single_gain = max(
                (scores[i] - scores[i - 1] for i in range(1, len(scores))), default=0.0
            )

            # Overall success metric for accuracy
            success = quota_achieved

            explanation_parts = [
                f"Throughput: {achieved_score:.2f}/{expected_score:.2f}" if metric_type == "throughput" else f"Score: {achieved_score:.2f}/{expected_score:.2f}",
                f"Ground Factorio Reward: {ground_truth_score:.2f}",
                f"Proportion: {throughput_proportion:.3f}",
                f"Last change: {last_step_change:+.3f}",
                f"Total change: {total_change:+.3f}",
                f"Steps: {total_steps}",
                f"Success: {success}",
            ]

            if error:
                explanation_parts.append(f"Error: {error}")

            explanation = ", ".join(explanation_parts)

            return Score(
                value=str(
                    throughput_proportion
                ),  # Boolean for accuracy metric, proportion for mean
                answer=str(1) if success else str(throughput_proportion),
                explanation=explanation,
                metadata={
                    # Core metrics
                    "throughput_proportion": throughput_proportion,
                    "achieved_throughput": achieved_score,
                    "ground_truth_score": ground_truth_score,  # Keep separate
                    "last_step_change": last_step_change,
                    # Additional context
                    "expected_score": expected_score,
                    "quota_achieved": quota_achieved,
                    "total_change": total_change,
                    "average_step_change": avg_step_change,
                    "score_per_step": score_per_step,
                    "max_single_step_gain": max_single_gain,
                    "total_steps": total_steps,
                    "has_error": bool(error),
                    "error": error or "",
                    # Trajectory analysis
                    "scores_count": len(scores),
                    "final_10_scores": scores[-10:] if len(scores) > 10 else scores,
                    # Task context
                    "env_id": metadata.get("env_id", "unknown"),
                    "trajectory_length": metadata.get("trajectory_length", 64),
                    # NEW: Metric type and reward tracking
                    "metric_type": metric_type,
                    "throughput_available": trajectory_data.throughput_available,
                    "cumulative_reward": trajectory_data.cumulative_reward or 0.0,
                    "task_override_count": trajectory_data.task_override_count or 0,
                    "reward_type_mix": f"{trajectory_data.task_override_count or 0} task / {total_steps - (trajectory_data.task_override_count or 0)} GT",
                },
            )

        except Exception as e:
            logger.error(f"Error in comprehensive scorer: {e}")
            return Score(
                value=False,
                answer="failure",
                explanation=f"Scorer error: {str(e)}",
                metadata={"scorer_error": str(e)},
            )

    return score


# @scorer(metrics=[accuracy()])
# def binary_success_scorer() -> Scorer:
#     """Simple binary success scorer for Pass@N evaluation"""
#
#     async def score(state: AgentState, target: Target) -> Score:
#         try:
#             trajectory_data = store_as(TrajectoryData)
#             production_score = trajectory_data.final_score or trajectory_data.production_score or 0.0
#             error = trajectory_data.error
#
#             metadata = getattr(state, 'metadata', {}) if hasattr(state, 'metadata') else {}
#             expected_score = metadata.get("expected_production_score", 100.0)
#
#             success = production_score >= expected_score and not error
#
#             return Score(
#                 value=success,
#                 answer="success" if success else "failure",
#                 explanation=f"Binary success: {success} (score: {production_score:.2f}/{expected_score:.2f})",
#                 metadata={
#                     "success": success,
#                     "production_score": production_score,
#                     "expected_score": expected_score,
#                     "quota_achieved": production_score >= expected_score,
#                     "error_occurred": bool(error)
#                 }
#             )
#
#         except Exception as e:
#             logger.error(f"Error in binary success scorer: {e}")
#             return Score(
#                 value=False,
#                 answer="failure",
#                 explanation=f"Scorer error: {str(e)}"
#             )
#
#     return score


@scorer(metrics=[mean()])
def ground_truth_score_tracker() -> Scorer:
    """Track ground truth production score (absolute)"""

    async def score(state: AgentState, target: Target) -> Score:
        try:
            trajectory_data = store_as(TrajectoryData)
            ground_truth_score = trajectory_data.final_ground_truth_score or 0.0
            total_steps = trajectory_data.total_steps or 0

            return Score(
                value=ground_truth_score,
                answer=f"{ground_truth_score:.2f}",
                explanation=f"Ground truth score: {ground_truth_score:.2f} over {total_steps} steps",
                metadata={
                    "ground_truth_score": ground_truth_score,
                    "total_steps": total_steps,
                    "score_per_step": ground_truth_score / total_steps
                    if total_steps > 0
                    else 0,
                    "score_trajectory": trajectory_data.ground_truth_scores[-10:]
                    if trajectory_data.ground_truth_scores
                    else [],
                },
            )
        except Exception as e:
            logger.error(f"Error in ground truth tracker: {e}")
            return Score(
                value=0.0, answer="0.00", explanation=f"Scorer error: {str(e)}"
            )

    return score


@scorer(metrics=[mean()])
def reward_tracker() -> Scorer:
    """Track cumulative reward (task-based or ground truth delta)"""

    async def score(state: AgentState, target: Target) -> Score:
        try:
            trajectory_data = store_as(TrajectoryData)
            cumulative_reward = trajectory_data.cumulative_reward or 0.0
            total_steps = trajectory_data.total_steps or 0
            task_override_count = trajectory_data.task_override_count or 0

            reward_mix = f"{task_override_count} task-based, {total_steps - task_override_count} ground-truth"

            return Score(
                value=cumulative_reward,
                answer=f"{cumulative_reward:.2f}",
                explanation=f"Cumulative reward: {cumulative_reward:.2f} ({reward_mix})",
                metadata={
                    "cumulative_reward": cumulative_reward,
                    "total_steps": total_steps,
                    "task_override_count": task_override_count,
                    "ground_truth_count": total_steps - task_override_count,
                    "avg_reward_per_step": cumulative_reward / total_steps
                    if total_steps > 0
                    else 0,
                    "reward_trajectory": trajectory_data.step_rewards[-10:]
                    if trajectory_data.step_rewards
                    else [],
                },
            )
        except Exception as e:
            logger.error(f"Error in reward tracker: {e}")
            return Score(
                value=0.0, answer="0.00", explanation=f"Scorer error: {str(e)}"
            )

    return score


@scorer(metrics=[mean()])
def reward_type_analyzer() -> Scorer:
    """Analyze proportion of task-based vs ground truth rewards"""

    async def score(state: AgentState, target: Target) -> Score:
        try:
            trajectory_data = store_as(TrajectoryData)
            total_steps = trajectory_data.total_steps or 0
            task_override_count = trajectory_data.task_override_count or 0

            task_proportion = (
                task_override_count / total_steps if total_steps > 0 else 0.0
            )

            return Score(
                value=task_proportion,
                answer=f"{task_proportion:.3f}",
                explanation=f"Task-based reward proportion: {task_proportion:.3f} ({task_override_count}/{total_steps})",
                metadata={
                    "task_proportion": task_proportion,
                    "task_override_count": task_override_count,
                    "ground_truth_count": total_steps - task_override_count,
                    "total_steps": total_steps,
                },
            )
        except Exception as e:
            logger.error(f"Error in reward type analyzer: {e}")
            return Score(
                value=0.0, answer="0.000", explanation=f"Scorer error: {str(e)}"
            )

    return score


# Intermediate scoring functions for real-time trajectory analysis


async def score_step_intermediate(
    state: AgentState,
    step_num: int,
    production_score: float,
    expected_score: float,
    scores_history: List[float],
    ground_truth_score: float = None,
    step_reward: float = None,
    reward_type: str = "unknown",
    measured_throughput: float = None,
    throughput_available: bool = False,
) -> List[Score]:
    """
    Score intermediate step during trajectory execution.
    Returns list of scores for different metrics.
    """
    intermediate_scores = []

    try:
        # 1. Throughput Proportion Score
        # Use measured throughput if available, otherwise fall back to production_score
        score_value = measured_throughput if throughput_available and measured_throughput > 0 else production_score
        metric_label = "Throughput" if throughput_available and measured_throughput > 0 else "Score"

        proportion = (
            min(score_value / expected_score, 1.0) if expected_score > 0 else 0.0
        )
        proportion_score = Score(
            value=proportion,
            answer=f"{proportion:.3f}",
            explanation=f"Step {step_num}: {metric_label} {score_value:.2f}/{expected_score:.2f} items/60s, Ground Factorio Reward: {ground_truth_score:.1f}, Proportion: {proportion:.3f}" if ground_truth_score is not None else f"Step {step_num}: {metric_label} proportion {score_value:.2f}/{expected_score:.2f} = {proportion:.3f}",
            metadata={
                "step": step_num,
                "metric_type": "throughput_proportion",
                "achieved_throughput": score_value,
                "ground_truth_score": ground_truth_score if ground_truth_score is not None else production_score,
                "expected_score": expected_score,
                "proportion": proportion,
                "using_throughput": throughput_available and measured_throughput > 0,
            },
        )
        intermediate_scores.append(proportion_score)

        # 2. Production Score Tracking
        production_score_obj = Score(
            value=production_score,
            answer=f"{production_score:.2f}",
            explanation=f"Step {step_num}: Production score {production_score:.2f}",
            metadata={
                "step": step_num,
                "metric_type": "production_score",
                "production_score": production_score,
                "score_per_step": production_score / step_num if step_num > 0 else 0,
            },
        )
        intermediate_scores.append(production_score_obj)

        # 3. Step Change Tracking (if we have previous scores)
        if len(scores_history) >= 2:
            last_change = scores_history[-1] - scores_history[-2]
            total_change = scores_history[-1] - scores_history[0]
            avg_change = total_change / (len(scores_history) - 1)

            step_change_score = Score(
                value=abs(last_change),  # Use magnitude for scoring
                answer=f"{last_change:+.3f}",
                explanation=f"Step {step_num}: Change {last_change:+.3f}, Total {total_change:+.3f}",
                metadata={
                    "step": step_num,
                    "metric_type": "step_change",
                    "last_step_change": last_change,
                    "total_change": total_change,
                    "average_change": avg_change,
                    "scores_count": len(scores_history),
                },
            )
            intermediate_scores.append(step_change_score)

        # NEW: Add ground truth tracking score
        if ground_truth_score is not None:
            gt_score_obj = Score(
                value=ground_truth_score,
                answer=f"{ground_truth_score:.2f}",
                explanation=f"Step {step_num}: Ground truth {ground_truth_score:.2f}",
                metadata={
                    "step": step_num,
                    "metric_type": "ground_truth_score",
                    "ground_truth_score": ground_truth_score,
                },
            )
            intermediate_scores.append(gt_score_obj)

        # NEW: Add reward tracking score
        if step_reward is not None:
            reward_score_obj = Score(
                value=step_reward,
                answer=f"{step_reward:.2f}",
                explanation=f"Step {step_num}: Reward {step_reward:.2f} ({reward_type})",
                metadata={
                    "step": step_num,
                    "metric_type": "step_reward",
                    "step_reward": step_reward,
                    "reward_type": reward_type,
                },
            )
            intermediate_scores.append(reward_score_obj)

        return intermediate_scores

    except Exception as e:
        logger.error(f"Error in intermediate scoring for step {step_num}: {e}")
        error_score = Score(
            value=0.0,
            answer="error",
            explanation=f"Intermediate scoring error at step {step_num}: {str(e)}",
            metadata={"step": step_num, "error": str(e)},
        )
        return [error_score]


async def apply_intermediate_scoring(
    state: AgentState,
    step_num: int,
    production_score: float,
    expected_score: float,
    scores_history: List[float],
    ground_truth_score: float = None,
    step_reward: float = None,
    reward_type: str = "unknown",
    measured_throughput: float = None,
    throughput_available: bool = False,
):
    """
    Apply intermediate scoring during trajectory execution.
    Uses inspect_ai.scorer.score() function for real-time scoring.
    """
    try:
        # Get intermediate scores for this step
        intermediate_scores = await score_step_intermediate(
            state, step_num, production_score, expected_score, scores_history,
            ground_truth_score, step_reward, reward_type,
            measured_throughput, throughput_available
        )

        # Apply each score using inspect_ai's score function
        # for score_obj in intermediate_scores:
        # Use the score function to record intermediate metrics
        await score(state)

        logger.info(
            f"📊 Step {step_num}: Applied {len(intermediate_scores)} intermediate scores"
        )

    except Exception as e:
        logger.error(f"Error applying intermediate scoring for step {step_num}: {e}")
