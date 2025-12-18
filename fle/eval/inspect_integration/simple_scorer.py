"""Simple scorer for testing Inspect integration."""

from inspect_ai.scorer import scorer, Score, Target, Scorer, accuracy
from inspect_ai.agent import AgentState
from inspect_ai.util import store_as

# Import the typed models from the controlled solver
from fle.eval.inspect_integration.controlled_solver import TrajectoryData


@scorer(metrics=[accuracy()])
def simple_production_score() -> Scorer:
    """Simple scorer function for production evaluation"""

    async def score(state: AgentState, target: Target) -> Score:
        try:
            # Use typed store to get trajectory data
            trajectory_data = store_as(TrajectoryData)

            # Use measured throughput if available, otherwise fall back to ground truth
            if trajectory_data.throughput_available and trajectory_data.final_measured_throughput > 0:
                achieved_throughput = trajectory_data.final_measured_throughput
                metric_type = "throughput"
            else:
                # Fallback to ground truth (for backward compat)
                achieved_throughput = (
                    trajectory_data.final_score or trajectory_data.production_score or 0.0
                )
                metric_type = "ground_truth"

            # Get ground truth separately for display
            ground_truth_score = trajectory_data.final_ground_truth_score or 0.0
            error = trajectory_data.error

            # Get metadata
            metadata = (
                getattr(state, "metadata", {}) if hasattr(state, "metadata") else {}
            )
            expected_score = metadata.get("expected_production_score", 16.0)

            # Calculate success based on throughput achievement
            success = achieved_throughput >= expected_score and not error

            # Build explanation showing both metrics
            if metric_type == "throughput":
                explanation = f"Throughput: {achieved_throughput:.1f}/{expected_score} items/60s, Ground Factorio Reward: {ground_truth_score:.1f}, Success: {success}"
            else:
                explanation = f"Score: {achieved_throughput:.1f}/{expected_score}, Ground Factorio Reward: {ground_truth_score:.1f}, Success: {success}"

            if error:
                explanation += f", Error: {error}"

            return Score(
                value=success,  # Boolean for accuracy metric
                answer="success" if success else "failure",
                explanation=explanation,
            )

        except Exception as e:
            return Score(
                value=False, answer="failure", explanation=f"Scorer error: {str(e)}"
            )

    return score
