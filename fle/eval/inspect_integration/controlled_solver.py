"""Controlled solver that manages 64-step Factorio trajectory execution."""

import logging
import time
import traceback
from typing import List

from pydantic import Field
from inspect_ai.log import transcript
from inspect_ai.solver import solver
from inspect_ai.agent import AgentState
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    ModelOutput,
    get_model,
    ContentImage,
    ContentText,
)
from inspect_ai.util import StoreModel, store_as

from fle.env.gym_env.environment import FactorioGymEnv
from fle.env.gym_env.action import Action
from fle.env.gym_env.observation import Observation
from fle.env.gym_env.observation_formatter import BasicObservationFormatter
from fle.env.gym_env.registry import get_environment_info
from fle.env.utils.controller_loader.system_prompt_generator import (
    SystemPromptGenerator,
)
from fle.eval.inspect_integration.factorio_agent import calculate_production_score
from fle.commons.constants import REWARD_OVERRIDE_KEY

from fle.eval.inspect_integration.simple_server_pool import get_simple_server_pool
from fle.eval.tasks.task_definitions.lab_play.throughput_tasks import THROUGHPUT_TASKS
from fle.agents.llm.parsing import parse_response


import importlib.resources
import gym

logger = logging.getLogger(__name__)


class StepResult(StoreModel):
    """Store model for individual step results"""

    step: int = Field(default=0)
    production_score: float = Field(default=0.0)
    program_length: int = Field(default=0)
    execution_time: float = Field(default=0.0)
    program_content: str = Field(default="")
    program_output: str = Field(default="")


class TrajectoryData(StoreModel):
    """Store model for trajectory tracking data"""

    production_score: float = Field(default=0.0)
    total_steps: int = Field(default=0)
    current_score: float = Field(default=0.0)
    final_score: float = Field(default=0.0)
    scores: List[float] = Field(default_factory=list)
    steps: List[dict] = Field(default_factory=list)  # Using dict for step data
    error: str = Field(default="")

    # NEW: Ground truth tracking
    ground_truth_scores: List[float] = Field(default_factory=list)
    final_ground_truth_score: float = Field(default=0.0)

    # NEW: Reward tracking
    step_rewards: List[float] = Field(default_factory=list)
    cumulative_reward: float = Field(default=0.0)

    # NEW: Reward type metadata
    reward_types: List[str] = Field(default_factory=list)  # "task_override" or "ground_truth_delta"
    task_override_count: int = Field(default=0)

    # NEW: Throughput tracking (task-specific measurement)
    measured_throughputs: List[float] = Field(default_factory=list)
    final_measured_throughput: float = Field(default=0.0)
    throughput_available: bool = Field(default=False)


@solver
def factorio_controlled_solver():
    """Controlled solver that runs exactly 64 Factorio steps with full logging"""

    async def solve(state: AgentState, *args, **kwargs) -> AgentState:
        run_idx = None
        gym_env = None

        try:
            # Get configuration from metadata
            metadata = (
                getattr(state, "metadata", {}) if hasattr(state, "metadata") else {}
            )
            env_id = metadata.get("env_id", "iron_ore_throughput")
            model_name = metadata.get("model", "openai/gpt-4o-mini")
            trajectory_length = metadata.get(
                "trajectory_length", 64
            )  # Full trajectory length

            logger.info(
                f"🚀 Starting controlled 64-step Factorio trajectory for {env_id}"
            )
            logger.info(
                f"🎯 Target: {trajectory_length} steps using model {model_name}"
            )

            # Get server allocation
            pool = await get_simple_server_pool()
            run_idx = await pool.get_run_idx()
            logger.info(f"📡 Allocated server factorio_{run_idx}")

            # Create gym environment
            gym_env: FactorioGymEnv = gym.make(env_id, run_idx=run_idx)
            gym_env.reset()
            logger.info("🎮 Connected to Factorio server")

            # Get task configuration
            env_info = get_environment_info(env_id)
            if not env_info:
                raise ValueError(f"No environment info for {env_id}")

            # task = TaskFactory.create_task(env_info["task_config_path"])

            # Generate system prompt
            generator = SystemPromptGenerator(
                str(importlib.resources.files("fle") / "env")
            )
            base_system_prompt = generator.generate_for_agent(agent_idx=0, num_agents=1)

            # Get task-specific instructions
            task_config = THROUGHPUT_TASKS.get(env_id)
            if task_config:
                goal_description = task_config.goal_description
                quota = task_config.quota
                task_instructions = f"""
## TASK OBJECTIVE
{goal_description}

## SUCCESS CRITERIA
- Produce at least {quota} {env_id.replace("_throughput", "").replace("_", "-")} per 60 in-game seconds
- Build a fully automated production system
- Complete the task within {trajectory_length} trajectory steps

## IMPORTANT NOTES
- You have {trajectory_length} steps to complete this task
- Each step should make meaningful progress toward the goal
- Focus on essential infrastructure first (mining, smelting, power)
- Then build the specific production chain required
"""
            else:
                goal_description = (
                    f"Create an automatic {env_id.replace('_', '-')} factory"
                )
                quota = 16
                task_instructions = f"## TASK OBJECTIVE\n{goal_description}"

            # Combine base instructions with task-specific instructions
            full_system_prompt = f"""{base_system_prompt}

{task_instructions}

Now begin working toward this objective step by step."""

            # Initialize conversation with proper system prompt
            original_user_message = (
                state.messages[0].content
                if state.messages
                else f"Begin task: {goal_description}"
            )

            state.messages = [
                ChatMessageSystem(content=full_system_prompt),
                ChatMessageUser(
                    content=f"{original_user_message}\n\nAnalyze the current game state and begin your first action."
                ),
            ]

            logger.info(
                f"📋 Initialized system prompt: {len(full_system_prompt)} chars"
            )
            logger.info(f"🎯 Task: {goal_description}")
            logger.info(f"📊 Quota: {quota} items per 60 seconds")
            logger.info(f"📈 Starting {trajectory_length}-step controlled execution...")
            logger.info(f"Trajectory length: {trajectory_length} steps")

            # Controlled trajectory execution - WE control the 64 steps
            production_scores = []
            step_results = []

            # NEW: Ground truth and reward tracking
            ground_truth_scores = []
            step_rewards = []
            reward_types = []
            measured_throughputs = []

            for step in range(trajectory_length):
                step_start = time.time()

                try:
                    # Get current observation from Factorio
                    observation: Observation = gym_env.get_observation()
                    obs_formatted = BasicObservationFormatter(
                        include_research=False
                    ).format(observation)

                    #total_game_score  = calculate_production_score(gym_env)

                    # Create step message with current game state
                    current_score = production_scores[-1] if production_scores else 0
                    step_content = f"""## Step {step + 1}/{trajectory_length} - Game State Analysis

Current production score: {current_score:.1f}/{quota}
Progress: {(step / trajectory_length) * 100:.1f}% complete

**Current Game State:**
{obs_formatted.raw_str.replace("\\n", "\n")}

**Next Action Required:**
Analyze the current state and write a Python program using the FLE API to progress toward the production goal. Focus on one specific improvement or setup task."""

                    step_message = ChatMessageUser(content=step_content)
                    state.messages.append(step_message)

                    # Generate response using Inspect's model with reasoning support
                    generation_config = {
                        "max_tokens": 4096,  # More tokens for complex programs
                        "reasoning_effort": "minimal",
                        # "temperature": 0.1
                    }

                    state.output = await get_model().generate(
                        input=state.messages,
                        config=generation_config,
                        # transforms = ['middle-out']
                    )

                    # Log reasoning usage if available
                    if hasattr(state.output, "usage") and hasattr(
                        state.output.usage, "reasoning_tokens"
                    ):
                        logger.info(
                            f"🧠 Step {step + 1}: Used {state.output.usage.reasoning_tokens} reasoning tokens"
                        )

                    # Add model response to conversation
                    state.messages.append(state.output.message)

                    # Extract Python program from the model response
                    program = parse_response(state.output)

                    if not program:
                        raise Exception(
                            "Could not parse program from model response. Be sure to wrap your code in ``` blocks."
                        )

                    logger.info(
                        f"📝 Step {step + 1}: Generated {len(program.code)} char program"
                    )

                    # Execute action in Factorio and capture results
                    action = Action(agent_idx=0, code=program.code)
                    try:
                        obs, reward, terminated, truncated, info = gym_env.step(action)
                    except Exception as ee:
                        logger.warning(f"Environment error: {ee}")
                        state.messages.append(
                            ChatMessageUser(content=f"Environment error: {ee}")
                        )
                        continue

                    # Log execution details
                    logger.info(
                        f"🎮 Step {step + 1}: reward={reward}, terminated={terminated}"
                    )

                    # Get post-execution observation and program output
                    # post_action_observation = gym_env.get_observation()
                    program_output = (
                        info.get("result", "No output captured")
                        if info
                        else "No info available"
                    )

                    # Calculate flows
                    flow = obs["flows"]

                    # Extract ground truth production score from info dict
                    ground_truth_score = info.get("production_score", 0.0)
                    production_score = ground_truth_score  # For backward compatibility with existing code

                    # Extract measured throughput from task verification (if available)
                    measured_throughput = 0.0
                    throughput_available = False

                    if info.get("task_verification"):
                        task_ver = info["task_verification"]
                        if hasattr(task_ver, "meta"):
                            # Look for throughput measurement in meta dict
                            # The key varies by task type (e.g., "iron_ore_per_minute", "rocket_per_minute")
                            for key, value in task_ver.meta.items():
                                if "per_minute" in key or key == REWARD_OVERRIDE_KEY:
                                    measured_throughput = value
                                    throughput_available = True
                                    break

                    # Determine reward type
                    reward_type = "ground_truth_delta"
                    if info.get("task_verification"):
                        task_ver = info["task_verification"]
                        if hasattr(task_ver, "meta") and REWARD_OVERRIDE_KEY in task_ver.meta:
                            reward_type = "task_override"

                    # Track all metrics
                    production_scores.append(ground_truth_score)  # Backward compat
                    ground_truth_scores.append(ground_truth_score)
                    step_rewards.append(reward)
                    reward_types.append(reward_type)
                    measured_throughputs.append(measured_throughput)

                    if not program_output:
                        if not program.code:
                            program_output = (
                                "No code was submitted. Write code in ``` blocks."
                            )
                        else:
                            program_output = "None"
                    # Create comprehensive feedback message
                    prev_ground_truth = ground_truth_scores[-2] if len(ground_truth_scores) >= 2 else 0.0
                    ground_truth_delta = ground_truth_score - prev_ground_truth

                    # Build throughput section if available
                    throughput_section = ""
                    if throughput_available:
                        throughput_section = f"""
**Task Throughput:**
- Measured: {measured_throughput:.2f} items/60s
- Quota: {quota} items/60s
- {"✓ Meeting quota" if measured_throughput >= quota else "✗ Below quota"}
"""

                    feedback_content = f"""## Step {step + 1} Execution Results

**Program Output (STDOUT/STDERR):**
```
{program_output}
```

**Reward Information:**
- Step Reward: {reward:.2f}
- Reward Type: {reward_type.replace('_', ' ').title()}
{throughput_section}
**Ground Factorio Reward (Economic Value):**
- Current: {ground_truth_score:.1f} (was {prev_ground_truth:.1f})
- Delta: {ground_truth_delta:+.1f}

**Flows**
{flow}

Continue to step {step + 2}."""
                    logger.debug(str(obs))
                    # Get updated rendered image after the action
                    updated_image_data_url = obs[
                        "map_image"
                    ]  # get_rendered_image(gym_env)

                    # Create feedback message with both image and text
                    if updated_image_data_url:
                        feedback_message = ChatMessageUser(
                            content=[
                                ContentImage(image=updated_image_data_url),
                                ContentText(text=feedback_content),
                            ]
                        )
                        logger.info(f"🖼️  Step {step + 1}:")
                    else:
                        feedback_message = ChatMessageUser(content=feedback_content)
                        logger.info(f"📝 Step {step + 1}:")

                    state.messages.append(feedback_message)

                    # Apply middle-out transformation to manage context window
                    from fle.eval.inspect_integration.transforms import middle_out
                    
                    # Use a safe token limit (e.g., 1M for Gemini 1.5 Pro, or lower for others)
                    # This can be made configurable via metadata if needed
                    state.messages = middle_out(state.messages, max_tokens=1_000_000)

                    step_time = time.time() - step_start

                    step_result = {
                        "step": step + 1,
                        "production_score": production_score,
                        "program_length": len(program.code),
                        "execution_time": step_time,
                        "program_content": program.code[:200] + "..."
                        if len(program.code) > 200
                        else program.code,
                        "program_output": program_output[:200] + "..."
                        if len(str(program_output)) > 200
                        else str(program_output),
                    }
                    step_results.append(step_result)

                    logger.info(
                        f"✅ Step {step + 1}/{trajectory_length}: Score={production_score:.1f}, Time={step_time:.1f}s"
                    )

                    # Store intermediate progress using typed store
                    trajectory_data = store_as(TrajectoryData)
                    # Backward compatible fields
                    trajectory_data.production_score = ground_truth_score
                    trajectory_data.current_score = ground_truth_score
                    trajectory_data.scores = production_scores

                    # New ground truth tracking
                    trajectory_data.ground_truth_scores = ground_truth_scores
                    trajectory_data.final_ground_truth_score = ground_truth_score

                    # New reward tracking
                    trajectory_data.step_rewards = step_rewards
                    trajectory_data.cumulative_reward = sum(step_rewards)

                    # Reward type metadata
                    trajectory_data.reward_types = reward_types
                    trajectory_data.task_override_count = sum(1 for rt in reward_types if rt == "task_override")

                    # NEW: Throughput tracking
                    trajectory_data.measured_throughputs = measured_throughputs
                    trajectory_data.final_measured_throughput = measured_throughputs[-1] if measured_throughputs else 0.0
                    trajectory_data.throughput_available = any(measured_throughputs)

                    # Existing fields
                    trajectory_data.total_steps = step + 1
                    trajectory_data.steps = step_results

                    # Apply intermediate scoring for real-time metrics tracking
                    try:
                        from fle.eval.inspect_integration.enhanced_scorer import (
                            apply_intermediate_scoring,
                        )

                        await apply_intermediate_scoring(
                            state=state,
                            step_num=step + 1,
                            production_score=production_score,
                            expected_score=quota,
                            scores_history=production_scores,
                            ground_truth_score=ground_truth_score,
                            step_reward=reward,
                            reward_type=reward_type,
                            measured_throughput=measured_throughput,
                            throughput_available=throughput_available,
                        )
                    except Exception as scoring_error:
                        logger.warning(
                            f"Intermediate scoring error at step {step + 1}: {scoring_error}"
                        )

                    # Check for early termination
                    if terminated or truncated:
                        logger.info(
                            f"⚠️ Episode ended early at step {step + 1}: terminated={terminated}, truncated={truncated}"
                        )
                        transcript().info(
                            f"⚠️ Episode ended early at step {step + 1}: terminated={terminated}, truncated={truncated}, score={production_score:.1f}, flows={flow}"
                        )

                        state.complete = True
                        break

                except Exception as step_error:
                    logger.error(f"❌ Step {step + 1} error: {step_error}")
                    feedback_message = ChatMessageUser(
                        content=f"❌ Step {step + 1} error: {step_error}"
                    )
                    state.messages.append(feedback_message)

                    # Continue with next step rather than failing completely
                    step += 1

            # Final results
            final_ground_truth = ground_truth_scores[-1] if ground_truth_scores else 0.0
            final_throughput = measured_throughputs[-1] if measured_throughputs else 0.0

            # Store final results using typed store
            trajectory_data = store_as(TrajectoryData)
            # Backward compatible fields
            trajectory_data.production_score = final_ground_truth
            trajectory_data.final_score = final_ground_truth
            trajectory_data.scores = production_scores

            # New ground truth tracking
            trajectory_data.ground_truth_scores = ground_truth_scores
            trajectory_data.final_ground_truth_score = final_ground_truth

            # New reward tracking
            trajectory_data.step_rewards = step_rewards
            trajectory_data.cumulative_reward = sum(step_rewards) if step_rewards else 0.0

            # Reward type metadata
            trajectory_data.reward_types = reward_types
            trajectory_data.task_override_count = sum(1 for rt in reward_types if rt == "task_override")

            # NEW: Throughput tracking
            trajectory_data.measured_throughputs = measured_throughputs
            trajectory_data.final_measured_throughput = final_throughput
            trajectory_data.throughput_available = any(measured_throughputs)

            # Existing fields
            trajectory_data.total_steps = len(step_results)
            trajectory_data.steps = step_results

            # Set final model output with summary
            state.output = ModelOutput(
                completion=f"Completed {len(step_results)}-step trajectory with final score: {final_ground_truth:.1f}",
                model=model_name,
            )

            logger.info(
                f"🎉 Controlled trajectory complete: {final_ground_truth:.1f} score after {len(step_results)} steps"
            )
            transcript().info(
                f"🎉 Controlled trajectory complete: {final_ground_truth:.1f} score after {len(step_results)} steps"
            )

        except Exception as e:
            error_msg = f"Controlled solver error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)

            # Store error information using typed store
            trajectory_data = store_as(TrajectoryData)
            trajectory_data.error = error_msg
            trajectory_data.production_score = 0.0
            trajectory_data.final_score = 0.0

            state.output = ModelOutput(
                completion=f"Error in controlled trajectory: {error_msg}",
                model=metadata.get("model", "unknown") if metadata else "unknown",
            )

        finally:
            # Clean up resources
            if run_idx is not None:
                try:
                    pool = await get_simple_server_pool()
                    await pool.release_run_idx(run_idx)
                    logger.info(f"🧹 Released server factorio_{run_idx}")
                except Exception as e:
                    logger.error(f"Error releasing server: {e}")

        return state

    return solve
